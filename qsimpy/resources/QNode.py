import simpy
import os
import re
from simpy.core import Environment
from qsimpy.utils.Log import Log
from qsimpy.tasks.TaskStatus import TaskStatus
from typing import Optional
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit import QuantumCircuit

class QNode(simpy.Resource):
    """A quantum node that can execute quantum tasks. This class inherits from simpy.Resource.
    Attributes:
        - id (int): The ID of the quantum node.
        - qubit_number (int): The number of qubits available on the quantum node.
        - quantum_volume (int): The quantum volume of the quantum node.
        - clops (int): The clops of the quantum node.
        - d1cps (int): The deep-1 circuit operation per second of the quantum node.
        - env (simpy.Environment): The simulation environment.
        - waiting_tasks (list): The list of tasks waiting to be executed on the quantum node.
        - current_tasks (dict): The dictionary of tasks currently being executed on the quantum node.
    """

    def __init__(
        self,
        env,
        id: Optional[int] = None,
        qubit_number: Optional[int] = None,
        quantum_volume: Optional[int] = None,
        clops: Optional[int] = None,
        capacity=1,
        error:Optional[dict] = None
    ):
        """
        Initialize a quantum node.
        Args:
            - env (simpy.Environment): The SimPy simulation environment.
            - id (int or None): The ID of the quantum node. If None, a unique ID will be generated.
            - qubit_number (int): The number of qubits available on the quantum node.
            - quantum_volume (int): The quantum volume of the quantum node.
            - clops (int): The clops rating of the quantum node.
            - capacity (int): The maximum number of tasks that can be executed on the quantum node simultaneously.
            - error (dict): The different error rates of the quantum node

        """
        super().__init__(env, capacity=capacity)
        # If no ID is provided, generate a unique ID
        if id is None:
            self.id = id(self)
        else:
            self.id = id
        self.env = env
        self.qubit_number = qubit_number
        self.quantum_volume = quantum_volume
        self.clops = clops
        self.d1cps = 0
        self.waiting_tasks = []
        self.current_tasks = {}
        self.next_available_time = -1
        self.completed_tasks = []
        self.failed_tasks = []
        self.total_busy_time = 0
        self.qiskit_backend = None
        self.qnode_name = None
        self.qnode_model = None
        self.error = error

    def request(self, task=None):
        """
        Request the quantum node to execute a task.
        """
        # If a task is provided, add it to the waiting_tasks list
        if task:
            self.waiting_tasks.append(task)
        req = super().request()
        # Once the request is processed, remove the task from the waiting_tasks list
        if task:
            self.env.process(self._remove_from_queue(req, task))

        return req

    def _remove_from_queue(self, req, task):
        """
        Remove a task from the waiting_tasks list once the request is processed.
        """
        yield req
        self.waiting_tasks.remove(task)

    def process_task(self, task):
        """
        Process a quantum task
        """
        task.set_status(TaskStatus.RUNNING)
        task.start_running_time = self.env.now
        task.waiting_time = task.start_running_time - task.arrival_time

        execution_time = (task.get_circuit_layers() / self.d1cps) * task.shots
        self.current_tasks[task] = self.env.now  # Set the start time for the task

        Log.print_with_current_time(
            self.env.now,
            f"⌛️ QNode {self.id} started executing QTask {task.id} with {task.get_circuit_layers()} layers for {execution_time} seconds",
        )
        yield self.env.timeout(execution_time)
        self.total_busy_time += execution_time
        del self.current_tasks[task]  # Remove the task
        task.set_status(TaskStatus.DONE)
        task.execution_time = execution_time
        Log.print_with_current_time(
            self.env.now, f"✅ QNode {self.id} finished executing QTask {task.id}"
        )
        self.completed_tasks.append(task)

    def simulate_task(self, task):
        transpiled_cl = task.qtask_data[self.qnode_model]["depth"]  # TODO: + 100k just for testing, remove for production
        Log.print_success(
            f"🔸 QTask {task.id}: Transpiling circuit from {task.circuit_layers} layer to {transpiled_cl} layers to adapt with {self.qnode_name} backend ({self.qubit_number} qubits, CLOPS {self.clops})",
        )
        self.completed_tasks.append(task)
        task.set_status(TaskStatus.DONE)
        return transpiled_cl

    def emulate_task(self, task):
        algo = task.qtask_data["algorithm"]
        qubit = task.qubit_number
        pattern = rf"{algo}_indep_qiskit_{qubit}"  # Regex pattern to match the desired files
        folder = "./qdataset/qasm/all"  # Folder where QASM files are located
        file_list = os.listdir(folder)  # List all files in the folder
        task.set_status(TaskStatus.VALIDATING)  # Set task status to VALIDATING
        # Initialize the transpiler with the specified backend and options
        pm_sim = generate_preset_pass_manager(backend=self.qiskit_backend, optimization_level=0)
        qasm_path = None  # Initialize qasm_path to None

        # Loop through the sorted list of files in the folder
        for file_name in sorted(file_list):
            if file_name.endswith(".qasm"):
                match = re.search(pattern, file_name)  # Search for the pattern in the filename
                if match:
                    qasm_path = os.path.join(folder, file_name)  # If a match is found, set qasm_path
                    break  # Break the loop once a valid QASM file is found

        # Check if a valid QASM file was found
        if qasm_path:
            task.qasm_file = qasm_path
            # Create a QuantumCircuit object from the QASM file
            circuit = QuantumCircuit.from_qasm_file(task.qasm_file)
            # Update the task's circuit attribute
            task.circuit = circuit
            # Transpile the circuit for the specified backend
            transpiled_circuit= pm_sim.run(task.circuit)
            # Update the task with the transpiled circuit and its layers
            transpiled_cl, _ = task.update_transpiled_circuit(transpiled_circuit)
            task.execution_time=(task.get_circuit_layers() / self.d1cps) * task.shots
            # Log the success of the transpiling process
            Log.print_success(
                f"🔸 QTask {task.id}: Transpiling circuit from {task.circuit_layers} layer to {transpiled_cl} layers to adapt with {self.qnode_name} backend ({self.qubit_number} qubits, CLOPS {self.clops}, Execution Time:{task.execution_time})"
            )
            self.completed_tasks.append(task)
            task.set_status(TaskStatus.DONE)
            return transpiled_cl
        else:
            # Handle the case where no valid QASM file was found
            Log.print_error(f"No matching QASM file found for task {task.id}")
            task.set_status(TaskStatus.ERROR)
            return None

    def transpile_task(self, task, mode: str):
        """
        Transpile a quantum task directly using QASM file or used pre-transpiled data from qtask_data
        """
        if mode == "emulate":
            return self.emulate_task(task)
        elif mode == "simulate":
            if self.qnode_model is not None:
                return self.simulate_task(task)
            else:
                raise ValueError("QNode model is None")
        else:
            raise ValueError("Invalid mode. Choose 'emulate' or 'simulate'.")

    def get_waiting_queue_ids(self):
        """Get the IDs of tasks in the waiting queue."""
        return [task.id for task in self.waiting_tasks]

    def get_waiting_queue(self):
        """Get the quantum tasks in the waiting queue."""
        waiting_queue = []
        for task in self.waiting_tasks:
            execution_time = self.get_estimated_execution_time(task,self.mode)
            waiting_queue.append((task.id, execution_time))
        return waiting_queue

    def get_estimated_execution_time(self, task, mode="simulate"):
        """Get the estimated execution time for a task."""
        if mode=="simulate":
            return (task.get_circuit_layers() / self.d1cps) * task.shots
        elif mode=="emulate":
            return task.execution_time

    def get_estimated_waiting_time(self, qtask, mode="simulate"):
        """Get the estimated waiting time for the quantum node."""
        if self.next_available_time > qtask.arrival_time:
            estimated_waiting_time = self.next_available_time - qtask.arrival_time
            self.next_available_time += self.get_estimated_execution_time(qtask,mode)
        else:
            estimated_waiting_time = 0
            self.next_available_time = (
                qtask.arrival_time + self.get_estimated_execution_time(qtask,mode)
            )
        return estimated_waiting_time

    def get_utilization(self, total_simulation_time):
        return self.total_busy_time / total_simulation_time

    def get_fidelity_score(self, task=None):
        """
        Calculate fidelity score based on error rates.
        Returns a probability of success (0-1) based on hardware error rates and task complexity.

        Args:
            task: QTask object (optional). If provided, calculates task-specific fidelity.

        Returns:
            float: Fidelity score (0-1), where 1.0 is perfect fidelity.
        """
        if self.error is None:
            return 1.0  # Assume perfect fidelity if no error data

        # Extract average error rates from the error dictionary
        readout_error = self.error.get("Readout_assignment_error", 0.0)
        gate_error = self.error.get("ID_error", 0.0)

        # Handle negative values (data artifacts) by taking absolute value
        readout_error = abs(readout_error)

        # Base fidelity calculation (simplified model)
        # Success probability = (1 - readout_error) for measurements
        base_fidelity = 1.0 - readout_error

        # If task is provided, account for circuit depth
        if task is not None:
            circuit_layers = task.get_circuit_layers()
            qubit_count = task.qubit_number

            # Estimate total gates (simplified: layers * qubits)
            # Each gate has error probability, accumulates multiplicatively
            # P_success ≈ (1 - gate_error)^num_gates * (1 - readout_error)^num_qubits
            num_gates = circuit_layers * qubit_count
            gate_fidelity = (1.0 - gate_error) ** num_gates
            readout_fidelity = (1.0 - readout_error) ** qubit_count

            return gate_fidelity * readout_fidelity

        return base_fidelity

    def get_average_gate_error(self):
        """
        Get the average gate error rate of the quantum node.

        Returns:
            float: Average gate error rate.
        """
        if self.error is None:
            return 0.0
        return self.error.get("ID_error", 0.0)

    def get_state(self):
        """Get the state of the quantum node."""
        return {
            "id": self.id,
            "qubit_number": self.qubit_number,
            "quantum_volume": self.quantum_volume,
            "clops": self.clops,
            "d1cps": self.d1cps,
            "next_available_time": self.next_available_time,
            "fidelity": self.get_fidelity_score(),
        }
