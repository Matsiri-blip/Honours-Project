import numpy as np
from scipy.optimize import minimize
import pylatexenc
from qiskit.visualization import plot_histogram
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_algorithms import AmplificationProblem
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager # Import transpiler

# Imports from Qiskit Runtime
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import Session, EstimatorV2 as Estimator
from qiskit_ibm_runtime import SamplerV2 as Sampler


from itertools import combinations


##define a function that returns a toal number of edges
def total_edges(n):
  return int((n**2-n)/2)
##define a binary vector using the graph enumeration
def binary_vector(n,k):
    m =total_edges(n)
    binary_vec = np.zeros(m)
    binary = bin(k)[2:]
    for i in range(len(binary)):
        binary_vec[m-i-1] = int(binary[::-1][i])
    return binary_vec
##define edges functions that returns the edges in a graph
def edges(n):
  edges = list(combinations(range(n), 2))
  return edges
def edge_values(n,k):
  binary_vec = binary_vector(n, k)
  edge_list = edges(n)     ##define map edges
  edge_values = {edge: binary_vec[idx] for idx, edge in enumerate(edge_list)}     ##define map edges
  return edge_values
## create a function a function that determines the total energy of a graph.
def clique_count(n, s, r, k):
    binary_vec = binary_vector(n, k)
    graph_edges = edges(n)                                                            ##define the total edges
    edge_value = edge_values(n,k)                                                ##define map edges
    objective = 0
    penalty = 1

    # Count the total number of Blue cliques
    for edge in combinations(range(n), s):
        blue_edges = [(min(i, j), max(i, j)) for i, j in combinations(edge, 2)]
        if all(edge_value.get(graph_edge, 0) == 1 for graph_edge in blue_edges):     ##if the edges selected are equal to 1 then add penalty to objective
            objective += penalty

    # Count the total number of Red cliques
    for edge in combinations(range(n), r):
        red_edges = [(min(i, j), max(i, j)) for i, j in combinations(edge, 2)]
        if all(edge_value.get(graph_edge, 0) == 0 for graph_edge in red_edges):      ##if the edges selected are equal to 0 then add penalty to objective
            objective += penalty

    return objective # Return the calculated objective directly

n,b,r = 4,3,3
m = total_edges(n)
N = 2**m
INPUT_LIST = [clique_count(n,b,r,i) for i in range(N)]
# Classical check for minimum value for verification
Y_ENERGIES = np.array(INPUT_LIST)
MIN_ENERGY = np.min(Y_ENERGIES)
good_indices = [k for k, E_k in enumerate(Y_ENERGIES) if E_k == MIN_ENERGY]
marked_states = [format(k, f'0{m}b') for k in good_indices]  # Binary strings for good states

def is_good_state_check(bitstring): ##takes good states from classical minimum
    k = int(bitstring, 2)
    return k < N and INPUT_LIST[k] == MIN_ENERGY
##oracle function 
def grover_oracle(marked_states):
    if not isinstance(marked_states, list):
        marked_states = [marked_states]

    num_qubits = len(marked_states[0])  # Use m (length of bitstring), not total_edges
    qc = QuantumCircuit(num_qubits)
    for target in marked_states:
        rev_target = target[::-1]  # Flip to match Qiskit bit-ordering
        zero_inds = [ind for ind in range(num_qubits) if rev_target.startswith("0", ind)]
        if zero_inds:
            qc.x(zero_inds)
        qc.compose(MCMTGate(ZGate(), num_qubits - 1, 1), qubits=list(range(num_qubits)), inplace=True)
        if zero_inds:
            qc.x(zero_inds)
    return qc

# Create oracle and problem
oracle_circuit = grover_oracle(marked_states)  # Use oracle
state_preparation = QuantumCircuit(m)
state_preparation.h(range(m))  # Uniform superposition
problem = AmplificationProblem(
    oracle=oracle_circuit,
    is_good_state=is_good_state_check,
    state_preparation=state_preparation
)


M = len(good_indices)  # Number of solutions
optimal_num_iterations = max(1, int(np.pi / 4 * np.sqrt(N/M)))  # π/4 * √(N/M)


# Build the circuit
grover_op = grover_operator(oracle=oracle_circuit, state_preparation=state_preparation)
qc = QuantumCircuit(grover_op.num_qubits)
qc.h(range(grover_op.num_qubits))  # Initial superposition
qc.compose(grover_op.power(optimal_num_iterations), inplace=True)
qc.measure_all()  # Add classical register 

backend = AerSimulator()
sampler = Sampler(mode= backend)  
sampler.options.default_shots = 10000

# Transpile the circuit for the backend
target = backend.target
pm = generate_preset_pass_manager(target=target, optimization_level=3)
circuit_isa = pm.run(qc)


# Run the circuit
result = sampler.run([circuit_isa]).result()


# Get and print measurement results
dist = result[0].data.meas.get_counts()  # Extract counts from PubResult
#print("Measurement Results:", dist)



def adjacency_matrix(n,k):
    m = total_edges(n)
    binary_vec =binary_vector(n,k)

    A = np.zeros((n,n))
    edge_index = 0
    for i in range(n):
        for j in range(i + 1, n):
            A[i][j] = binary_vec[edge_index]
            A[j][i] = binary_vec[edge_index] # Ensure symmetry
            edge_index += 1
    return A

def Graph(n,k):

    # Create a new figure and axes
    fig, ax = plt.subplots()
    A =adjacency_matrix(n,k)
    x = -np.cos(2 * np.pi * np.arange(n) / n+np.pi/2)       ##points where the vertices will be placed
    y = np.sin(2 * np.pi * np.arange(n) / n+np.pi/2)         ## these points are around a circle
    for i in range(n):
        ax.scatter(x[i],y[i], label =f"{i+1}",s=400, color = 'black', zorder=2)  ##drawing vertices
        ax.text(x[i],y[i],"$v_{{{}}}$".format(i+1),fontsize = 15,color = 'white', horizontalalignment='center', verticalalignment='center')
    for i in range(n):                                                         ##looping through the vertices
        for j in range(n):
            if A[i][j]==1:                                                            ## check if there is an edge between the edges using adjacency matrix defined
                ax.plot([x[i],x[j]],[y[i],y[j]],color = 'blue', linewidth = 3,zorder=1)

            else:
                ax.plot([x[i],x[j]],[y[i],y[j]],color = 'red',linewidth = 3, zorder=1)

    ax.text(0,0,f'{k}',fontsize = 20, horizontalalignment='center', verticalalignment='center')
    ax.axis('off')
