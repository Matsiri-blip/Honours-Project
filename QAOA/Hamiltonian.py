from classical_algorithm import *
import numpy as np
from itertools import combinations
from qiskit.quantum_info import SparsePauliOp, Pauli


def get_ramsey_hamiltonian(num_vertices: int, r_sized_vertices: int, b_sized_vertices: int):

    edges = list(combinations(range(num_vertices), 2))
    edge_map = {edge: idx for idx, edge in enumerate(edges)}
    num_qubits = total_edges(num_vertices)
  
    # Initialize the total Hamiltonian using from_list
    H_total = SparsePauliOp.from_list([("I" * num_qubits, 0.0)])

     # 1. Penalty for Red Cliques (Size r): P_Red = Product_{e in K_r} (I + Z_e)/2
  
     for subset_v in combinations(range(num_vertices), r_sized_vertices):
        clique_edges = [(min(i, j), max(i, j)) for i, j in combinations(subset_v, 2)]

        # Start the projector for this specific r-clique with the Identity operator
        projector_r = SparsePauliOp.from_list([("I" * num_qubits, 1.0)])

        for edge in clique_edges:
            qubit_idx = edge_map[edge]

            # (I + Z)/2 is the factor that penalizes Red (|0>)
            # Represent as SparsePauliOp from list
            I_term = SparsePauliOp.from_list([('I' * num_qubits, 0.5)])
            Z_label = ['I'] * num_qubits
            Z_label[qubit_idx] = 'Z'
            Z_term = SparsePauliOp.from_list([("".join(Z_label), 0.5)])

            # The factor for this edge: (I + Z_e)/2
            factor = I_term + Z_term

            # Perform the multiplication and simplification
            projector_r = projector_r.compose(factor, front=True).simplify()

        # Add the completed r-clique projector to the total Hamiltonian
        H_total += projector_r

    # ----------------------------------------------------------------------
    # 2. Penalty for Blue Cliques (Size s): P_Blue = Product_{e in K_s} (I - Z_e)/2
    # ----------------------------------------------------------------------
    for subset_v in combinations(range(num_vertices), b_sized_vertices):
        clique_edges = [(min(i, j), max(i, j)) for i, j in combinations(subset_v, 2)]

        # Start the projector for this specific s-clique
        projector_s = SparsePauliOp.from_list([("I" * num_qubits, 1.0)])

        for edge in clique_edges:
            qubit_idx = edge_map[edge]

            # (I - Z)/2 is the factor that penalizes Blue (|1>)
            I_term = SparsePauliOp.from_list([('I' * num_qubits, 0.5)])
            Z_label = ['I'] * num_qubits
            Z_label[qubit_idx] = 'Z'
            Z_term = SparsePauliOp.from_list([("".join(Z_label), -0.5)]) # Note the negative coefficient

            # The factor for this edge: (I - Z_e)/2
            factor = I_term + Z_term

            # Perform the multiplication and simplification
            projector_s = projector_s.compose(factor, front=True).simplify()

        # Add the completed s-clique projector to the total Hamiltonian
        H_total += projector_s

    # ----------------------------------------------------------------------
    # 3. Separate the Identity Term (Shift)
    # ----------------------------------------------------------------------
    H_total = H_total.simplify()

    total_shift = 0.0
    identity_pauli = Pauli('I' * num_qubits)
    new_pauli_list = []
    new_coeffs = []

    for pauli, coeff in zip(H_total.paulis, H_total.coeffs):
        # Compare Pauli objects directly
        if pauli == identity_pauli:
             total_shift += coeff.real
        else:
            new_pauli_list.append(pauli)
            new_coeffs.append(coeff.real) # Ensure coeffs are real for the final operator


    #  return the identity operator
    if not new_pauli_list:
        ramsey_op = SparsePauliOp.from_list([("I" * num_qubits, 0.0)]) # Return zero operator if only identity remains
    else:
        ramsey_op = SparsePauliOp(new_pauli_list, coeffs=new_coeffs)

    return ramsey_op, total_shift
