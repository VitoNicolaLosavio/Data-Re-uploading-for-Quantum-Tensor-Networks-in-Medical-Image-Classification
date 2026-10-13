from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes, TwoLocal
from qiskit.circuit import ParameterVector

def _generate_tree_tuples(n):
    """
    Generate a list of tuples representing the tree structure
    of consecutive numbers up to n.

    Args:
        n (int): The number up to which the tuples are generated.

    Returns:
        list: A list of tuples representing the tree structure.
    """
    tuples_list = []
    indices = []

    # Generate initial tuples with consecutive numbers up to n
    for i in range(0, n, 2):
        tuples_list.append((i, i + 1))

    indices += [tuples_list]

    # Perform iterations until we reach a single tuple
    while len(tuples_list) > 1:
        new_tuples = []

        # Generate new tuples by combining adjacent larger numbers
        for i in range(0, len(tuples_list), 2):
            new_tuples.append((tuples_list[i][1], tuples_list[i + 1][1]))

        tuples_list = new_tuples
        indices += [tuples_list]

    return indices


def TTN(num_qubits, prefix='λ', **kwargs):
    """
    Constructs a Tree Tensor Network (TTN) quantum circuit.

    Args:
        num_qubits (int): The number of qubits in the circuit.
        prefix (str): The prefix for the parameter names.
        **kwargs: Additional keyword arguments to be passed to the
        RealAmplitudes.

    Returns:
        QuantumCircuit: The constructed TTN quantum circuit.

    Raises:
        AssertionError: If the number of qubits is not a power of 2
        or zero.
    """
    qc = QuantumCircuit(num_qubits)

    # Compute qubit indices
    assert num_qubits & (
            num_qubits -
            1) == 0 and num_qubits != 0, "Number of qubits must be a power of 2"

    indices = _generate_tree_tuples(num_qubits)

    # Iterate over each layer of TTN indices
    for layer_indices in indices:
        for i, j in layer_indices:
            qc.compose(RealAmplitudes(num_qubits=2,
                                      parameter_prefix=f'{prefix}_{i},{j}',
                                      **kwargs), [i, j],
                       inplace=True)
        qc.barrier(
        )  # Add a barrier after each layer for clarity and separation

    return qc


def encoding_features_h_ry(num_qubits):
    qc = QuantumCircuit(num_qubits)
    feature_params = ParameterVector('x', num_qubits)

    for i in range(num_qubits):
        qc.h(i)
        qc.ry(feature_params[i], i)

    return qc


def encoding_features_ry(num_qubits):
    qc = QuantumCircuit(num_qubits)
    feature_params = ParameterVector('x', num_qubits)

    for i in range(num_qubits):
        qc.ry(feature_params[i], i)

    return qc

def partial_encoding(num_qubits):
    qc = QuantumCircuit(num_qubits)
    qubits = list(range(num_qubits))
    feature_params = ParameterVector('x', (num_qubits - 1) * 2)
    idx = 0
    for q1, q2 in zip(qubits[:-1], qubits[1:]):
        qc.ry(feature_params[idx], q1)
        idx += 1
        qc.ry(feature_params[idx], q2)
        idx += 1

        qc.compose(
            TwoLocal(
                2,
                rotation_blocks='ry',
                entanglement='reverse_linear',
                entanglement_blocks='cx',
                parameter_prefix=f'θ_{idx / 2}',
                insert_barriers=True,
                reps=1
            ).decompose(),
            qubits=[q1, q2],
            inplace=True
        )

        qc.cx(q2, q1)
        qc.barrier()

    return qc


def tensor_ring(num_qubits, prefix='θ', **kwargs):
    """
    Constructs a Full Entanglement Tensor Ring quantum circuit.

    Args:
        num_qubits (int): The number of qubits in the circuit.
        **kwargs: Additional keyword arguments to be passed to the
        RealAmplitudes.

    Returns:
        QuantumCircuit: The constructed MPS quantum circuit.

    """
    qc = QuantumCircuit(num_qubits)
    qubits = range(num_qubits)

    # Iterate over adjacent qubit pairs
    for i, j in zip(qubits[:-1], qubits[1:]):
        qc.compose(
            RealAmplitudes(num_qubits=2,
                           parameter_prefix=f'{prefix}_{i},{j}',
                           **kwargs), [i, j],
            inplace=True)
        qc.barrier(
        )  # Add a barrier after each block for clarity and separation

    qc.compose(RealAmplitudes(num_qubits=2,
                              parameter_prefix=f'{prefix}_{num_qubits - 1},{0}',
                              **kwargs), [num_qubits - 1, 0],
               inplace=True)
    qc.barrier(
    )

    return qc


def construct_tensor_ring_ansatz_circuit(num_qubits, measure=False, prefix_tn='λ',
                                         prefix_tr='θ') -> QuantumCircuit:
    # Function for the construction of the TR+TTN ansatz
    ansatz = QuantumCircuit(num_qubits, 1)

    ttn = TTN(num_qubits, prefix=prefix_tn, reps=1).decompose()
    tr = tensor_ring(num_qubits, prefix=prefix_tr, reps=1).decompose()

    ansatz.compose(tr, range(num_qubits), inplace=True)
    ansatz.compose(ttn, range(num_qubits), inplace=True)

    if measure:
        ansatz.measure(num_qubits - 1, 0)

    return ansatz


def standard_re_uploading(num_qubits):
    qc = QuantumCircuit(num_qubits, 1)
    feature_params = ParameterVector('x', num_qubits)

    for i in range(num_qubits):
        qc.ry(feature_params[i], i)

    tn = construct_tensor_ring_ansatz_circuit(num_qubits)

    qc.compose(tn, inplace=True)
    for i in range(num_qubits):
        qc.rx(feature_params[i], i)

    tn = construct_tensor_ring_ansatz_circuit(num_qubits, prefix_tn='φ', prefix_tr='ψ_mps')
    qc.compose(tn, inplace=True)

    qc.measure(num_qubits - 1, 0)
    return qc
