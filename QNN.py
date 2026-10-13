from copy import deepcopy

from qiskit import QuantumCircuit, generate_preset_pass_manager
from qiskit_machine_learning.algorithms import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.algorithms.classifiers import VQC
import numpy as np
from qiskit.primitives import Sampler
from qiskit_aer.primitives import Sampler as AerSampler
from sklearn.base import BaseEstimator
from loguru import logger
from qiskit_aer import AerSimulator

seed = 123
algorithm_globals.random_seed = seed

options = {'seed': 12345, 'shots': 4096}


def construct_qnn(
        feature_map,
        ansatz,
        callback_graph=None,
        maxiter=100,
        interpret=None,
        output_shape=2,
        standard_reuploading=False
) -> VQC:
    device = "CUDA" if "GPU" in AerSimulator().available_devices() else "CPU"
    sim = AerSimulator(method='statevector', device=device)
    # logger.info(f'Run on: {AerSimulator().available_devices()}')
    if device == "CUDA":
        logger.info('Run with GPU acceleration')
        sampler = AerSampler(
            backend_options={
                "method": "statevector",
                "device": "GPU"
            },
            run_options=options
        )
    else:
        sampler = Sampler(options=options)


    if not standard_reuploading:
        if ansatz is None:
            assert "Ansatz cannot be None if standard_reuploading is False"
        qc = QuantumCircuit(ansatz.num_qubits, ansatz.num_clbits)
        qc.compose(feature_map, inplace=True)
        qc.compose(ansatz, inplace=True)
    else:
        logger.info('Using standard reuploading')
        qc = QuantumCircuit(feature_map.num_qubits, feature_map.num_clbits)
        qc.compose(feature_map, inplace=True)

    feature_params = [p for p in qc.parameters if p.name.startswith("x")]
    weight_params = [p for p in qc.parameters if p not in feature_params]

    initial_point = np.asarray([0.5] * len(weight_params))

    nn = SamplerQNN(
        sampler=sampler,
        circuit=qc,
        weight_params=weight_params,
        input_params=feature_params,
        interpret=interpret,
        output_shape=output_shape,
        input_gradients=False
    )
    classifier = NeuralNetworkClassifier(
        neural_network=nn,
        optimizer=COBYLA(maxiter=maxiter),
        callback=callback_graph,
        initial_point=initial_point,
        loss='cross_entropy',
        one_hot=True
    )

    return classifier

class OVA_Classifier():
    def __init__(
            self,
            feature_map,
            ansatz,
            maxiter,
            callback_graph=None,
            interpret=None,
            output_shape=2,
            standard_reuploading=False
    ):
        self.feature_map = feature_map
        self.ansatz = ansatz
        self.callback_graph = callback_graph
        if maxiter is None:
            self.maxiter = 1000
        else:
            self.maxiter = maxiter

        if ansatz is not None and not standard_reuploading:
            assert feature_map.num_qubits == ansatz.num_qubits, ""
            "Feature map and ansatz must have the same number of qubits "
            "if standard_reuploading is False"

        self.models = {i: construct_qnn(
            feature_map=deepcopy(feature_map),
            ansatz=deepcopy(ansatz),
            callback_graph=deepcopy(callback_graph),
            maxiter=deepcopy(maxiter),
            interpret=deepcopy(interpret),
            output_shape=2,
            standard_reuploading=standard_reuploading
        ) for i in range(output_shape)
        }

        self.output_shape = output_shape

    def fit(self, X, y):
        for key, model in self.models.items():
            logger.info(f'Training model for class {key}...')
            y_new = (y == key).astype(int)
            model.fit(X, y_new)

        return self

    def predict(self, X):
        scores = []
        for cls, model in self.models.items():
            proba = model.predict_proba(X)[:, 1]
            scores.append(proba)
        scores = np.vstack(scores).T
        return np.argmax(scores, axis=1)

