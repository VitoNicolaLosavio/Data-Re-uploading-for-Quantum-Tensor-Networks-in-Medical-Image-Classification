import time
import random

from Utility import update_results_csv
from circuits import *
from preprocessing_data import *
from QNN import *
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

import keras
import pandas as pd
from loguru import logger
from medmnist import INFO

torch.manual_seed(123)
keras.utils.set_random_seed(812)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed_value = 12345

algorithm_globals.random_seed = seed_value
np.random.seed(seed_value)
random.seed(seed_value)

n_qubits = 4
all_output_shape = [4, 6, 8, 9]

all_n_examples = [50, 100, 200, 300]

data_flag = 'pathmnist'

# chose between this feature map strategy: ry, h-ry, partial re-uploading, standard re-uploading
strategy = 'standard re-uploading'


if strategy == 'partial re-uploading':
    n_features = (n_qubits - 1) * 2
else:
    n_features = n_qubits

info = INFO[data_flag]
n_classes = len(info['label'])


def parity(x: int) -> int:
    return x % 2


maxiter = 1000
n_epochs_ae = 100

results = pd.DataFrame(columns=[
    'Model',
    'Train Accuracy',
    'Test Accuracy',
    'N. Classes',
    'N. Examples',
    'Seed'
]
)

valid_strategies = [
            'ry',
            'h-ry',
            'partial re-uploading',
            'standard re-uploading',
        ]

for idx, output_shape in enumerate(all_output_shape):
    logger.info(f'Running experiments for {output_shape} classes...')
    for n_examples in all_n_examples:

        assert strategy in valid_strategies, (
            f"Invalid strategy '{strategy}'. "
            f"Expected one of: {valid_strategies}"
        )

        X_train, X_test, y_train, y_test = data_load_and_process_medmnist(
            data_flag=data_flag,
            num_classes=output_shape,
            all_samples=False,
            seed=seed_value,
            n_features=n_features,
            num_examples_per_class=n_examples,
            margin=.2,
            epochs=n_epochs_ae,
            batch_size=100,
            device=device,
            alpha=1.0
        )

        ansatz = construct_tensor_ring_ansatz_circuit(n_qubits, measure=True)

        # if we use standard re-uploading
        use_standard = False
        if strategy == 'h-ry':
            feature_map = encoding_features_h_ry(num_qubits=n_qubits)

        elif strategy == 'partial re-uploading':
            feature_map = partial_encoding(num_qubits=n_qubits)

        elif strategy == 'standard re-uploading':
            feature_map = standard_re_uploading(n_qubits)
            ansatz = None
            use_standard = True
        else:
            # as default we set standard angle encoding
            feature_map = encoding_features_ry(num_qubits=n_qubits)

        qnn = OVA_Classifier(
            feature_map=feature_map,
            ansatz=ansatz,
            maxiter=maxiter,
            interpret=parity,
            output_shape=output_shape,
            standard_reuploading=use_standard
        )
        start = time.time()
        qnn.fit(X_train, y_train)
        end = time.time()

        mean_time = (end - start) / output_shape
        m, s = divmod(mean_time, 60)
        h, m = divmod(m, 60)
        logger.info(
            f'Mean training time for {strategy} OVA with {output_shape} classes '
            f'and {n_examples} examples: {int(h)}h {int(m)}m {int(s)}s'
        )

        y_pred = qnn.predict(X_train)
        acc_train = accuracy_score(y_train, y_pred)
        f1_train = f1_score(y_train, y_pred, average='weighted')
        acc_train = round(acc_train, 4) * 100

        y_pred = qnn.predict(X_test)
        acc_test = accuracy_score(y_test, y_pred)
        bal_acc_test = balanced_accuracy_score(y_test, y_pred)
        f1_test = f1_score(y_test, y_pred, average='weighted')
        acc_test = round(acc_test, 4) * 100

        results.loc[len(results)] = [
            f'{strategy} OVA', acc_train, acc_test,
            output_shape, n_examples, seed
        ]

        logger.info(
            f'Accuracy for {strategy} OVA with {output_shape} classes '
            f'and {n_examples} examples: Train={acc_train}%, Test={acc_test}%'
        )


        logger.info(
            f'F1 Score for {strategy} OVA with {output_shape} classes '
            f'and {n_examples} examples: Train={round(f1_train, 4) * 100}%, '
            f'Test={round(f1_test, 4) * 100}%'
        )


    logger.info(f'Completed experiments for {output_shape} classes')


update_results_csv(
    results,
    key_columns=['Model','N. Classes','N. Examples', 'Seed'],
    csv_path="./results/results.csv",
)
