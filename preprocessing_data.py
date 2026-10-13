from loguru import logger

import tensorflow as tf
import torch

import numpy as np

from ClassicModels import TripletAutoencoder, train_triplet_autoencoder, extract_embeddings

from torch.utils.data import DataLoader, Subset
import medmnist
from medmnist import INFO
import open_clip
from tqdm.notebook import tqdm

from sklearn.preprocessing import MinMaxScaler

# Load BiomedCLIP
logger.info('Loading BiomedCLIP transformer')
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
)


def data_load_and_process_medmnist(
        data_flag,
        num_classes,
        seed,
        all_samples=False,
        num_examples_per_class=None,
        n_features=8,
        epochs=300,
        margin=0.2,
        alpha=1.0,
        batch_size=64,
        device='cpu',
):
    # Step 0 - Reproducibility
    # Fix random seeds across TensorFlow, NumPy, and PyTorch so that dataset
    # subsampling and model training can be reproduced consistently.
    if seed is not None:
        tf.random.set_seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    # BiomedCLIP is used only as a frozen visual feature extractor.
    clip_model.to(device).eval()

    # Step 1 - Load the official MedMNIST train/test partitions
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])

    logger.info("Loading MedMNIST dataset: {}", data_flag)
    train_ds = DataClass(
        split='train',
        transform=clip_preprocess,
        download=True
    )
    test_ds = DataClass(
        split='test',
        transform=clip_preprocess,
        download=True
    )

    # Step 2 - Build the low-data multiclass training subset
    # For each selected class, the first N examples are retained from the official
    # training split, ensuring consistent class-specific subsets across different
    # values of K. The resulting subset is class-balanced by construction.

    if not all_samples and num_examples_per_class is not None:
        all_labels = train_ds.labels.squeeze()

        selected_indices = []
        for class_label in range(num_classes):
            indices = np.where(all_labels == class_label)[0][:num_examples_per_class]

            if len(indices) < num_examples_per_class:
                logger.warning(
                    "Class {}: found only {} samples out of {} requested",
                    class_label,
                    len(indices),
                    num_examples_per_class
                )

            selected_indices.extend(indices)

        # Randomise the order of the selected training samples.
        shuffle_indices = np.random.permutation(len(selected_indices))
        selected_indices = np.array(selected_indices)[shuffle_indices]

        train_ds = Subset(train_ds, selected_indices)

    # Restrict the official test split to the same K target classes.
    # All available test samples for the selected classes are retained.
    all_test_labels = test_ds.labels.squeeze()
    test_indices = np.where(
        np.isin(all_test_labels, range(num_classes))
    )[0]
    test_ds = Subset(test_ds, test_indices)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False
    )

    # Step 3 - Extract frozen BiomedCLIP visual embeddings
    # Each image is mapped to a high-dimensional semantic representation using
    # the pre-trained BiomedCLIP image encoder. Embeddings are L2-normalised.

    @torch.no_grad()
    def _extract(dataloader):
        feats_list, labels_list = [], []

        for imgs, labels in tqdm(
            dataloader,
            desc="Extracting embeddings"
        ):
            feats = clip_model.encode_image(imgs.to(device))
            feats = feats / feats.norm(dim=-1, keepdim=True)

            feats_list.append(feats.cpu())
            labels_list.append(labels.view(-1).cpu())

        return (
            torch.cat(feats_list, dim=0).numpy(),
            torch.cat(labels_list, dim=0).numpy(),
        )

    logger.info("Extracting BiomedCLIP embeddings...")
    X_train, Y_train = _extract(train_loader)
    X_test, Y_test = _extract(test_loader)

    logger.info("Shape embedding train: {}", X_train.shape)
    logger.info("Shape embedding test:  {}", X_test.shape)

    # Step 4 - Training-only feature scaling
    # Scaling parameters are estimated exclusively on the sampled training
    # embeddings and then applied unchanged to the test embeddings.
    scaler = MinMaxScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Step 5 - Latent-space structuring with the Triplet Autoencoder
    # The T-AE compresses the BiomedCLIP embeddings to an f-dimensional latent
    # representation, where f matches the number of features required by the
    # quantum feature map under evaluation.

    autoencoder = TripletAutoencoder(
        input_dim=X_train.shape[1],
        bottleneck_dim=n_features,
    )

    autoencoder = train_triplet_autoencoder(
        autoencoder,
        X_train,
        Y_train,
        n_epochs=epochs,
        batch_size=batch_size,
        lr=1e-3,
        margin=margin,
        alpha=alpha,
    )

    # Use the trained encoder to generate compact latent representations for
    # both training and test samples.
    X_train = extract_embeddings(autoencoder, X_train)
    X_test = extract_embeddings(autoencoder, X_test)

    # Step 6 - Map latent variables to the angular encoding domain [0, pi]
    # The scaling range is estimated from the training latent space only and
    # then applied unchanged to the test set. The resulting vectors constitute
    # the direct input to the quantum feature-map strategies.

    max_value = np.max(X_train)
    min_value = np.min(X_train)

    X_train = (
        (X_train - min_value)
        * (np.pi / (max_value - min_value))
    )

    X_test = (
        (X_test - min_value)
        * (np.pi / (max_value - min_value))
    )


    return X_train, X_test, Y_train, Y_test