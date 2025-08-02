
# Import necessary libraries
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import scipy.io as sio
import os
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib import cm
import datetime
from pathlib import Path
import json
import logging
import warnings
import traceback
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import logging
import os
import scipy.io as sio
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hyperspectral_processing.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def load_hyperspectral_data(data_dir, dataset_name):
    """
    Load hyperspectral data and labels from .mat files.
    """
    try:
        logger.info(f"Starting to load hyperspectral data for dataset: {dataset_name}")
        logger.info(f"Looking for dataset in directory: {data_dir}")

        dataset_path = os.path.join(data_dir, dataset_name)

        if not os.path.exists(dataset_path):
            logger.error(f"Dataset folder '{dataset_name}' not found in '{data_dir}'")
            raise FileNotFoundError(f"Dataset folder '{dataset_name}' not found in '{data_dir}'.")

        logger.info(f"Searching for .mat files in: {dataset_path}")
        image_file = next((f for f in os.listdir(dataset_path) if f.endswith('.mat') and '_gt' not in f), None)
        label_file = next((f for f in os.listdir(dataset_path) if f.endswith('_gt.mat')), None)

        if not image_file or not label_file:
            logger.error(f"Image or label .mat files not found in the '{dataset_name}' dataset directory")
            raise FileNotFoundError(f"Image or label .mat files not found in the '{dataset_name}' dataset directory.")

        logger.info(f"Found image file: {image_file}")
        logger.info(f"Found label file: {label_file}")

        # Loading image data
        logger.info(f"Loading image data from: {image_file}")
        image_data = sio.loadmat(os.path.join(dataset_path, image_file))
        logger.debug(f"Keys in the image file '{image_file}': {image_data.keys()}")

        image_key = next(
            (key for key in image_data.keys() if dataset_name.lower() in key.lower() or 'data' in key.lower() or key.lower() in ['pavia', 'ksc', 'botswana']),
            None
        )

        if image_key is None:
            logger.error(f"Image data key for '{dataset_name}' not found in the image file {image_file}")
            raise KeyError(f"Image data key for '{dataset_name}' not found in the image file {image_file}.")

        logger.info(f"Found image key: {image_key}")
        images = image_data.get(image_key)

        # Loading label data
        logger.info(f"Loading label data from: {label_file}")
        label_data = sio.loadmat(os.path.join(dataset_path, label_file))
        logger.debug(f"Keys in the label file '{label_file}': {label_data.keys()}")

        label_key = next(
            (key for key in label_data.keys() if 'gt' in key.lower() or 'labels' in key.lower()),
            None
        )

        if label_key is None:
            logger.error(f"Label data key for '{dataset_name}' not found in the label file {label_file}")
            raise KeyError(f"Label data key for '{dataset_name}' not found in the label file {label_file}.")

        logger.info(f"Found label key: {label_key}")
        labels = label_data.get(label_key)

        logger.info(f"Image shape: {images.shape}")
        logger.info(f"Label shape: {labels.shape}")
        logger.info(f"Unique labels: {set(labels.flatten())}")
        logger.info("Successfully loaded hyperspectral data and labels")

        return images, labels

    except Exception as e:
        logger.error(f"Error occurred while loading hyperspectral data: {str(e)}", exc_info=True)
        raise


import logging
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [User: TirumalaManav] - %(message)s',
    handlers=[
        logging.FileHandler('hyperspectral_preprocessing.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def apply_pca(images, n_components):
    """
    Apply PCA to reduce the dimensionality of the hyperspectral data.
    """
    try:
        logger.info(f"Starting PCA reduction with n_components={n_components}")

        h, w, c = images.shape
        logger.info(f"Input image shape: height={h}, width={w}, channels={c}")

        logger.info("Reshaping images for PCA")
        reshaped_images = images.reshape(-1, c)

        logger.info("Initializing PCA")
        pca = PCA(n_components=n_components)

        logger.info("Applying PCA transformation")
        reduced_data = pca.fit_transform(reshaped_images)

        logger.info("Reshaping data back to image format")
        reduced_images = reduced_data.reshape(h, w, n_components)

        explained_variance = np.sum(pca.explained_variance_ratio_) * 100
        logger.info(f"PCA completed: Original bands = {c}, Reduced bands = {n_components}")
        logger.info(f"Total explained variance: {explained_variance:.2f}%")

        return reduced_images

    except Exception as e:
        logger.error(f"Error in PCA application: {str(e)}", exc_info=True)
        raise

def extract_patches(images, labels, patch_size=7):
    """
    Extract patches from the hyperspectral image based on valid label locations.
    """
    try:
        logger.info(f"Starting patch extraction with patch_size={patch_size}")
        logger.info(f"Input image shape: {images.shape}")
        logger.info(f"Input labels shape: {labels.shape}")

        patches = []
        valid_labels = []
        patch_count = 0

        logger.info("Extracting patches...")
        for i in range(patch_size // 2, images.shape[0] - patch_size // 2):
            for j in range(patch_size // 2, images.shape[1] - patch_size // 2):
                if labels[i, j] != 0:
                    patch = images[i - patch_size // 2:i + patch_size // 2 + 1,
                                 j - patch_size // 2:j + patch_size // 2 + 1, :]
                    patches.append(patch)
                    valid_labels.append(labels[i, j])
                    patch_count += 1

        patches_array = np.array(patches)
        valid_labels_array = np.array(valid_labels)

        logger.info(f"Patch extraction completed. Total patches extracted: {patch_count}")
        logger.info(f"Output patches shape: {patches_array.shape}")
        logger.info(f"Output labels shape: {valid_labels_array.shape}")

        return patches_array, valid_labels_array

    except Exception as e:
        logger.error(f"Error in patch extraction: {str(e)}", exc_info=True)
        raise

def preprocess_hyperspectral_data(images, labels, model_type='standard', patch_size=7, batch_size=32):
    """
    Unified preprocessing function for both CNN and Autoencoder
    """
    try:
        logger.info(f"Starting preprocessing with model_type={model_type}, patch_size={patch_size}, batch_size={batch_size}")
        logger.info(f"Input images shape: {images.shape}")
        logger.info(f"Input labels shape: {labels.shape if labels is not None else 'None'}")

        # Normalize images
        logger.info("Normalizing images...")
        images = tf.cast(images, tf.float32) / 255.0

        if labels is not None:
            logger.info("Converting labels to int32")
            labels = tf.cast(labels, tf.int32)

        # Extract patches
        logger.info("Extracting patches...")
        patches, valid_labels = extract_patches(images, labels, patch_size)
        X = patches
        y = valid_labels - 1  # Adjust labels to start from 0

        logger.info(f"Patches shape: {X.shape}")
        logger.info(f"Valid labels shape: {y.shape}")

        # Convert to numpy arrays
        logger.info("Converting to numpy arrays...")
        X = X.numpy() if isinstance(X, tf.Tensor) else X
        y = y.numpy() if isinstance(y, tf.Tensor) else y

        # Split data
        logger.info("Splitting data into train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logger.info(f"Training set size: {len(X_train)}")
        logger.info(f"Test set size: {len(X_test)}")

        logger.info(f"Creating {model_type} datasets...")
        if model_type == 'standard':
            train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
            test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

        elif model_type == 'autoencoder':
            train_dataset = tf.data.Dataset.from_tensor_slices((X_train, X_train))
            test_dataset = tf.data.Dataset.from_tensor_slices((X_test, X_test))

        elif model_type == 'autoencoder_classifier':
            train_dataset = tf.data.Dataset.from_tensor_slices((
                X_train,
                {
                    'decoded': X_train,
                    'classifier': y_train
                }
            ))
            test_dataset = tf.data.Dataset.from_tensor_slices((
                X_test,
                {
                    'decoded': X_test,
                    'classifier': y_test
                }
            ))

        # Batch and prefetch
        logger.info(f"Batching datasets with batch_size={batch_size}")
        train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        logger.info("Preprocessing completed successfully")
        return train_dataset, test_dataset

    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}", exc_info=True)
        raise


import tensorflow as tf
from tensorflow.keras import layers, models

class HyperspectralAE(tf.keras.Model):
    def __init__(self, in_channels, n_classes):
        super(HyperspectralAE, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes

        # Call build_model to initialize layers
        self.build_model()

    def build_model(self):
        """
        Function to build the model architecture.
        The architecture is unchanged from the previous version you shared.
        """
        # Encoder
        self.encoder = models.Sequential([
            layers.Conv2D(64, kernel_size=3, padding='same', input_shape=(7, 7, self.in_channels)),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(128, kernel_size=3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(256, kernel_size=3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()
        ])
        # Decoder - Keep the same spatial dimensions
        self.decoder = models.Sequential([
            layers.Conv2D(128, kernel_size=3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(64, kernel_size=3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(self.in_channels, kernel_size=3, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('sigmoid')
        ])
        # Classifier
        self.classifier = models.Sequential([
            layers.Conv2D(512, kernel_size=3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(1024, kernel_size=3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.GlobalAveragePooling2D(),
            layers.Dense(1024, activation='relu'),
            layers.Dense(self.n_classes, activation='softmax')
        ])

    def call(self, x):
        # Forward pass through the encoder
        encoded = self.encoder(x)
        # Forward pass through the decoder and classifier
        decoded = self.decoder(encoded)
        classified = self.classifier(encoded)

        return {
            'decoded': decoded,
            'classifier': classified
        }

from tensorflow.keras import layers, models

class HyperspectralCNN(models.Model):
    def __init__(self, in_channels, n_classes):
        super(HyperspectralCNN, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.model = self._build_model()

    def _build_model(self):
        return models.Sequential([
            # First Conv Block
            layers.Conv2D(64, kernel_size=3, padding='same', input_shape=(7, 7, self.in_channels)),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(pool_size=2, strides=1),
            # Second Conv Block
            layers.Conv2D(128, kernel_size=3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(pool_size=2, strides=1),
            # Third Conv Block
            layers.Conv2D(256, kernel_size=3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(pool_size=2, strides=1),
            # Fourth Conv Block
            layers.Conv2D(512, kernel_size=3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(pool_size=2, strides=1),
            # Fifth Conv Block
            layers.Conv2D(1024, kernel_size=3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(pool_size=2, strides=1),
            # Flatten and Dense layers
            layers.Flatten(),
            layers.Dense(1024, activation='relu'),
            layers.Dense(self.n_classes, activation='softmax')
        ])

    def call(self, x):
        return self.model(x)

import logging
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [User: TirumalaManav] - %(message)s',
    handlers=[
        logging.FileHandler('model_training.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def compile_model_dynamic(model, model_type='standard'):
    """
    Dynamically compile model based on its type
    """
    try:
        logger.info(f"Starting model compilation for model_type: {model_type}")
        logger.info(f"Model summary:\n{model.summary()}")

        if model_type == 'standard':
            logger.info("Compiling standard classification model")
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            logger.info("Model compiled with sparse_categorical_crossentropy loss and accuracy metric")

        elif model_type == 'autoencoder':
            logger.info("Compiling autoencoder model")
            model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mse']
            )
            logger.info("Model compiled with MSE loss and MSE metric")

        elif model_type == 'autoencoder_classifier':
            logger.info("Compiling autoencoder-classifier model")
            model.compile(
                optimizer='adam',
                loss={
                    'decoded': 'mse',
                    'classifier': 'sparse_categorical_crossentropy'
                },
                loss_weights={
                    'decoded': 1.0,
                    'classifier': 1.0
                },
                metrics={
                    'classifier': 'accuracy'
                }
            )
            logger.info("Model compiled with multiple losses and metrics for autoencoder-classifier")
        else:
            logger.error(f"Unknown model type: {model_type}")
            raise ValueError(f"Unknown model type: {model_type}")

        logger.info("Model compilation completed successfully")

    except Exception as e:
        logger.error(f"Error during model compilation: {str(e)}", exc_info=True)
        raise

def train_model(model, train_dataset, test_dataset, model_type='standard', epochs=10):
    """
    Train the model with appropriate callbacks and monitoring
    """
    try:
        start_time = datetime.utcnow()
        logger.info(f"Starting model training at {start_time}")
        logger.info(f"Training parameters: model_type={model_type}, epochs={epochs}")

        # Log dataset information
        try:
            train_size = sum(1 for _ in train_dataset)
            test_size = sum(1 for _ in test_dataset)
            logger.info(f"Training dataset size: {train_size} batches")
            logger.info(f"Testing dataset size: {test_size} batches")
        except Exception as e:
            logger.warning(f"Could not determine dataset sizes: {str(e)}")

        # Define monitoring metric based on model type
        if model_type == 'standard':
            monitor_metric = 'val_accuracy'
        elif model_type == 'autoencoder_classifier':
            monitor_metric = 'val_classifier_accuracy'
        else:
            monitor_metric = 'val_loss'

        logger.info(f"Using monitoring metric: {monitor_metric}")

        # Define callbacks
        model_save_path = f'best_model_{model_type}'
        logger.info(f"Model will be saved to: {model_save_path}")

        callbacks = [
            EarlyStopping(
                monitor=monitor_metric,
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=model_save_path,
                monitor=monitor_metric,
                save_best_only=True,
                verbose=1,
                save_format='tf'
            )
        ]

        logger.info("Starting model training...")
        history = model.fit(
            train_dataset,
            validation_data=test_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )

        end_time = datetime.utcnow()
        training_duration = end_time - start_time

        # Log training results
        logger.info(f"Training completed at {end_time}")
        logger.info(f"Total training time: {training_duration}")
        logger.info("Final training metrics:")
        for metric, values in history.history.items():
            logger.info(f"{metric}: {values[-1]:.4f}")

        # Log best performance
        best_epoch = history.history[monitor_metric].index(max(history.history[monitor_metric]))
        logger.info(f"Best {monitor_metric}: {max(history.history[monitor_metric]):.4f} at epoch {best_epoch + 1}")

        return history

    except Exception as e:
        logger.error(f"Error during model training: {str(e)}", exc_info=True)
        raise


# Main execution
if __name__ == "__main__":
    try:
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        # Print startup information
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"Starting execution at: {current_time}")
        logger.info(f"User: TirumalaManav")

        # Set up parameters
        data_dir = r"D:\HIC\HIC Practise\Hyperspectral-Classification-master\Hyperspectral-Classification-master\Datasets"
        dataset_name = "PaviaU"
        n_components = 30
        batch_size = 32
        epochs = 2
        patch_size = 7

        # Create results directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path(f'results_{timestamp}')
        results_dir.mkdir(exist_ok=True)

        # Set up file logging
        fh = logging.FileHandler(results_dir / 'training.log')
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)

        # Load and preprocess data
        logger.info("Loading and preprocessing data...")
        images, labels = load_hyperspectral_data(data_dir, dataset_name)
        images = apply_pca(images, n_components)

        # Get number of classes and create label values
        unique_labels = np.unique(labels)
        n_classes = len(unique_labels[unique_labels != 0])
        label_values = [f"Class_{i}" for i in range(n_classes)]
        logger.info(f"Number of classes: {n_classes}")

        # Train and save CNN Model
        logger.info("Starting CNN training...")
        cnn_model = HyperspectralCNN(n_components, n_classes)

        train_dataset, test_dataset = preprocess_hyperspectral_data(
            images,
            labels,
            model_type='standard',
            patch_size=patch_size,
            batch_size=batch_size
        )

        compile_model_dynamic(cnn_model, 'standard')
        cnn_history = train_model(cnn_model, train_dataset, test_dataset, 'standard', epochs)

        # Plot CNN training history
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(cnn_history.history['loss'], label='Training Loss')
        plt.plot(cnn_history.history['val_loss'], label='Validation Loss')
        plt.title('CNN Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(cnn_history.history['accuracy'], label='Training Accuracy')
        plt.plot(cnn_history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('CNN Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.savefig(results_dir / 'cnn_training_history.png')
        plt.close()

        # Train and save Autoencoder Model
        logger.info("Starting Autoencoder training...")
        ae_model = HyperspectralAE(n_components, n_classes)

        train_dataset, test_dataset = preprocess_hyperspectral_data(
            images,
            labels,
            model_type='autoencoder_classifier',
            patch_size=patch_size,
            batch_size=batch_size
        )

        compile_model_dynamic(ae_model, 'autoencoder_classifier')
        ae_history = train_model(
            ae_model,
            train_dataset,
            test_dataset,
            'autoencoder_classifier',
            epochs
        )

        # Plot Autoencoder results
        plt.figure(figsize=(15, 4))

        plt.subplot(1, 3, 1)
        plt.plot(ae_history.history['decoded_loss'], label='Train')
        plt.plot(ae_history.history['val_decoded_loss'], label='Validation')
        plt.title('Reconstruction Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(ae_history.history['classifier_loss'], label='Train')
        plt.plot(ae_history.history['val_classifier_loss'], label='Validation')
        plt.title('Classification Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.plot(ae_history.history['classifier_accuracy'], label='Train')
        plt.plot(ae_history.history['val_classifier_accuracy'], label='Validation')
        plt.title('Classification Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.savefig(results_dir / 'autoencoder_training_history.png')
        plt.close()

        # Save models
        logger.info("Saving models...")
        # Save models using TensorFlow SavedModel format
        tf.saved_model.save(cnn_model, str(results_dir / f'cnn_model_{timestamp}'))
        tf.saved_model.save(ae_model, str(results_dir / f'autoencoder_model_{timestamp}'))

        # Save weights separately
        cnn_model.save_weights(str(results_dir / f'cnn_model_weights_{timestamp}'))
        ae_model.save_weights(str(results_dir / f'autoencoder_model_weights_{timestamp}'))

        # Prepare training summary
        training_summary = {
            'timestamp': timestamp,
            'user': 'TirumalaManav',
            'parameters': {
                'n_components': n_components,
                'batch_size': batch_size,
                'epochs': epochs,
                'patch_size': patch_size,
                'n_classes': n_classes
            },
            'model_paths': {
                'cnn_model': str(results_dir / f'cnn_model_{timestamp}'),
                'autoencoder_model': str(results_dir / f'autoencoder_model_{timestamp}'),
                'cnn_weights': str(results_dir / f'cnn_model_weights_{timestamp}'),
                'autoencoder_weights': str(results_dir / f'autoencoder_model_weights_{timestamp}')
            },
            'cnn_training': {
                'final_train_loss': float(cnn_history.history['loss'][-1]),
                'final_train_accuracy': float(cnn_history.history['accuracy'][-1]),
                'final_val_loss': float(cnn_history.history['val_loss'][-1]),
                'final_val_accuracy': float(cnn_history.history['val_accuracy'][-1])
            },
            'autoencoder_training': {
                'final_reconstruction_loss': float(ae_history.history['decoded_loss'][-1]),
                'final_classifier_loss': float(ae_history.history['classifier_loss'][-1]),
                'final_classifier_accuracy': float(ae_history.history['classifier_accuracy'][-1]),
                'final_val_classifier_accuracy': float(ae_history.history['val_classifier_accuracy'][-1])
            }
        }

        # Save training summary
        with open(results_dir / 'training_summary.json', 'w') as f:
            json.dump(training_summary, f, indent=4)

        # Save training history
        history_data = {
            'cnn_history': {k: [float(v) if isinstance(v, (list, np.ndarray)) else v
                           for v in history] for k, history in cnn_history.history.items()},
            'ae_history': {k: [float(v) if isinstance(v, (list, np.ndarray)) else v
                          for v in history] for k, history in ae_history.history.items()}
        }
        with open(results_dir / 'training_history.json', 'w') as f:
            json.dump(history_data, f, indent=4)

        # Log completion
        logger.info(f"\nExecution completed at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Results saved in: {results_dir}")

    except Exception as e:
        logger.error(f"Error occurred during execution: {str(e)}")
        logger.error(traceback.format_exc())
        raise
