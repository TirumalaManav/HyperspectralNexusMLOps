# Contributors to Hyperspectral Image Classification Framework

## Primary Authors

**Tirumala Manav** <thirumalamanav123@gmail.com>
- Lead Developer and Researcher
- Institution: Hyderabad Institute of Technology and Management
- Contributions:
  - Overall project architecture and design
  - HICNN implementation and optimization
  - Web framework development (Flask)
  - Cryptographic face verification system
  - Documentation and project management
  - Performance optimization and testing

**Lalan Kumar**
- Co-Developer and Researcher
- Institution: Hyderabad Institute of Technology and Management
- Contributions:
  - HICAE autoencoder architecture development
  - Data preprocessing and augmentation pipelines
  - GAN implementation and training
  - Dataset integration and management
  - Performance evaluation and metrics analysis

**Prof. Bhaaskar Das**
- Research Supervisor and Mentor
- Institution: Hyderabad Institute of Technology and Management
- Contributions:
  - Research guidance and technical oversight
  - Architectural design consultation
  - Academic supervision and review
  - Research methodology and validation
  - Publication and dissemination support

## Institutional Affiliation

**Hyderabad Institute of Technology and Management (HITAM)**
- Department: Computer Science and Engineering
- Research Focus: Deep Learning, Computer Vision, Remote Sensing
- Support: Infrastructure, computational resources, academic guidance

## Contact Information

For questions, collaborations, or technical support:
- **Primary Contact**: Tirumala Manav (thirumalamanav123@gmail.com)
- **Institution**: Hyderabad Institute of Technology and Management
- **Project Repository**: https://github.com/TirumalaManav/Hyperspectral-Image-Classification-Framework

## Acknowledgments

We thank the open-source community for providing foundational tools and datasets
that made this research possible, and HITAM for providing the academic environment
and resources necessary for advanced research in hyperspectral image classification.

---

*This project represents collaborative research in advancing hyperspectral image
classification through innovative deep learning architectures.*

import os
import json
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [User: TirumalaManav] - %(message)s',
    handlers=[
        logging.FileHandler('hyperspectral_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global variables
CURRENT_TIMESTAMP = "2025-01-23 18:13:46"
CURRENT_USER = "TirumalaManav"

# Dataset mappings configuration with metadata
DATASET_MAPPINGS = {
    'PaviaU': {
        'data_file': 'PaviaU.mat',
        'label_file': 'PaviaU_gt.mat',
        'data_keys': ['paviaU'],
        'label_keys': ['paviaU_gt'],
        'description': 'Pavia University scene',
        'n_classes': 9,
        'n_bands': 103,
        'class_names': ['Background', 'Asphalt', 'Meadows', 'Gravel', 'Trees',
                       'Metal Sheets', 'Bare Soil', 'Bitumen', 'Bricks', 'Shadows']
    },
    'PaviaC': {
        'data_file': 'Pavia.mat',
        'label_file': 'Pavia_gt.mat',
        'data_keys': ['pavia'],
        'label_keys': ['pavia_gt'],
        'description': 'Pavia Centre scene',
        'n_classes': 9,
        'n_bands': 102,
        'class_names': ['Background', 'Water', 'Trees', 'Asphalt', 'Self-Blocking Bricks',
                       'Bitumen', 'Tiles', 'Shadows', 'Meadows', 'Bare Soil']
    },
    'Salinas': {
        'data_file': 'Salinas.mat',
        'label_file': 'Salinas_gt.mat',
        'data_keys': ['salinas'],
        'label_keys': ['salinas_gt'],
        'description': 'Salinas Valley scene',
        'n_classes': 16,
        'n_bands': 204,
        'class_names': ['Background', 'Brocoli_green_weeds_1', 'Brocoli_green_weeds_2',
                       'Fallow', 'Fallow_rough_plow', 'Fallow_smooth', 'Stubble',
                       'Celery', 'Grapes_untrained', 'Soil_vinyard_develop',
                       'Corn_senesced_green_weeds', 'Lettuce_romaine_4wk',
                       'Lettuce_romaine_5wk', 'Lettuce_romaine_6wk',
                       'Lettuce_romaine_7wk', 'Vinyard_untrained',
                       'Vinyard_vertical_trellis']
    },
    'SalinasA': {
        'data_file': 'SalinasA.mat',
        'label_file': 'SalinasA_gt.mat',
        'data_keys': ['salinasA'],
        'label_keys': ['salinasA_gt'],
        'description': 'Salinas-A scene',
        'n_classes': 6,
        'n_bands': 204,
        'class_names': ['Background', 'Brocoli_green_weeds_1', 'Corn_senesced_green_weeds',
                       'Lettuce_romaine_4wk', 'Lettuce_romaine_5wk', 'Lettuce_romaine_6wk',
                       'Lettuce_romaine_7wk']
    },
    'IndianPines': {
        'data_file': 'Indian_pines.mat',
        'label_file': 'Indian_pines_gt.mat',
        'data_keys': ['indian_pines'],
        'label_keys': ['indian_pines_gt'],
        'description': 'Indian Pines scene',
        'n_classes': 16,
        'n_bands': 200,
        'class_names': ['Background', 'Alfalfa', 'Corn-notill', 'Corn-mintill',
                       'Corn', 'Grass-pasture', 'Grass-trees',
                       'Grass-pasture-mowed', 'Hay-windrowed', 'Oats',
                       'Soybean-notill', 'Soybean-mintill', 'Soybean-clean',
                       'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
                       'Stone-Steel-Towers']
    },
    'KSC': {
        'data_file': 'KSC.mat',
        'label_file': 'KSC_gt.mat',
        'data_keys': ['KSC'],
        'label_keys': ['KSC_gt'],
        'description': 'Kennedy Space Center scene',
        'n_classes': 13,
        'n_bands': 176,
        'class_names': ['Background', 'Scrub', 'Willow swamp', 'CP hammock',
                       'Slash pine', 'Oak/Broadleaf', 'Hardwood swamp',
                       'Graminoid marsh', 'Spartina marsh', 'Cattail marsh',
                       'Salt marsh', 'Mud flats', 'Water', 'Road/Buildings']
    },
    'Botswana': {
        'data_file': 'Botswana.mat',
        'label_file': 'Botswana_gt.mat',
        'data_keys': ['Botswana'],
        'label_keys': ['Botswana_gt'],
        'description': 'Botswana scene',
        'n_classes': 14,
        'n_bands': 145,
        'class_names': ['Background', 'Water', 'Hippo grass', 'Floodplain grasses 1',
                       'Floodplain grasses 2', 'Reeds', 'Riparian', 'Firescar',
                       'Island interior', 'Acacia woodlands', 'Acacia shrublands',
                       'Acacia grasslands', 'Short mopane', 'Mixed mopane',
                       'Exposed soils']
    }
}

def register_custom_dataset(dataset_info):
    """Register a custom dataset with the system."""
    try:
        required_fields = ['name', 'data_file', 'label_file', 'data_keys', 'label_keys',
                          'description', 'n_classes', 'n_bands', 'class_names']

        # Verify all required fields are present
        for field in required_fields:
            if field not in dataset_info:
                raise ValueError(f"Missing required field '{field}' in dataset information")

        # Verify class names match number of classes
        if len(dataset_info['class_names']) != dataset_info['n_classes'] + 1:  # +1 for background
            raise ValueError("Number of class names does not match n_classes (+1 for background)")

        # Add to dataset mappings
        DATASET_MAPPINGS[dataset_info['name']] = {
            'data_file': dataset_info['data_file'],
            'label_file': dataset_info['label_file'],
            'data_keys': dataset_info['data_keys'],
            'label_keys': dataset_info['label_keys'],
            'description': dataset_info['description'],
            'n_classes': dataset_info['n_classes'],
            'n_bands': dataset_info['n_bands'],
            'class_names': dataset_info['class_names']
        }

        logger.info(f"Successfully registered custom dataset: {dataset_info['name']}")
        logger.info(f"Description: {dataset_info['description']}")
        logger.info(f"Number of classes: {dataset_info['n_classes']}")
        logger.info(f"Number of bands: {dataset_info['n_bands']}")

        return True

    except Exception as e:
        logger.error(f"Error registering custom dataset: {str(e)}")
        raise

def validate_dataset_files(dataset_name, data_dir):
    """Validate that all required files for a dataset exist."""
    try:
        if dataset_name not in DATASET_MAPPINGS:
            raise ValueError(f"Dataset '{dataset_name}' is not registered")

        dataset_info = DATASET_MAPPINGS[dataset_name]
        dataset_path = os.path.join(data_dir, dataset_name)

        # Check data file
        data_file_path = os.path.join(dataset_path, dataset_info['data_file'])
        if not os.path.exists(data_file_path):
            raise FileNotFoundError(f"Data file not found: {data_file_path}")

        # Check label file
        label_file_path = os.path.join(dataset_path, dataset_info['label_file'])
        if not os.path.exists(label_file_path):
            raise FileNotFoundError(f"Label file not found: {label_file_path}")

        logger.info(f"Validated files for dataset: {dataset_name}")
        return True

    except Exception as e:
        logger.error(f"Error validating dataset files: {str(e)}")
        raise

def apply_pca(images, n_components):
    """Apply PCA to reduce the dimensionality of the hyperspectral data."""
    try:
        logger.info(f"Starting PCA reduction with n_components={n_components}")
        h, w, c = images.shape

        reshaped_images = images.reshape(-1, c)
        pca = PCA(n_components=n_components)
        reduced_data = pca.fit_transform(reshaped_images)
        reduced_images = reduced_data.reshape(h, w, n_components)

        explained_variance = np.sum(pca.explained_variance_ratio_) * 100
        logger.info(f"PCA completed with {explained_variance:.2f}% explained variance")

        return reduced_images

    except Exception as e:
        logger.error(f"Error in PCA application: {str(e)}")
        raise

def extract_patches(images, labels, patch_size, prediction_mode=False):
    """Extract patches only from labeled pixels (where labels != 0)."""
    try:
        logger.info(f"Starting patch extraction with patch_size={patch_size}")

        patches = []
        patch_labels = []
        patch_locations = []

        h, w = images.shape[:2]
        pad_size = patch_size // 2

        # Count total labeled pixels for progress tracking
        total_labeled = np.sum(labels != 0)
        processed = 0

        for i in range(pad_size, h - pad_size):
            for j in range(pad_size, w - pad_size):
                if labels[i, j] != 0:  # Only extract patches for labeled pixels
                    patch = images[i - pad_size:i + pad_size + 1,
                                 j - pad_size:j + pad_size + 1, :]
                    patches.append(patch)
                    patch_labels.append(labels[i, j])
                    patch_locations.append((i, j))

                    processed += 1
                    if processed % 1000 == 0:
                        logger.info(f"Processed {processed}/{total_labeled} labeled pixels")

        patches = np.array(patches)
        patch_labels = np.array(patch_labels)

        logger.info(f"Extracted {len(patches)} patches from labeled pixels")
        logger.info(f"Patch shape: {patches.shape}")

        return patches, patch_labels, patch_locations

    except Exception as e:
        logger.error(f"Error in patch extraction: {str(e)}")
        raise

def load_model_metadata(model_dir):
    """Load model metadata from training_results.json file."""
    try:
        metadata_path = os.path.join(model_dir, 'training_results.json')
        if not os.path.exists(metadata_path):
            logger.error(f"Training results file not found at: {metadata_path}")
            raise FileNotFoundError(f"Training results file not found at: {metadata_path}")

        with open(metadata_path, 'r') as f:
            training_results = json.load(f)

        metadata = {
            'training_dataset': training_results.get('dataset'),
            'n_classes': 9 if training_results.get('dataset') == 'PaviaU' else None,
            'n_components': training_results.get('hyperparameters', {}).get('n_components'),
            'patch_size': training_results.get('hyperparameters', {}).get('patch_size'),
            'original_bands': 103 if training_results.get('dataset') == 'PaviaU' else None,
            'training_accuracy': training_results.get('final_metrics', {}).get('accuracy'),
            'validation_accuracy': training_results.get('final_metrics', {}).get('val_accuracy'),
            'batch_size': training_results.get('hyperparameters', {}).get('batch_size'),
            'model_type': training_results.get('model_type'),
            'training_history': training_results.get('training_history', {})
        }

        logger.info(f"Loaded model metadata from training results:")
        logger.info(f"Dataset: {metadata['training_dataset']}")
        logger.info(f"Number of components: {metadata['n_components']}")
        logger.info(f"Patch size: {metadata['patch_size']}")
        logger.info(f"Training accuracy: {metadata['training_accuracy']:.4f}")
        logger.info(f"Validation accuracy: {metadata['validation_accuracy']:.4f}")
        logger.info(f"Model type: {metadata['model_type']}")

        return metadata
    except Exception as e:
        logger.error(f"Error loading model metadata: {str(e)}")
        raise

def load_hyperspectral_data(data_dir, dataset_name):
    """Load hyperspectral data and labels from .mat files."""
    try:
        logger.info(f"Starting to load hyperspectral data for dataset: {dataset_name}")

        if dataset_name not in DATASET_MAPPINGS:
            raise ValueError(f"Dataset {dataset_name} is not registered")

        mapping = DATASET_MAPPINGS[dataset_name]
        dataset_path = os.path.join(data_dir, dataset_name)

        # Load image data
        data_file_path = os.path.join(dataset_path, mapping['data_file'])
        image_data = sio.loadmat(data_file_path)

        # Get image data
        images = None
        for key in mapping['data_keys']:
            if key in image_data:
                images = image_data[key]
                logger.info(f"Found image data using key: {key}")
                break

        if images is None:
            raise KeyError(f"No valid data key found in {mapping['data_file']}")

        # Load label data
        label_file_path = os.path.join(dataset_path, mapping['label_file'])
        label_data = sio.loadmat(label_file_path)

        # Get label data
        labels = None
        for key in mapping['label_keys']:
            if key in label_data:
                labels = label_data[key]
                logger.info(f"Found label data using key: {key}")
                break

        if labels is None:
            raise KeyError(f"No valid label key found in {mapping['label_file']}")

        logger.info(f"Image shape: {images.shape}")
        logger.info(f"Label shape: {labels.shape}")
        logger.info(f"Unique labels: {np.unique(labels)}")

        return images, labels

    except Exception as e:
        logger.error(f"Error loading hyperspectral data: {str(e)}")
        raise

import os
import json
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [User: TirumalaManav] - %(message)s',
    handlers=[
        logging.FileHandler('hyperspectral_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HyperspectralPredictor:
    def __init__(self):
        self.timestamp = "2025-01-23 18:21:43"
        self.user = "TirumalaManav"

        # Set GPU memory growth
        self.configure_gpu()

        # Find directories
        self.base_dir = self.find_project_root()
        self.results_dir = os.path.join(self.base_dir, 'results')
        self.Datasets_dir = os.path.join(self.base_dir, 'Datasets')

        # Create necessary directories
        os.makedirs(self.results_dir, exist_ok=True)

        logger.info(f"Initialized predictor with:")
        logger.info(f"Base directory: {self.base_dir}")
        logger.info(f"Results directory: {self.results_dir}")
        logger.info(f"Datasets directory: {self.Datasets_dir}")
        logger.info(f"Timestamp: {self.timestamp}")
        logger.info(f"User: {self.user}")

    def configure_gpu(self):
        """Configure GPU memory growth"""
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"Found {len(gpus)} GPU(s), configured memory growth")
        except Exception as e:
            logger.warning(f"Error configuring GPU: {str(e)}")

    def find_project_root(self):
        """Find project root directory"""
        try:
            current_dir = os.getcwd()
            while True:
                if os.path.exists(os.path.join(current_dir, 'Datasets')):
                    logger.info(f"Found project root at: {current_dir}")
                    return current_dir
                parent = os.path.dirname(current_dir)
                if parent == current_dir:
                    logger.warning(f"Using current directory: {os.getcwd()}")
                    return os.getcwd()
                current_dir = parent
        except Exception as e:
            logger.error(f"Error finding project root: {str(e)}")
            raise

    def get_latest_model(self):
        """Find and load the most recent model"""
        try:
            training_dirs = glob.glob(os.path.join(self.results_dir, 'training_*'))
            if not training_dirs:
                raise FileNotFoundError("No training directories found!")

            latest_dir = max(training_dirs, key=os.path.getctime)
            logger.info(f"Found latest training directory: {latest_dir}")

            # Load model metadata
            model_metadata = load_model_metadata(latest_dir)

            # Load the model
            if os.path.exists(os.path.join(latest_dir, 'autoencoder_model')):
                model_path = os.path.join(latest_dir, 'autoencoder_model')
                model_type = 'autoencoder'
            else:
                model_path = os.path.join(latest_dir, 'cnn_model')
                model_type = 'cnn'

            model = tf.keras.models.load_model(model_path)
            logger.info(f"Loaded {model_type} model from {model_path}")

            return model, latest_dir, model_type, model_metadata
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def generate_predictions(self, model, images, labels, model_metadata):
        """Generate predictions using the loaded model"""
        try:
            logger.info("Starting prediction generation...")

            # Apply PCA if needed
            if images.shape[-1] > model_metadata['n_components']:
                images = apply_pca(images, model_metadata['n_components'])

            # Normalize images
            images = tf.cast(images, tf.float32) / 255.0

            # Extract patches only from labeled pixels
            patches, patch_labels, patch_locations = extract_patches(
                images, labels, model_metadata['patch_size']
            )

            logger.info(f"Processing {len(patches)} patches in batches")

            # Generate predictions in batches
            batch_size = 256
            n_batches = (len(patches) + batch_size - 1) // batch_size
            predictions = []

            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(patches))
                batch = patches[start_idx:end_idx]

                batch_predictions = model.predict(batch, verbose=0)
                if isinstance(batch_predictions, dict):
                    batch_predictions = batch_predictions['classifier']

                predictions.append(batch_predictions)

                if (i + 1) % 10 == 0:
                    logger.info(f"Processed batch {i + 1}/{n_batches}")

            # Combine predictions
            predictions = np.concatenate(predictions, axis=0)
            predicted_labels = np.argmax(predictions, axis=-1) + 1  # Add 1 to match original labels

            # Reconstruct full image
            predicted_image = self.reconstruct_image(
                predicted_labels, patch_locations, labels.shape
            )

            logger.info("Prediction generation completed")
            return predicted_image, predictions
        except Exception as e:
            logger.error(f"Error generating predictions: {str(e)}")
            raise

    def reconstruct_image(self, predictions, patch_locations, original_shape):
        """Reconstruct full image from predictions"""
        try:
            # Initialize with zeros (background)
            reconstructed = np.zeros(original_shape)

            # Place predictions at their original locations
            for pred, (i, j) in zip(predictions, patch_locations):
                reconstructed[i, j] = pred

            return reconstructed
        except Exception as e:
            logger.error(f"Error reconstructing image: {str(e)}")
            raise

    def plot_training_history(self, model_metadata, save_dir):
        """Plot training history metrics."""
        try:
            # Create visualization directory if it doesn't exist
            vis_dir = os.path.join(save_dir, 'visualizations')
            os.makedirs(vis_dir, exist_ok=True)

            # Get training history from metadata
            history = model_metadata.get('training_history', {})

            if not history:
                logger.warning("No training history found in metadata")
                return

            # Create figure with subplots
            fig = plt.figure(figsize=(20, 15))

            # Create GridSpec for better control over subplot sizing
            gs = plt.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

            # Plot Training vs Validation Accuracy
            ax1 = fig.add_subplot(gs[0, 0])
            if 'classifier_accuracy' in history:
                ax1.plot(history['classifier_accuracy'], 'b-', label='Training', marker='o', markersize=4)
                ax1.plot(history['val_classifier_accuracy'], 'r-', label='Validation', marker='o', markersize=4)
            else:
                ax1.plot(history['accuracy'], 'b-', label='Training', marker='o', markersize=4)
                ax1.plot(history['val_accuracy'], 'r-', label='Validation', marker='o', markersize=4)
            ax1.set_title('Model Accuracy', fontsize=12, pad=20)
            ax1.set_xlabel('Epoch', fontsize=10)
            ax1.set_ylabel('Accuracy', fontsize=10)
            ax1.grid(True, linestyle='--', alpha=0.7)
            ax1.legend(loc='lower right', fontsize=10)

            # Plot Training vs Validation Loss
            ax2 = fig.add_subplot(gs[0, 1])
            if 'classifier_loss' in history:
                ax2.plot(history['classifier_loss'], 'b-', label='Training', marker='o', markersize=4)
                ax2.plot(history['val_classifier_loss'], 'r-', label='Validation', marker='o', markersize=4)
            else:
                ax2.plot(history['loss'], 'b-', label='Training', marker='o', markersize=4)
                ax2.plot(history['val_loss'], 'r-', label='Validation', marker='o', markersize=4)
            ax2.set_title('Model Loss', fontsize=12, pad=20)
            ax2.set_xlabel('Epoch', fontsize=10)
            ax2.set_ylabel('Loss', fontsize=10)
            ax2.grid(True, linestyle='--', alpha=0.7)
            ax2.legend(loc='upper right', fontsize=10)

            # Plot Accuracy Trends with Moving Average
            ax3 = fig.add_subplot(gs[1, 0])
            if 'classifier_accuracy' in history:
                acc_data = history['classifier_accuracy']
                val_acc_data = history['val_classifier_accuracy']
            else:
                acc_data = history['accuracy']
                val_acc_data = history['val_accuracy']

            # Calculate moving averages
            window = 3
            acc_ma = np.convolve(acc_data, np.ones(window)/window, mode='valid')
            val_acc_ma = np.convolve(val_acc_data, np.ones(window)/window, mode='valid')

            ax3.plot(acc_data, 'g-', alpha=0.3, label='Training')
            ax3.plot(val_acc_data, 'y-', alpha=0.3, label='Validation')
            ax3.plot(range(window-1, len(acc_data)), acc_ma, 'b-', label='Training Trend')
            ax3.plot(range(window-1, len(val_acc_data)), val_acc_ma, 'r-', label='Validation Trend')
            ax3.set_title('Accuracy Trends (Moving Average)', fontsize=12, pad=20)
            ax3.set_xlabel('Epoch', fontsize=10)
            ax3.set_ylabel('Accuracy', fontsize=10)
            ax3.grid(True, linestyle='--', alpha=0.7)
            ax3.legend(loc='lower right', fontsize=10)

            # Plot Loss Trends with Moving Average
            ax4 = fig.add_subplot(gs[1, 1])
            if 'classifier_loss' in history:
                loss_data = history['classifier_loss']
                val_loss_data = history['val_classifier_loss']
            else:
                loss_data = history['loss']
                val_loss_data = history['val_loss']

            # Calculate moving averages for loss
            loss_ma = np.convolve(loss_data, np.ones(window)/window, mode='valid')
            val_loss_ma = np.convolve(val_loss_data, np.ones(window)/window, mode='valid')

            ax4.plot(loss_data, 'g-', alpha=0.3, label='Training')
            ax4.plot(val_loss_data, 'y-', alpha=0.3, label='Validation')
            ax4.plot(range(window-1, len(loss_data)), loss_ma, 'b-', label='Training Trend')
            ax4.plot(range(window-1, len(val_loss_data)), val_loss_ma, 'r-', label='Validation Trend')
            ax4.set_title('Loss Trends (Moving Average)', fontsize=12, pad=20)
            ax4.set_xlabel('Epoch', fontsize=10)
            ax4.set_ylabel('Loss', fontsize=10)
            ax4.grid(True, linestyle='--', alpha=0.7)
            ax4.legend(loc='upper right', fontsize=10)

            # Add overall title
            plt.suptitle(
                f'Training History Metrics\nDataset: {model_metadata["training_dataset"]} | '
                f'Timestamp: {self.timestamp} | User: {self.user}',
                fontsize=14, y=0.95
            )

            # Save the figure
            history_path = os.path.join(vis_dir, 'training_history.png')
            plt.savefig(history_path, dpi=300, bbox_inches='tight')
            plt.close()

            # Save detailed metrics as text
            self._save_training_history_text(history, model_metadata, vis_dir)

            logger.info(f"Saved training history visualization to {vis_dir}")
            return vis_dir

        except Exception as e:
            logger.error(f"Error plotting training history: {str(e)}")
            raise

    def _save_training_history_text(self, history, model_metadata, save_dir):
        """Save detailed training history metrics as text."""
        try:
            history_text_path = os.path.join(save_dir, 'training_history.txt')
            with open(history_text_path, 'w') as f:
                f.write(f"Training History Summary\n")
                f.write("="*50 + "\n")
                f.write(f"Dataset: {model_metadata['training_dataset']}\n")
                f.write(f"Timestamp: {self.timestamp}\n")
                f.write(f"User: {self.user}\n")
                f.write("="*50 + "\n\n")

                # Write model configuration
                f.write("Model Configuration:\n")
                f.write("-"*20 + "\n")
                f.write(f"Model Type: {model_metadata['model_type']}\n")
                f.write(f"Number of components: {model_metadata['n_components']}\n")
                f.write(f"Patch size: {model_metadata['patch_size']}\n\n")

                # Write final metrics
                f.write("Final Metrics:\n")
                f.write("-"*20 + "\n")
                if 'classifier_accuracy' in history:
                    f.write(f"Training Accuracy: {history['classifier_accuracy'][-1]:.4f}\n")
                    f.write(f"Validation Accuracy: {history['val_classifier_accuracy'][-1]:.4f}\n")
                    f.write(f"Training Loss: {history['classifier_loss'][-1]:.4f}\n")
                    f.write(f"Validation Loss: {history['val_classifier_loss'][-1]:.4f}\n")
                else:
                    f.write(f"Training Accuracy: {history['accuracy'][-1]:.4f}\n")
                    f.write(f"Validation Accuracy: {history['val_accuracy'][-1]:.4f}\n")
                    f.write(f"Training Loss: {history['loss'][-1]:.4f}\n")
                    f.write(f"Validation Loss: {history['val_loss'][-1]:.4f}\n")

                # Write epoch-wise metrics
                f.write("\nEpoch-wise Metrics:\n")
                f.write("-"*20 + "\n")
                f.write("Epoch  Train_Acc  Val_Acc  Train_Loss  Val_Loss\n")
                f.write("-" * 50 + "\n")

                n_epochs = len(history['accuracy' if 'accuracy' in history else 'classifier_accuracy'])
                for epoch in range(n_epochs):
                    if 'classifier_accuracy' in history:
                        f.write(f"{epoch+1:5d}  {history['classifier_accuracy'][epoch]:9.4f}  "
                               f"{history['val_classifier_accuracy'][epoch]:7.4f}  "
                               f"{history['classifier_loss'][epoch]:10.4f}  "
                               f"{history['val_classifier_loss'][epoch]:8.4f}\n")
                    else:
                        f.write(f"{epoch+1:5d}  {history['accuracy'][epoch]:9.4f}  "
                               f"{history['val_accuracy'][epoch]:7.4f}  "
                               f"{history['loss'][epoch]:10.4f}  "
                               f"{history['val_loss'][epoch]:8.4f}\n")

        except Exception as e:
            logger.error(f"Error saving training history text: {str(e)}")
            raise

    def visualize_results(self, ground_truth, predicted, dataset_name, save_dir, model_metadata):
        """Visualize and save prediction results"""
        try:
            # Create visualization directory
            vis_dir = os.path.join(save_dir, 'visualizations')
            os.makedirs(vis_dir, exist_ok=True)

            # Get dataset info and class names
            dataset_info = DATASET_MAPPINGS[dataset_name]
            class_names = dataset_info['class_names']

            # Plot comparison
            self._plot_prediction_comparison(ground_truth, predicted, dataset_name, vis_dir)

            # Calculate and plot metrics
            metrics = self.calculate_metrics(ground_truth, predicted, class_names)
            self.plot_metrics(metrics, dataset_name, vis_dir)

            # Plot training history
            self.plot_training_history(model_metadata, save_dir)

            logger.info(f"Saved all visualizations to {vis_dir}")
            logger.info(f"Overall accuracy: {metrics['accuracy']:.4f}%")

            return vis_dir, metrics
        except Exception as e:
            logger.error(f"Error visualizing results: {str(e)}")
            raise

    def _plot_prediction_comparison(self, ground_truth, predicted, dataset_name, vis_dir):
        """Plot comparison between ground truth and predictions."""
        try:
            plt.figure(figsize=(20, 10))

            # Plot ground truth
            plt.subplot(121)
            plt.title(f'Ground Truth - {dataset_name}', fontsize=12)
            plt.imshow(ground_truth, cmap='nipy_spectral')
            plt.colorbar(label='Classes')
            plt.axis('off')

            # Plot predictions
            plt.subplot(122)
            plt.title(f'Predicted Labels - {dataset_name}', fontsize=12)
            plt.imshow(predicted, cmap='nipy_spectral')
            plt.colorbar(label='Classes')
            plt.axis('off')

            # Add title with metadata
            plt.suptitle(
                f'Prediction Results\nTimestamp: {self.timestamp} | User: {self.user}',
                fontsize=14
            )

            # Save visualization
            save_path = os.path.join(vis_dir, f'{dataset_name}_comparison.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            logger.error(f"Error plotting prediction comparison: {str(e)}")
            raise

    def calculate_metrics(self, ground_truth, predicted, class_names):
        """Calculate prediction metrics"""
        try:
            # Flatten arrays and remove background
            mask = ground_truth != 0
            y_true = ground_truth[mask]
            y_pred = predicted[mask]

            # Get unique labels (excluding background)
            unique_labels = sorted(np.unique(y_true))

            # Calculate metrics
            accuracy = accuracy_score(y_true, y_pred) * 100
            conf_matrix = confusion_matrix(y_true, y_pred, labels=unique_labels)

            # Get class names without background and matching the actual labels
            target_names = [class_names[i] for i in unique_labels]

            class_report = classification_report(
                y_true, y_pred,
                labels=unique_labels,
                target_names=target_names,
                zero_division=0
            )

            logger.info(f"Calculated metrics for {len(unique_labels)} classes")
            logger.info(f"Labels present in data: {unique_labels}")

            return {
                'accuracy': accuracy,
                'confusion_matrix': conf_matrix,
                'classification_report': class_report,
                'labels': unique_labels,
                'target_names': target_names
            }
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            raise

    def plot_metrics(self, metrics, dataset_name, save_dir):
        """Plot and save metrics visualizations"""
        try:
            # Get class names for plotting
            target_names = metrics['target_names']

            # Plot confusion matrix
            plt.figure(figsize=(12, 10))
            sns.heatmap(
                metrics['confusion_matrix'],
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=target_names,
                yticklabels=target_names
            )
            plt.title(f'Confusion Matrix - {dataset_name}\nAccuracy: {metrics["accuracy"]:.2f}%')
            plt.xlabel('Predicted')
            plt.ylabel('True')

            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)

            # Save confusion matrix
            conf_matrix_path = os.path.join(save_dir, f'{dataset_name}_confusion_matrix.png')
            plt.savefig(conf_matrix_path, dpi=300, bbox_inches='tight')
            plt.close()

            # Save classification report
            report_path = os.path.join(save_dir, f'{dataset_name}_classification_report.txt')
            with open(report_path, 'w') as f:
                f.write(f"Classification Report for {dataset_name}\n")
                f.write("="*50 + "\n")
                f.write(f"Timestamp: {self.timestamp}\n")
                f.write(f"User: {self.user}\n")
                f.write("="*50 + "\n\n")
                f.write(f"Overall Accuracy: {metrics['accuracy']:.2f}%\n\n")
                f.write(f"Classes present: {metrics['labels']}\n")
                f.write(f"Class names: {metrics['target_names']}\n\n")
                f.write("Detailed Classification Report:\n")
                f.write("-"*30 + "\n")
                f.write(metrics['classification_report'])

                # Add per-class accuracies
                f.write("\nPer-class Accuracies:\n")
                f.write("-"*30 + "\n")
                conf_matrix = metrics['confusion_matrix']
                for i, class_name in enumerate(target_names):
                    class_correct = conf_matrix[i, i]
                    class_total = conf_matrix[i, :].sum()
                    class_accuracy = (class_correct / class_total) * 100
                    f.write(f"{class_name}: {class_accuracy:.2f}%\n")

            logger.info(f"Saved metrics visualizations to {save_dir}")

        except Exception as e:
            logger.error(f"Error plotting metrics: {str(e)}")
            raise

def main():
    """Main execution function"""
    try:
        logger.info("Starting hyperspectral prediction pipeline...")
        logger.info(f"Timestamp: 2025-01-23 18:24:38")
        logger.info(f"User: TirumalaManav")

        predictor = HyperspectralPredictor()
        model, latest_dir, model_type, model_metadata = predictor.get_latest_model()

        # Process dataset
        dataset_name = model_metadata['training_dataset']
        logger.info(f"Processing dataset: {dataset_name}")

        # Load and process data
        images, ground_truth = load_hyperspectral_data(predictor.Datasets_dir, dataset_name)
        predicted_labels, predictions = predictor.generate_predictions(
            model, images, ground_truth, model_metadata
        )

        # Visualize and save results
        vis_dir, metrics = predictor.visualize_results(
            ground_truth, predicted_labels, dataset_name, latest_dir, model_metadata
        )

        logger.info(f"""
        Completed processing {dataset_name}:
        - Model type: {model_type}
        - Visualization saved in: {vis_dir}
        - Overall accuracy: {metrics['accuracy']:.2f}%
        - Timestamp: 2025-01-23 18:24:38
        - User: TirumalaManav
        """)

        # Print summary of generated files
        logger.info("\nGenerated files:")
        logger.info(f"1. Prediction Comparison: {vis_dir}/{dataset_name}_comparison.png")
        logger.info(f"2. Confusion Matrix: {vis_dir}/{dataset_name}_confusion_matrix.png")
        logger.info(f"3. Classification Report: {vis_dir}/{dataset_name}_classification_report.txt")
        logger.info(f"4. Training History Plot: {vis_dir}/training_history.png")
        logger.info(f"5. Training History Details: {vis_dir}/training_history.txt")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
