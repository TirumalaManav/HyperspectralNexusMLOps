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
import pickle
import base64
import numpy as np
import logging
import shutil
import cv2
from deepface import DeepFace
from datetime import datetime
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
from cryptography.fernet import Fernet

class AdvancedCyberSecuritySystem:
    def __init__(self, database_path='secure_data'):
        """Initialize the security system with necessary directories and logging."""

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('security_system.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('CyberSecuritySystem')

        # Setup directory structure
        self.database_path = database_path
        self.reference_images_path = os.path.join(database_path, 'reference_images')
        self.face_image_path = os.path.join(os.getcwd(), 'captured_images')

        # Create directories if they don't exist
        for path in [self.database_path, self.reference_images_path, self.face_image_path]:
            os.makedirs(path, exist_ok=True)

        self.logger.info("Security system initialized successfully")

    def _generate_secure_key(self):
        """Generate a secure encryption key using Fernet."""
        try:
            entropy = os.urandom(32)
            derived_key = base64.urlsafe_b64encode(entropy)
            return Fernet(derived_key)
        except Exception as e:
            self.logger.error(f"Key generation failed: {e}")
            return None

    def _encrypt_data(self, data, fernet_instance):
        """Encrypt data using the provided Fernet instance."""
        try:
            if not isinstance(data, (bytes, bytearray)):
                data = pickle.dumps(data)
            return fernet_instance.encrypt(data)
        except Exception as e:
            self.logger.error(f"Encryption failed: {e}")
            return None

    def _decrypt_data(self, encrypted_data, fernet_instance):
        """Decrypt data using the provided Fernet instance."""
        try:
            decrypted_data = fernet_instance.decrypt(encrypted_data)
            return pickle.loads(decrypted_data)
        except Exception as e:
            self.logger.error(f"Decryption failed: {e}")
            return None

    def _generate_dynamic_key(self, passphrase, salt=None):
        """Generate a dynamic key using PBKDF2 with the provided passphrase."""
        try:
            if not salt:
                salt = os.urandom(16)

            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA512(),
                length=32,
                salt=salt,
                iterations=100000,
                backend=default_backend()
            )

            key = base64.urlsafe_b64encode(kdf.derive(passphrase.encode()))
            return key, salt
        except Exception as e:
            self.logger.error(f"Dynamic key generation failed: {e}")
            return None, None

    def process_base64_image(self, base64_image, username):
        """Process and save a base64 encoded image."""
        try:
            # Handle data URI scheme
            if ',' in base64_image:
                base64_image = base64_image.split(',')[1]

            # Decode base64 image
            image_data = base64.b64decode(base64_image)
            nparr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                raise ValueError("Failed to decode image data")

            # Generate unique filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{username}_face_{timestamp}.jpg"
            image_path = os.path.join(self.face_image_path, filename)

            # Resize and save image
            resized_frame = cv2.resize(frame, (640, 480))
            cv2.imwrite(image_path, resized_frame)

            self.logger.info(f"Face image processed and saved at {image_path}")
            return image_path

        except Exception as e:
            self.logger.error(f"Image processing failed: {e}")
            return None

    def extract_face_features(self, image_path):
        """Extract facial features using multiple detection backends."""
        try:
            backends = ['retinaface', 'mtcnn', 'opencv', 'ssd']

            for backend in backends:
                try:
                    self.logger.info(f"Attempting face detection with {backend} backend")
                    face_features = DeepFace.represent(
                        img_path=image_path,
                        model_name='ArcFace',
                        detector_backend=backend,
                        enforce_detection=True,
                        align=True
                    )

                    if face_features and len(face_features) > 0:
                        self.logger.info(f"Face features extracted successfully using {backend}")
                        return face_features[0]['embedding']

                except Exception as backend_error:
                    self.logger.warning(f"Face detection failed with {backend}: {backend_error}")
                    continue

            self.logger.error("Face detection failed with all backends")
            return None

        except Exception as e:
            self.logger.error(f"Face feature extraction failed: {e}")
            return None

    def register_user(self, username, passphrase, base64_image):
        """Register a new user with their face biometric data."""
        try:
            # Validate inputs
            if not all([username, passphrase, base64_image]):
                self.logger.error("Missing required registration parameters")
                return False

            user_file = os.path.join(self.database_path, f'{username}.bin')

            # Check for existing user
            if os.path.exists(user_file):
                self.logger.warning(f"User {username} already exists")
                return False

            # Process the face image
            face_image = self.process_base64_image(base64_image, username)
            if not face_image:
                return False

            # Extract face features
            face_features = self.extract_face_features(face_image)
            if face_features is None:
                if os.path.exists(face_image):
                    os.remove(face_image)
                return False

            # Save reference image
            reference_path = os.path.join(self.reference_images_path, f'{username}_reference.jpg')
            shutil.move(face_image, reference_path)

            # Generate encryption keys and encrypt data
            fernet_instance = self._generate_secure_key()
            if not fernet_instance:
                return False

            encrypted_biometrics = self._encrypt_data(face_features, fernet_instance)
            if not encrypted_biometrics:
                return False

            encrypted_key, salt = self._generate_dynamic_key(passphrase)
            if not encrypted_key or not salt:
                return False

            # Prepare and save user data
            user_data = {
                'encrypted_biometrics': encrypted_biometrics,
                'salt': salt,
                'encryption_key': encrypted_key,
                'reference_image': reference_path,
                'registration_date': datetime.now().isoformat()
            }

            with open(user_file, 'wb') as f:
                pickle.dump(user_data, f)

            self.logger.info(f"User {username} registered successfully")
            return True

        except Exception as e:
            self.logger.error(f"User registration failed: {e}")
            if 'face_image' in locals() and os.path.exists(face_image):
                os.remove(face_image)
            return False

    def authenticate_user(self, username, passphrase, base64_image):
        """Authenticate a user using their face biometric data."""
        try:
            # Validate inputs
            if not all([username, passphrase, base64_image]):
                self.logger.error("Missing required authentication parameters")
                return False

            user_file = os.path.join(self.database_path, f'{username}.bin')

            # Check if user exists
            if not os.path.exists(user_file):
                self.logger.warning(f"User {username} not found")
                return False

            # Load user data
            with open(user_file, 'rb') as f:
                user_data = pickle.load(f)

            # Verify passphrase
            encryption_key, _ = self._generate_dynamic_key(passphrase, user_data['salt'])
            if encryption_key != user_data['encryption_key']:
                self.logger.warning("Invalid passphrase")
                return False

            # Process the verification image
            current_face_image = self.process_base64_image(base64_image, f"{username}_verify")
            if not current_face_image:
                return False

            try:
                # Perform face verification
                verification_result = DeepFace.verify(
                    img1_path=current_face_image,
                    img2_path=user_data['reference_image'],
                    model_name='ArcFace',
                    detector_backend='retinaface',
                    enforce_detection=True,
                    align=True
                )

                is_verified = verification_result.get('verified', False)

                if is_verified:
                    self.logger.info(f"User {username} authenticated successfully")
                else:
                    self.logger.warning(f"Face verification failed for user {username}")

                return is_verified

            except Exception as verify_error:
                self.logger.error(f"Face verification error: {verify_error}")
                return False

            finally:
                # Clean up temporary verification image
                if os.path.exists(current_face_image):
                    os.remove(current_face_image)

        except Exception as e:
            self.logger.error(f"Authentication failed: {e}")
            return False

    def delete_user(self, username, passphrase):
        """Delete a user's data and associated files."""
        try:
            user_file = os.path.join(self.database_path, f'{username}.bin')

            if not os.path.exists(user_file):
                self.logger.warning(f"User {username} not found")
                return False

            # Load and verify user data
            with open(user_file, 'rb') as f:
                user_data = pickle.load(f)

            # Verify passphrase
            encryption_key, _ = self._generate_dynamic_key(passphrase, user_data['salt'])
            if encryption_key != user_data['encryption_key']:
                self.logger.warning("Invalid passphrase for deletion")
                return False

            # Remove reference image
            reference_image = user_data.get('reference_image')
            if reference_image and os.path.exists(reference_image):
                os.remove(reference_image)

            # Remove user file
            os.remove(user_file)

            self.logger.info(f"User {username} deleted successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error deleting user: {e}")
            return False
