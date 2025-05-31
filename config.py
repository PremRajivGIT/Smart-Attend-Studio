import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Directory for video uploads
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Directory for processed data
DATA_FOLDER = os.path.join(BASE_DIR, 'data')
os.makedirs(DATA_FOLDER, exist_ok=True)

# Directory for datasets
DATASET_FOLDER = os.path.join(BASE_DIR, 'datasets')
os.makedirs(DATASET_FOLDER, exist_ok=True)

# Directory for models
MODEL_FOLDER = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_FOLDER, exist_ok=True)

# Directory for TFLite models
TFLITE_FOLDER = os.path.join(BASE_DIR, 'tflite_models')
os.makedirs(TFLITE_FOLDER, exist_ok=True)

# Directory for logs
LOG_FOLDER = os.path.join(BASE_DIR, 'logs')
os.makedirs(LOG_FOLDER, exist_ok=True)

# Valid video extensions
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm', '3gp', 'flv'}

# Database configuration
SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', 'sqlite:///' + os.path.join(BASE_DIR, 'database.db'))
SQLALCHEMY_TRACK_MODIFICATIONS = False

# Processing parameters
MAX_FRAMES_PER_VIDEO = 150
ORIGINAL_FRAMES_TO_KEEP = 20
AUGMENTATIONS_PER_FRAME = 5
