import os
import uuid
import logging
import threading
import json
from datetime import datetime
from flask import current_app
from models import db, Student, Video, Dataset, Model, ProcessingJob, ProcessStatus

# Simulated secure_filename function if werkzeug is not available
def secure_filename(filename):
    """Simulated secure_filename function"""
    return filename.replace(' ', '_').replace('/', '_').replace('\\', '_')

logger = logging.getLogger(__name__)

def allowed_file(filename):
    """Check if uploaded file has an allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']

def save_uploaded_video(file, roll_number, department, section):
    """Save uploaded video file and create database entries"""
    # Secure the filename
    original_filename = secure_filename(file.filename)
    
    # Generate a unique filename
    unique_id = str(uuid.uuid4())
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{roll_number}_{timestamp}_{unique_id}.{original_filename.rsplit('.', 1)[1].lower()}"
    
    # Define the file path
    file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    
    # Find or create student
    student = Student.query.filter_by(roll_number=roll_number).first()
    if not student:
        student = Student(roll_number=roll_number, department=department, section=section)
        db.session.add(student)
        # We need to commit here to get a valid student.id value
        db.session.commit()
    
    # Save the file
    file.save(file_path)
    
    # Create video record
    video = Video(
        filename=filename,
        original_filename=original_filename,
        file_path=file_path,
        student_id=student.id
    )
    
    db.session.add(video)
    db.session.commit()
    
    logger.info(f"Video saved: {file_path} for student {roll_number}")
    return video

def create_dataset_config(department, section):
    """Create a dataset configuration file (YAML) for YOLO training"""
    # Generate a unique name for the dataset
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    dataset_name = f"{department}_{section}_{timestamp}"
    
    # Create dataset directory
    dataset_dir = os.path.join(current_app.config['DATASET_FOLDER'], dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Create YOLO dataset directory structure
    images_dir = os.path.join(dataset_dir, 'images')
    labels_dir = os.path.join(dataset_dir, 'labels')
    train_img_dir = os.path.join(dataset_dir, 'train', 'images')
    train_label_dir = os.path.join(dataset_dir, 'train', 'labels')
    val_img_dir = os.path.join(dataset_dir, 'validation', 'images')
    val_label_dir = os.path.join(dataset_dir, 'validation', 'labels')
    
    # Create all necessary directories
    for directory in [images_dir, labels_dir, train_img_dir, train_label_dir, val_img_dir, val_label_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Get students for this department and section to create class names
    students = Student.query.filter_by(department=department, section=section).all()
    class_names = [student.roll_number for student in students]
    
    # Create YAML configuration path
    yaml_path = os.path.join(dataset_dir, 'dataset.yaml')
    
    # Create the dataset record
    dataset = Dataset(
        name=dataset_name,
        department=department,
        section=section,
        path=dataset_dir,
        config_file=yaml_path,
        num_students=len(students)
    )
    
    db.session.add(dataset)
    db.session.commit()
    
    # Create the YAML config file
    try:
        # Try to use PyYAML if available
        import yaml
        config = {
            'train': './train',
            'val': './validation',
            'nc': len(class_names),
            'names': class_names
        }
        
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    except ImportError:
        # Fall back to simple text-based YAML
        with open(yaml_path, 'w') as f:
            f.write(f"# YOLOv5/YOLOv8 dataset configuration\n")
            f.write(f"train: ./train\n")
            f.write(f"val: ./validation\n\n")
            f.write(f"nc: {len(class_names)}  # number of classes\n")
            
            # Format class names for YAML
            names_str = ", ".join([f"'{name}'" for name in class_names])
            f.write(f"names: [{names_str}]  # class names\n")
    
    logger.info(f"Created dataset configuration: {yaml_path} with {len(class_names)} classes")
    return dataset

def start_background_job(job_type, department=None, section=None, dataset_id=None, model_id=None):
    """Create and start a background processing job"""
    # Create job record
    job = ProcessingJob(
        job_type=job_type,
        status=ProcessStatus.PENDING.value,
        department=department,
        section=section
    )
    
    # Add metadata if provided
    metadata = {}
    if dataset_id:
        metadata['dataset_id'] = dataset_id
    if model_id:
        metadata['model_id'] = model_id
    
    if metadata:
        job.set_metadata(metadata)
    
    db.session.add(job)
    db.session.commit()
    
    # Start the job in a background thread
    from worker import process_job
    thread = threading.Thread(target=process_job, args=(job.job_id,))
    thread.daemon = True
    thread.start()
    
    logger.info(f"Started background job: {job.job_id} of type {job_type}")
    return job

def get_video_metadata(video_path):
    """Extract metadata from a video file"""
    if not os.path.exists(video_path):
        return None
    
    # Get basic file information
    file_size = os.path.getsize(video_path) # in bytes
    base_metadata = {
        'file_exists': True,
        'file_path': video_path,
        'file_size': file_size,
        'file_size_mb': round(file_size / (1024 * 1024), 2)
    }
    
    # Try to use OpenCV for more detailed metadata if available
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            # Get video properties
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            # Add detailed metadata
            detailed_metadata = {
                'frame_count': frame_count,
                'fps': fps,
                'width': width,
                'height': height,
                'duration': duration,
                'duration_formatted': f"{int(duration // 60)}:{int(duration % 60):02d}"
            }
            
            # Release video capture object
            cap.release()
            
            # Merge base and detailed metadata
            return {**base_metadata, **detailed_metadata}
    except (ImportError, Exception) as e:
        logger.warning(f"Could not get detailed video metadata: {str(e)}")
    
    # Return basic metadata if OpenCV is not available or fails
    return base_metadata
