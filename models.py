from app import db
from datetime import datetime
from enum import Enum
import json
import uuid

class ProcessStatus(Enum):
    PENDING = 'pending'
    PROCESSING = 'processing'
    COMPLETED = 'completed'
    FAILED = 'failed'


class Student(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    roll_number = db.Column(db.String(20), nullable=False, unique=True)
    department = db.Column(db.String(50), nullable=False)
    section = db.Column(db.String(10), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    videos = db.relationship('Video', backref='student', lazy=True)
    
    def __repr__(self):
        return f"<Student {self.roll_number}>"


class Video(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(512), nullable=False)
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    student_id = db.Column(db.Integer, db.ForeignKey('student.id'), nullable=False)
    processed = db.Column(db.Boolean, default=False)
    
    def __repr__(self):
        return f"<Video {self.original_filename}>"


class Dataset(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    department = db.Column(db.String(50), nullable=False)
    section = db.Column(db.String(10), nullable=False)
    path = db.Column(db.String(512), nullable=False)
    num_students = db.Column(db.Integer, default=0)
    creation_date = db.Column(db.DateTime, default=datetime.utcnow)
    config_file = db.Column(db.String(512), nullable=True)  # Path to YAML config
    
    def __repr__(self):
        return f"<Dataset {self.name}>"


class Model(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    department = db.Column(db.String(50), nullable=False)
    section = db.Column(db.String(10), nullable=False)
    model_path = db.Column(db.String(512), nullable=False)
    tflite_path = db.Column(db.String(512), nullable=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey('dataset.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    metrics = db.Column(db.Text, nullable=True)  # JSON string with training metrics
    
    dataset = db.relationship('Dataset', backref='models')
    
    def __repr__(self):
        return f"<Model {self.name}>"
    
    def get_metrics(self):
        if self.metrics:
            return json.loads(self.metrics)
        return {}


class ProcessingJob(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    job_id = db.Column(db.String(36), default=lambda: str(uuid.uuid4()), unique=True)
    job_type = db.Column(db.String(50), nullable=False)  # 'dataset_creation', 'model_training', 'tflite_export'
    status = db.Column(db.String(20), default=ProcessStatus.PENDING.value)
    department = db.Column(db.String(50), nullable=True)
    section = db.Column(db.String(10), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime, nullable=True)
    error_message = db.Column(db.Text, nullable=True)
    result_id = db.Column(db.Integer, nullable=True)  # ID of the resulting dataset or model
    job_metadata = db.Column(db.Text, nullable=True)  # Additional JSON metadata - renamed from metadata (reserved name)
    
    def __repr__(self):
        return f"<ProcessingJob {self.job_id} ({self.job_type})>"
    
    def get_metadata(self):
        if self.job_metadata:
            return json.loads(self.job_metadata)
        return {}
    
    def set_metadata(self, data):
        self.job_metadata = json.dumps(data)
