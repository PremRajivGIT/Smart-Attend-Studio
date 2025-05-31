import glob
import os
import json
from flask import request, jsonify, render_template, send_file, url_for, redirect, flash,abort
from app import app, db

from models import Student, Video, Dataset, Model, ProcessingJob, ProcessStatus
from utils import allowed_file, save_uploaded_video, start_background_job, secure_filename

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')
    
@app.route('/check-dataset/<department>/<section>')
def check_dataset(department, section):
    # Check if any dataset directory name contains both department and section
    datasets = os.listdir('./datasets')
    
    for dataset_name in datasets:
        if department in dataset_name and section in dataset_name:
            return jsonify({"exists": True})
    
    # If no matching dataset was found
    return jsonify({"exists": False}), 404

@app.route('/upload', methods=['POST'])
def upload_video():
    """
    Handle video upload with student metadata
    
    Expected POST parameters:
    - roll_number: Student roll number
    - department: Department name
    - section: Section identifier
    - video: Video file
    
    Returns:
    - JSON response with success/error message
    """
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Check if the file is allowed
    if not allowed_file(file.filename):
        return jsonify({
            'error': f'Invalid file type. Allowed types: {", ".join(app.config["ALLOWED_EXTENSIONS"])}'
        }), 400
    
    # Get student metadata
    roll_number = request.form.get('roll_number')
    department = request.form.get('department')
    section = request.form.get('section')
    
    if not all([roll_number, department, section]):
        return jsonify({'error': 'Missing required student metadata'}), 400
    
    try:
        # Save the uploaded video
        video = save_uploaded_video(file, roll_number, department, section)
        
        return jsonify({
            'success': True,
            'message': 'Video uploaded successfully',
            'video_id': video.id,
            'student': {
                'roll_number': roll_number,
                'department': department,
                'section': section
            }
        })
    except Exception as e:
        app.logger.exception("Error saving uploaded video")
        return jsonify({'error': str(e)}), 500

@app.route('/create-dataset', methods=['POST'])
def create_dataset():
    """
    Trigger dataset creation for a specific department and section
    
    Expected POST parameters:
    - department: Department name
    - section: Section identifier
    
    Returns:
    - JSON response with job details
    """
    department = request.form.get('department')
    section = request.form.get('section')
    
    if not department or not section:
        return jsonify({'error': 'Department and section are required'}), 400
    
    # Check if there are uploaded videos for this department and section
    students = Student.query.filter_by(department=department, section=section).all()
    if not students:
        return jsonify({'error': f'No students found for {department} {section}'}), 404
    
    student_ids = [student.id for student in students]
    videos = Video.query.filter(Video.student_id.in_(student_ids)).all()
    
    if not videos:
        return jsonify({'error': f'No videos found for {department} {section}'}), 404
        
    # Check if dataset already exists
    existing_dataset = Dataset.query.filter_by(department=department, section=section).first()
    if existing_dataset:
        app.logger.info(f"Dataset already exists for {department} {section}. Will recreate it.")
    
    # Start a background job to create the dataset
    job = start_background_job('dataset_creation', department=department, section=section)
    
    # Return information about the job
    return jsonify({
        'success': True,
        'message': 'Dataset creation started',
        'job_id': job.job_id,
        'status_url': url_for('job_status', job_id=job.job_id),
        'student_count': len(students),
        'video_count': len(videos)
    })

@app.route('/train/<department>/<section>', methods=['POST'])
def train_model_endpoint(department, section):
    """
    Train a model for a specific classroom
    
    URL parameters:
    - department: Department name
    - section: Section identifier
    
    Returns:
    - JSON response with job details
    """
    # Check if dataset exists for this department and section
    dataset = Dataset.query.filter_by(department=department, section=section).first()
    
    if not dataset:
        return jsonify({'error': f'No dataset found for {department} {section}'}), 404
    
    # Start a background job to train the model
    job = start_background_job('model_training', department=department, section=section, dataset_id=dataset.id)
    
    return jsonify({
        'success': True,
        'message': 'Model training started',
        'job_id': job.job_id,
        'status_url': url_for('job_status', job_id=job.job_id)
    })

@app.route('/models', methods=['GET'])
def list_models():
    """
    List all available trained models
    
    Returns:
    - Rendered template with model list
    """
    models = Model.query.all()
    return render_template('models.html', models=models)

@app.route('/models/json', methods=['GET'])
def list_models_json():
    """
    List all available trained models as JSON
    
    Returns:
    - JSON response with model details
    """
    models = Model.query.all()
    result = []
    
    for model in models:
        result.append({
            'id': model.id,
            'name': model.name,
            'department': model.department,
            'section': model.section,
            'has_tflite': model.tflite_path is not None and os.path.exists(model.tflite_path),
            'has_model': model.model_path is not None and os.path.exists(model.model_path),
            'created_at': model.created_at.isoformat(),
            'download_tflite_url': url_for('download_tflite_model', model_id=model.id) if model.tflite_path else None,
            'download_model_url': url_for('download_original_model', model_id=model.id) if model.model_path else None
        })
    
    return jsonify(result)

@app.route('/export/<int:model_id>', methods=['POST'])
def export_model(model_id):
    """
    Export a model to TFLite format
    
    URL parameters:
    - model_id: ID of the model to export
    
    Returns:
    - JSON response with job details
    """
    model = Model.query.get(model_id)
    
    if not model:
        return jsonify({'error': f'Model {model_id} not found'}), 404
    
    # Start a background job to export the model
    job = start_background_job('tflite_export', model_id=model_id)
    
    return jsonify({
        'success': True,
        'message': 'Model export started',
        'job_id': job.job_id,
        'status_url': url_for('job_status', job_id=job.job_id)
    })

@app.route('/download-tflite/<int:model_id>', methods=['GET'])
def download_tflite_model(model_id):
    """
    Download a TFLite model
    
    URL parameters:
    - model_id: ID of the model to download
    
    Returns:
    - File download
    """
    model = Model.query.get(model_id)
    
    if not model:
        return jsonify({'error': f'Model {model_id} not found'}), 404
    
    if not model.tflite_path or not os.path.exists(model.tflite_path):
        return jsonify({'error': 'TFLite model not available'}), 404
    
    return send_file(
        model.tflite_path,
        as_attachment=True,
        download_name=f"{model.department}_{model.section}.tflite",
        mimetype='application/octet-stream'
    )

@app.route('/download-model/<int:model_id>', methods=['GET'])
def download_original_model(model_id):
    """
    Download the original model file (.pt)
    
    URL parameters:
    - model_id: ID of the model to download
    
    Returns:
    - File download
    """
    model = Model.query.get(model_id)
    
    if not model:
        return jsonify({'error': f'Model {model_id} not found'}), 404
    
    if not model.model_path or not os.path.exists(model.model_path):
        return jsonify({'error': 'Model file not available'}), 404
    
    return send_file(
        model.model_path,
        as_attachment=True,
        download_name=f"{model.department}_{model.section}.pt",
        mimetype='application/octet-stream'
    )

@app.route('/status/<job_id>', methods=['GET'])
def job_status(job_id):
    """
    Check the status of a processing job
    
    URL parameters:
    - job_id: ID of the job to check
    
    Returns:
    - Rendered template with job status
    """
    job = ProcessingJob.query.filter_by(job_id=job_id).first()
    
    if not job:
        return render_template('status.html', error=f'Job {job_id} not found')
    
    return render_template('status.html', job=job)

@app.route('/status/<job_id>/json', methods=['GET'])
def job_status_json(job_id):
    """
    Check the status of a processing job as JSON
    
    URL parameters:
    - job_id: ID of the job to check
    
    Returns:
    - JSON response with job status
    """
    job = ProcessingJob.query.filter_by(job_id=job_id).first()
    
    if not job:
        return jsonify({'error': f'Job {job_id} not found'}), 404
    
    status = {
        'job_id': job.job_id,
        'job_type': job.job_type,
        'status': job.status,
        'created_at': job.created_at.isoformat(),
        'completed_at': job.completed_at.isoformat() if job.completed_at else None,
        'error_message': job.error_message,
        'result_id': job.result_id
    }
    
    # Add result information if job is completed
    if job.status == ProcessStatus.COMPLETED.value and job.result_id:
        if job.job_type == 'dataset_creation':
            dataset = Dataset.query.get(job.result_id)
            if dataset:
                status['result'] = {
                    'dataset_id': dataset.id,
                    'name': dataset.name,
                    'department': dataset.department,
                    'section': dataset.section,
                    'num_students': dataset.num_students
                }
        elif job.job_type == 'model_training':
            model = Model.query.get(job.result_id)
            if model:
                status['result'] = {
                    'model_id': model.id,
                    'name': model.name,
                    'department': model.department,
                    'section': model.section,
                    'export_url': url_for('export_model', model_id=model.id),
                    'download_model_url': url_for('download_original_model', model_id=model.id) if model.model_path else None,
                    'download_tflite_url': url_for('download_tflite_model', model_id=model.id) if model.tflite_path else None,
                    'has_model': model.model_path is not None and os.path.exists(model.model_path),
                    'has_tflite': model.tflite_path is not None and os.path.exists(model.tflite_path)
                }
    
    return jsonify(status)

@app.route('/model-info/<department>/<section>', methods=['GET'])
def get_model_info(department, section):
    """
    Get model information for a specific department and section
    
    URL parameters:
    - department: Department name
    - section: Section identifier
    
    Returns:
    - JSON response with model details
    """
    model = Model.query.filter_by(department=department, section=section).order_by(Model.created_at.desc()).first()
    
    if not model:
        return jsonify({
            'exists': False,
            'message': f'No model found for {department}/{section}'
        }), 404
    
    return jsonify({
        'exists': True,
        'model_id': model.id,
        'name': model.name,
        'department': model.department,
        'section': model.section,
        'has_model': model.model_path is not None and os.path.exists(model.model_path),
        'has_tflite': model.tflite_path is not None and os.path.exists(model.tflite_path),
        'created_at': model.created_at.isoformat(),
        'download_model_url': url_for('download_original_model', model_id=model.id) if model.model_path else None,
        'download_tflite_url': url_for('download_tflite_model', model_id=model.id) if model.tflite_path else None,
        'export_url': url_for('export_model', model_id=model.id)
    })

@app.route('/departments', methods=['GET'])
def list_departments():
    """
    List all departments and sections with student counts
    
    Returns:
    - JSON response with department and section details
    """
    # Query all students grouped by department and section
    result = {}
    students = Student.query.all()
    
    for student in students:
        dept = student.department
        sect = student.section
        
        if dept not in result:
            result[dept] = {}
        
        if sect not in result[dept]:
            result[dept][sect] = 0
        
        result[dept][sect] += 1
    
    formatted_result = []
    for dept, sections in result.items():
        dept_data = {
            'department': dept,
            'sections': [
                {'section': sect, 'student_count': count}
                for sect, count in sections.items()
            ]
        }
        formatted_result.append(dept_data)
    
    return jsonify(formatted_result)


BASE_DIR = 'tflite_models'
def find_model_folder(dept, section):
    pattern = f"{dept}_{section}_model*"
    search_path = os.path.join(BASE_DIR, pattern)
    matching_dirs = glob.glob(search_path)
    if not matching_dirs:
        return None
    return matching_dirs[0]  # Return first match


@app.route('/models/<dept_section>')
def get_model(dept_section):
    try:
        dept, section = dept_section.split('_')
    except ValueError:
        return abort(400, "Invalid format. Use /models/DEPT_SECTION")

    model_dir = find_model_folder(dept, section)
    if not model_dir:
        return abort(404, "Model not found.")

    model_path = os.path.join(model_dir, 'model.tflite')
    if not os.path.exists(model_path):
        return abort(404, "Model file missing.")

    return send_file(model_path, as_attachment=True)


@app.route('/labels/<dept_section>')
def get_labels(dept_section):
    try:
        dept, section = dept_section.split('_')
    except ValueError:
        return abort(400, "Invalid format. Use /labels/DEPT_SECTION")

    model_dir = find_model_folder(dept, section)
    if not model_dir:
        return abort(404, "Model not found.")

    label_path = os.path.join(model_dir, 'labels.txt')
    if not os.path.exists(label_path):
        return abort(404, "Labels file missing.")

    return send_file(label_path, as_attachment=True)
@app.route('/attendance', methods=['POST'])
def receive_attendance():
    try:
        data = request.get_json(force=True)

        # Log the received attendance data
        print("\n--- Attendance Data Received ---")
        print(json.dumps(data, indent=4))

        # You could process/save data here if needed
        # For now, just return success response
        return jsonify({'status': 'success', 'message': 'Attendance received successfully'}), 200
    except Exception as e:
        print(f"Error receiving attendance data: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 400
