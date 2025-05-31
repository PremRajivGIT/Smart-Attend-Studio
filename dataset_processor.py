import os
import cv2
import yaml
import logging
import numpy as np
import shutil
from pathlib import Path
from flask import current_app
from ultralytics import YOLO
import albumentations as A
import random
from tqdm import tqdm

# Define functions that were supposed to be imported
logger = logging.getLogger(__name__)

def extract_frames(video_path, output_dir, class_name, num_frames=150, original_frames=20):
    """Extract frames from a video file
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        class_name: Class name for the extracted frames
        num_frames: Number of frames to extract
        original_frames: Number of original frames to keep
        
    Returns:
        List of paths to extracted frames and original frames
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        return [], []
    
    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate frame indices to extract
    if frame_count <= num_frames:
        # If video has fewer frames than requested, use all frames
        frame_indices = list(range(frame_count))
    else:
        # Otherwise, select frames uniformly
        frame_indices = np.linspace(0, frame_count-1, num_frames, dtype=int)
    
    # Extract frames
    frames = []
    orig_frames = []
    
    for i, frame_idx in enumerate(frame_indices):
        # Set the video position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        # Save the frame
        frame_path = os.path.join(output_dir, f"{class_name}_{i:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        frames.append(frame_path)
        
        # Keep original frames separately
        if i < original_frames:
            orig_path = os.path.join(output_dir, f"{class_name}_orig_{i:04d}.jpg")
            cv2.imwrite(orig_path, frame)
            orig_frames.append(orig_path)
    
    # Release the video capture object
    cap.release()
    
    return frames, orig_frames

def apply_face_augmentations(image_paths, output_dir, num_augmentations=5):
    """Apply augmentations to face images
    
    Args:
        image_paths: List of paths to face images
        output_dir: Directory to save augmented images
        num_augmentations: Number of augmentations per image
        
    Returns:
        List of paths to augmented images
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Define stronger augmentation pipeline - ensure these transformations are visible
    aug_pipeline = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3), p=1.0),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.7),
        A.GaussianBlur(blur_limit=(3, 7), p=0.5),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.8, border_mode=cv2.BORDER_CONSTANT),
        A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), p=0.5),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.7),
        A.OneOf([
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0),
        ], p=0.7),
        A.OneOf([
            A.RandomRain(p=1.0),
            A.RandomSunFlare(p=1.0),
            A.RandomFog(p=1.0),
        ], p=0.3),
    ])
    
    augmented_paths = []
    
    logger.info(f"Applying {num_augmentations} augmentations to {len(image_paths)} images")
    
    # Apply augmentations to each image
    for idx, img_path in enumerate(tqdm(image_paths, desc="Augmenting images")):
        # Read the image
        img = cv2.imread(img_path)
        
        if img is None:
            logger.warning(f"Could not read image: {img_path}")
            continue
        
        # Apply augmentations
        for aug_idx in range(num_augmentations):
            try:
                # Apply augmentation transform
                augmented = aug_pipeline(image=img)['image']
                
                # Save augmented image
                filename = os.path.basename(img_path)
                base_name, ext = os.path.splitext(filename)
                aug_path = os.path.join(output_dir, f"{base_name}_aug_{aug_idx}{ext}")
                
                # Ensure the augmented image is written successfully
                success = cv2.imwrite(aug_path, augmented)
                if not success:
                    logger.error(f"Failed to write augmented image to {aug_path}")
                    continue
                
                # Verify the file was actually created
                if not os.path.exists(aug_path):
                    logger.error(f"Augmented image file does not exist after write: {aug_path}")
                    continue
                
                augmented_paths.append(aug_path)
                logger.debug(f"Created augmented image: {aug_path}")
            except Exception as e:
                logger.error(f"Error augmenting image {img_path}: {str(e)}")
    
    logger.info(f"Successfully created {len(augmented_paths)} augmented images out of {len(image_paths) * num_augmentations} attempted")
    
    # Verify all augmented files exist
    missing_files = [path for path in augmented_paths if not os.path.exists(path)]
    if missing_files:
        logger.warning(f"Found {len(missing_files)} missing augmented files")
        augmented_paths = [path for path in augmented_paths if os.path.exists(path)]
    
    return augmented_paths

def process_selfie_images(image_paths, output_dir):
    """Process selfie images by treating the entire image as a face
    
    Args:
        image_paths: List of paths to images
        output_dir: Directory to save processed images
        
    Returns:
        List of paths to processed images, dictionary with labels
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    processed_images = []
    labels_dict = {}
    
    logger.info(f"Processing {len(image_paths)} selfie images (treating entire image as face)")
    
    # Process each image
    for img_path in tqdm(image_paths, desc="Processing selfie images"):
        try:
            # Get original image filename
            filename = os.path.basename(img_path)
            base_name, ext = os.path.splitext(filename)
            
            # Read the image to get dimensions
            img = cv2.imread(img_path)
            if img is None:
                logger.warning(f"Could not read image: {img_path}")
                continue
                
            img_height, img_width = img.shape[:2]
            
            # Copy the entire image as a "face crop"
            processed_path = os.path.join(output_dir, f"{base_name}_face{ext}")
            shutil.copy2(img_path, processed_path)
            processed_images.append(processed_path)
            
            # Create YOLO format label for the entire image
            # Center coordinates (0.5, 0.5) and full width/height (1.0, 1.0)
            class_idx = 0  # Face class
            center_x = 0.5
            center_y = 0.5
            width = 1.0
            height = 1.0
            
            labels_dict[processed_path] = f"{class_idx} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"
            
            logger.debug(f"Processed selfie image {img_path}, saved to {processed_path}")
        
        except Exception as e:
            logger.error(f"Error processing selfie image {img_path}: {str(e)}")
    
    logger.info(f"Processed {len(processed_images)} selfie images from {len(image_paths)} input images")
    return processed_images, labels_dict

def extract_frames_wrapper(video, output_dir, num_frames=150, original_frames=20):
    """Wrapper function to extract frames from a video using the imported function"""
    try:
        # Create class name from student roll number
        student = video.student
        class_name = student.roll_number
        
        # Extract frames using the imported function
        frames, original_frames = extract_frames(
            video.file_path, 
            output_dir, 
            class_name, 
            num_frames=num_frames, 
            original_frames=original_frames
        )
        
        logger.info(f"Extracted {len(frames)} frames from video {video.id} for student {class_name}")
        return frames, original_frames
    except Exception as e:
        logger.exception(f"Error extracting frames from video {video.id}: {str(e)}")
        return [], []

def process_videos_for_dataset(videos, dataset):
    """Process videos to create a dataset"""
    logger.info(f"Processing {len(videos)} videos for dataset {dataset.id}")
    
    # Load dataset config
    yaml_path = dataset.config_file
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create a mapping of class indices to roll numbers
    class_mapping = {}
    
    # Note: YOLO model is no longer needed since we're treating entire images as faces
    logger.info("Processing selfie videos - treating entire frames as faces, no face detection needed")
    
    # Process each video
    for index, video in enumerate(videos):
        student = video.student
        roll_number = student.roll_number
        
        # Add roll number to class mapping if not already there
        if roll_number not in class_mapping:
            class_idx = len(class_mapping)
            class_mapping[roll_number] = class_idx
            config['names'][class_idx] = roll_number
        
        # Create directories for student data
        train_dir = os.path.join(dataset.path, 'train', 'images', roll_number)
        train_labels_dir = os.path.join(dataset.path, 'train', 'labels', roll_number)
        val_dir = os.path.join(dataset.path, 'val', 'images', roll_number)
        val_labels_dir = os.path.join(dataset.path, 'val', 'labels', roll_number)
        
        # Create all required directories
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(train_labels_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        os.makedirs(val_labels_dir, exist_ok=True)
        
        # Extract frames from the video
        temp_dir = os.path.join(dataset.path, 'temp', roll_number)
        frames_dir = os.path.join(temp_dir, 'frames')
        os.makedirs(frames_dir, exist_ok=True)
        
        num_frames = current_app.config.get('MAX_FRAMES_PER_VIDEO', 150)
        original_frames_count = current_app.config.get('ORIGINAL_FRAMES_TO_KEEP', 20)
        
        frames, original_frames = extract_frames_wrapper(
            video, 
            frames_dir, 
            num_frames=num_frames,
            original_frames=original_frames_count
        )
        
        if not frames:
            logger.warning(f"No frames extracted from video {video.id}")
            continue
        
        # Process selfie images (treat entire frame as face)
        face_crops_dir = os.path.join(temp_dir, 'selfie_crops')
        os.makedirs(face_crops_dir, exist_ok=True)
        
        face_crops, face_labels = process_selfie_images(frames, face_crops_dir)
        
        if not face_crops:
            logger.warning(f"No images processed from video {video.id}")
            # Fallback to original frames
            face_crops = frames
            # Create labels for original frames
            face_labels = {}
            for frame_path in frames:
                processed_path = frame_path
                face_labels[processed_path] = "0 0.500000 0.500000 1.000000 1.000000"
        
        # Generate augmented images
        aug_dir = os.path.join(temp_dir, 'augmented')
        os.makedirs(aug_dir, exist_ok=True)
        
        augmentations_per_frame = current_app.config.get('AUGMENTATIONS_PER_FRAME', 5)
        augmented_images = apply_face_augmentations(
            face_crops, 
            aug_dir, 
            num_augmentations=augmentations_per_frame
        )
        
        logger.info(f"Generated {len(augmented_images)} augmented images for student {roll_number}")
        
        # Create labels for augmented images (same as original - entire image as face)
        for aug_img in augmented_images:
            face_labels[aug_img] = "0 0.500000 0.500000 1.000000 1.000000"
        
        # Split data into train/val (80/20 split)
        all_images = face_crops + augmented_images
        random.shuffle(all_images)
        
        split_idx = int(len(all_images) * 0.8)
        train_images = all_images[:split_idx]
        val_images = all_images[split_idx:]
        
        logger.info(f"Copying {len(train_images)} images to train and {len(val_images)} to validation")
        
        # Copy images and create label files for train set
        for img_path in train_images:
            # Copy image
            filename = os.path.basename(img_path)
            base_name, ext = os.path.splitext(filename)
            dst_path = os.path.join(train_dir, filename)
            
            try:
                shutil.copy2(img_path, dst_path)
                logger.debug(f"Copied {img_path} to {dst_path}")
                
                # Create label file - always create one since we're treating entire image as face
                label_content = face_labels.get(img_path, "0 0.500000 0.500000 1.000000 1.000000")
                label_path = os.path.join(train_labels_dir, f"{base_name}.txt")
                
                with open(label_path, 'w') as f:
                    f.write(label_content)
                logger.debug(f"Created label file {label_path}")
            except Exception as e:
                logger.error(f"Error copying train image {img_path}: {str(e)}")
        
        # Copy images and create label files for validation set
        for img_path in val_images:
            # Copy image
            filename = os.path.basename(img_path)
            base_name, ext = os.path.splitext(filename)
            dst_path = os.path.join(val_dir, filename)
            
            try:
                shutil.copy2(img_path, dst_path)
                logger.debug(f"Copied {img_path} to {dst_path}")
                
                # Create label file - always create one since we're treating entire image as face
                label_content = face_labels.get(img_path, "0 0.500000 0.500000 1.000000 1.000000")
                label_path = os.path.join(val_labels_dir, f"{base_name}.txt")
                
                with open(label_path, 'w') as f:
                    f.write(label_content)
                logger.debug(f"Created label file {label_path}")
            except Exception as e:
                logger.error(f"Error copying validation image {img_path}: {str(e)}")
    
    # Update the configuration file with class mapping and dataset structure
    config['path'] = dataset.path
    config['train'] = os.path.join(dataset.path, 'train', 'images')
    config['val'] = os.path.join(dataset.path, 'val', 'images')
    config['nc'] = len(class_mapping)  # Number of classes
    
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f)
    
    logger.info(f"Dataset {dataset.id} processing completed with {len(class_mapping)} students")