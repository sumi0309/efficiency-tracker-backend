# video_analysis.py

import cv2
import torch
from ultralytics import YOLO
import os
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List, Any

def analyze_single_video(video_path: str, model: YOLO) -> Dict[str, Any]:
    """Analyze a single video for worker activity"""
    print(f"Analyzing video: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return None
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps
    
    # Initialize metrics
    frame_metrics = []
    sample_interval = int(fps * 5)  # Sample every 5 seconds
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % sample_interval == 0:
            # Detect people in frame
            results = model(frame, classes=[0])  # class 0 is person in COCO
            
            # Calculate activity based on bounding box movement
            boxes = results[0].boxes
            num_people = len(boxes)
            
            if num_people > 0:
                # Calculate movement/activity based on box sizes and positions
                box_areas = []
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    area = (x2 - x1) * (y2 - y1)
                    box_areas.append(area)
                
                # Larger boxes typically mean people are moving/working
                active_threshold = np.mean(box_areas) * 0.8
                active_workers = sum(1 for area in box_areas if area > active_threshold)
                
                frame_metrics.append({
                    'timestamp': frame_count / fps,
                    'total_workers': num_people,
                    'active_workers': active_workers,
                    'inactive_workers': num_people - active_workers
                })
        
        frame_count += 1
        if frame_count >= total_frames:
            break
    
    cap.release()
    
    if not frame_metrics:
        return None
    
    # Calculate overall metrics
    total_samples = len(frame_metrics)
    avg_total_workers = np.mean([m['total_workers'] for m in frame_metrics])
    avg_active_workers = np.mean([m['active_workers'] for m in frame_metrics])
    
    return {
        'duration_seconds': duration,
        'frames_analyzed': frame_count,
        'average_workers': round(avg_total_workers, 2),
        'average_active_workers': round(avg_active_workers, 2),
        'productivity_rate': round((avg_active_workers / avg_total_workers * 100) if avg_total_workers > 0 else 0, 2),
        'samples_taken': total_samples
    }

def analyze_surveillance_videos(video_folder: str) -> Dict[str, Any]:
    """Analyze all videos in the surveillance folder"""
    print(f"\nAnalyzing videos in folder: {video_folder}")
    
    # Check for video files
    video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi'))]
    if not video_files:
        print("No video files found!")
        return None
    
    print(f"Found {len(video_files)} video files")
    
    # Load YOLO model
    try:
        model = YOLO('yolov8n.pt')
    except Exception as e:
        print(f"Error loading YOLO model: {str(e)}")
        return None
    
    # Process each video
    results = {
        'overall_metrics': {
            'total_videos_analyzed': len(video_files),
            'total_workers_detected': 0,
            'average_productivity_rate': 0,
            'total_analysis_duration': 0
        },
        'camera_feeds': {}
    }
    
    productivity_rates = []
    
    for video_file in video_files:
        video_path = os.path.join(video_folder, video_file)
        video_results = analyze_single_video(video_path, model)
        
        if video_results:
            results['camera_feeds'][video_file] = video_results
            results['overall_metrics']['total_workers_detected'] += video_results['average_workers']
            results['overall_metrics']['total_analysis_duration'] += video_results['duration_seconds']
            productivity_rates.append(video_results['productivity_rate'])
    
    # Calculate overall metrics
    if productivity_rates:
        results['overall_metrics']['average_productivity_rate'] = round(np.mean(productivity_rates), 2)
    
    return results

def update_dashboard_with_video_analysis(dashboard_data: Dict[str, Any], video_folder: str) -> Dict[str, Any]:
    """Add video analysis results to dashboard data"""
    print("\nStarting video analysis...")
    video_results = analyze_surveillance_videos(video_folder)
    
    if video_results:
        dashboard_data['video_inference'] = video_results
        print("Video analysis results added to dashboard")
    
    return dashboard_data