import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction, predict
import os
import shutil
import json
from datetime import datetime
import traceback

def log_message(message):
    """Log message with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")
    
    # Also save to log file
    os.makedirs("detection_logs", exist_ok=True)
    with open(f"detection_logs/detection_{datetime.now().strftime('%Y%m%d')}.log", "a") as f:
        f.write(f"[{timestamp}] {message}\n")

def save_detection_summary(processed_files, total_detections, errors):
    """Save detection summary to JSON"""
    summary = {
        "timestamp": datetime.now().isoformat(),
        "processed_files": processed_files,
        "total_detections": total_detections,
        "errors": errors,
        "files_with_detections": sum(1 for f in processed_files if f["detections"] > 0),
        "success_rate": len([f for f in processed_files if f["status"] == "success"]) / len(processed_files) if processed_files else 0
    }
    
    os.makedirs("detection_logs", exist_ok=True)
    with open("detection_logs/detection_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

def process_image(image_path, detection_model, output_dir):
    """Process a single image and return detection results"""
    try:
        log_message(f"Processing: {image_path}")
        
        # Get prediction
        result = get_prediction(image_path, detection_model)
        
        # Read image to get dimensions
        img = cv2.imread(image_path)
        if img is None:
            log_message(f"ERROR: Could not read image {image_path}")
            return {"status": "error", "error": "Could not read image", "detections": 0}
        
        h, w = img.shape[:2]
        
        # Prepare output file
        filename = os.path.splitext(os.path.basename(image_path))[0]
        output_file = os.path.join(output_dir, f"{filename}.txt")
        
        # Process detections
        detections = []
        log_message(f"Found {len(result.object_prediction_list)} detections in {filename}")
        
        with open(output_file, 'w') as f:  # Use 'w' instead of 'a' to overwrite
            for object_prediction in result.object_prediction_list:
                bbox = object_prediction.bbox
                x1, y1, x2, y2 = bbox.minx, bbox.miny, bbox.maxx, bbox.maxy
                conf = object_prediction.score.value
                class_id = object_prediction.category.id
                
                # Calculate normalized coordinates (YOLO format)
                center_x = (x1 + x2) / 2
                normalized_cx = center_x / w
                center_y = (y1 + y2) / 2
                normalized_cy = center_y / h
                width_x = abs(x1 - x2)
                normalized_wx = width_x / w
                width_y = abs(y1 - y2)
                normalized_wy = width_y / h
                
                # Format: class_id center_x center_y width height
                detection_line = f"{class_id} {normalized_cx:.6f} {normalized_cy:.6f} {normalized_wx:.6f} {normalized_wy:.6f}"
                
                detections.append({
                    "class_id": class_id,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2],
                    "normalized": [normalized_cx, normalized_cy, normalized_wx, normalized_wy]
                })
                
                f.write(detection_line + "\n")
                log_message(f"  Detection: {detection_line} (conf: {conf:.3f})")
        
        return {
            "status": "success",
            "detections": len(detections),
            "detection_data": detections,
            "output_file": output_file
        }
        
    except Exception as e:
        error_msg = f"Error processing {image_path}: {str(e)}"
        log_message(f"ERROR: {error_msg}")
        log_message(f"Traceback: {traceback.format_exc()}")
        return {
            "status": "error",
            "error": error_msg,
            "detections": 0
        }

def main():
    log_message("Starting object detection processing...")
    
    # Configuration
    model_path = "model.pt"
    confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
    input_path = os.getenv("INPUT_PATH", "output")
    
    log_message(f"Configuration:")
    log_message(f"  Model path: {model_path}")
    log_message(f"  Confidence threshold: {confidence_threshold}")
    log_message(f"  Input path: {input_path}")
    
    # Check if model file exists
    if not os.path.exists(model_path):
        log_message(f"ERROR: Model file not found at {model_path}")
        return
    
    # Check if input directory exists
    if not os.path.exists(input_path):
        log_message(f"ERROR: Input directory not found at {input_path}")
        return
    
    try:
        # Initialize detection model
        log_message("Loading detection model...")
        detection_model = AutoDetectionModel.from_pretrained(
            model_type='ultralytics',
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            device="cpu"  # Use CPU for GitHub Actions
        )
        log_message("Model loaded successfully!")
        
    except Exception as e:
        log_message(f"ERROR: Failed to load model: {e}")
        log_message(f"Traceback: {traceback.format_exc()}")
        return
    
    # Find images to process
    valid_extensions = [".jpg", ".jpeg", ".png", ".gif", ".tga", ".bmp"]
    image_files = []
    
    for f in os.listdir(input_path):
        ext = os.path.splitext(f)[1].lower()
        if ext in valid_extensions:
            image_files.append(os.path.join(input_path, f))
    
    log_message(f"Found {len(image_files)} images to process")
    
    if not image_files:
        log_message("No images found to process")
        return
    
    # Process images
    processed_files = []
    total_detections = 0
    errors = []
    
    for i, image_path in enumerate(image_files):
        log_message(f"Processing image {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
        
        result = process_image(image_path, detection_model, input_path)
        
        file_info = {
            "filename": os.path.basename(image_path),
            "status": result["status"],
            "detections": result["detections"]
        }
        
        if result["status"] == "error":
            file_info["error"] = result["error"]
            errors.append(result["error"])
        else:
            total_detections += result["detections"]
        
        processed_files.append(file_info)
        
        # Progress update every 10 files
        if (i + 1) % 10 == 0:
            log_message(f"Progress: {i+1}/{len(image_files)} files processed")
    
    # Save summary
    save_detection_summary(processed_files, total_detections, errors)
    
    # Final summary
    successful_files = len([f for f in processed_files if f["status"] == "success"])
    failed_files = len([f for f in processed_files if f["status"] == "error"])
    
    log_message("="*50)
    log_message("DETECTION SUMMARY")
    log_message("="*50)
    log_message(f"Total images processed: {len(image_files)}")
    log_message(f"Successfully processed: {successful_files}")
    log_message(f"Failed to process: {failed_files}")
    log_message(f"Total detections found: {total_detections}")
    log_message(f"Average detections per image: {total_detections/successful_files:.2f}" if successful_files > 0 else "Average detections per image: 0")
    
    if errors:
        log_message(f"Errors encountered:")
        for error in set(errors[:5]):  # Show unique errors, max 5
            log_message(f"  - {error}")
    
    log_message("Object detection processing complete!")

if __name__ == "__main__":
    main()
