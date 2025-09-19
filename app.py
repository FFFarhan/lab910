# Smart Traffic Surveillance (Vehicles + Pedestrians + Helmets) - YOLOv8
# Detects cars, buses, trucks, motorbikes, pedestrians, and helmets.

# Install dependencies first:
# pip install ultralytics opencv-python-headless matplotlib tqdm pillow

from ultralytics import YOLO
import os
import shutil
import json
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tqdm import tqdm


# ------------------ Utility Functions ------------------
def show_image(img_path, figsize=(12, 8)):
    img = Image.open(img_path)
    plt.figure(figsize=figsize)
    plt.axis("off")
    plt.imshow(img)
    plt.show()


def ensure_dir(d):
    Path(d).mkdir(parents=True, exist_ok=True)


# ------------------ Dataset & Base Model ------------------
def download_dataset():
    # Using dataset download API directly instead of CLI command
    from ultralytics.data.utils import download
    download(url='https://ultralytics.com/assets/coco128.zip')


def main():
    # Create necessary directories
    ensure_dir("images")
    ensure_dir("results")
    
    print("Please place your images in the 'images' folder")
    print("For videos, place your video file as 'traffic_sample.mp4' in the root directory")
    
    # Check and download dataset if needed
    if not Path("coco128").exists():
        print("Downloading COCO128 dataset...")
        download_dataset()
        print("Dataset downloaded successfully.")

    # Load pretrained YOLOv8 model (traffic detection)
    model = YOLO("yolov8n.pt").to('cuda')
    
    # Load helmet detection model
    helmet_model_path = "helmet_best.pt"
    if Path(helmet_model_path).exists():
        helmet_model = YOLO(helmet_model_path).to('cuda')
        print("Helmet detection model loaded.")
    else:
        helmet_model = None
        print("⚠️ Helmet model not found. Skipping helmet detection.")

    def detect_helmets(img_path, conf=0.25):
        if helmet_model is None:
            return None
        return helmet_model.predict(img_path, conf=conf, save=False, imgsz=640)

    # ------------------ Inference on Sample Image ------------------
    # Try different possible image paths
    possible_paths = [
        "sample.jpg",
        "images/sample.jpg",
        "coco128/images/train2017/000000000785.jpg"
    ]
    
    sample_img = next((p for p in possible_paths if Path(p).exists()), None)
    if sample_img:
        print(f"Using sample image: {sample_img}")
        preds = model.predict(source=sample_img, imgsz=640, conf=0.25,
                            iou=0.45, max_det=300, save=False, show=False)
        out_dir = "results/pred_images"
        ensure_dir(out_dir)
        annotated_path = str(Path(out_dir) / f"annotated_{Path(sample_img).name}")
        preds[0].save(annotated_path)
        print("Traffic annotated image saved to:", annotated_path)
        show_image(annotated_path)
        
        # Calculate counts here to ensure preds is defined
        counts = count_classes_from_results(preds, class_names=COCO_CLASSES,
                                        target_classes=['person','car','truck','bus','motorcycle'])
        print("Counts in sample image:", counts)
    else:
        print("No sample images found. Please add images to the 'images' folder")
        preds = None
        counts = {}

    # ------------------ Combined Annotation (Traffic + Helmets) ------------------
    def annotate_with_helmets(img_path, save_path="results/annotated_with_helmets.jpg"):
        img = cv2.imread(img_path)

        # Traffic detections
        traffic_res = model.predict(source=img_path, imgsz=640, conf=0.25, save=False)
        for r in traffic_res[0].boxes:
            x1, y1, x2, y2 = map(int, r.xyxy[0])
            cls_id = int(r.cls[0])
            label = model.names[cls_id]
            color = (0, 255, 0) if label == "person" else (0, 0, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Helmet detections
        helmet_counts = 0
        if helmet_model:
            helmet_res = detect_helmets(img_path)
            for r in helmet_res[0].boxes:
                x1, y1, x2, y2 = map(int, r.xyxy[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(img, "Helmet", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                helmet_counts += 1

        ensure_dir("results")
        cv2.imwrite(save_path, img)
        print(f"✅ Helmet-annotated image saved at: {save_path}")
        return helmet_counts

    # ------------------ Batch Inference (Traffic only) ------------------
    input_folder = "images"
    if Path(input_folder).exists() and any(Path(input_folder).iterdir()):
        output_folder = "results/pred_images_batch"
        ensure_dir(output_folder)
        model.predict(source=input_folder, imgsz=640, conf=0.25,
                     save=True, save_dir=output_folder, show=False)
        print("Batch inference complete. Images in:", output_folder)
    elif Path("coco128/images/val2017").exists():
        input_folder = "coco128/images/val2017"
        output_folder = "results/pred_images_batch"
        ensure_dir(output_folder)
        model.predict(source=input_folder, imgsz=640, conf=0.25,
                     save=True, save_dir=output_folder, show=False)
        print("Batch inference complete. Images in:", output_folder)
    else:
        print("No images found for batch inference")

    # ------------------ Video Inference ------------------
    video_in = "traffic_sample.mp4"
    video_out_dir = "results/pred_videos"
    ensure_dir(video_out_dir)
    if Path(video_in).exists():
        print(f"Processing video: {video_in}")
        video_results = model.predict(
            source=video_in, 
            imgsz=640, 
            conf=0.35,
            save=True, 
            save_dir=video_out_dir, 
            show=False
        )
        
        # Calculate video statistics
        total_vehicles = {
            'cars': 0,
            'trucks': 0,
            'buses': 0,
            'total_frames': len(video_results)
        }
        
        for frame in video_results:
            names = frame.names
            boxes = frame.boxes
            for box in boxes:
                cls = names[int(box.cls[0])]
                if cls == 'car':
                    total_vehicles['cars'] += 1
                elif cls == 'truck':
                    total_vehicles['trucks'] += 1
                elif cls == 'bus':
                    total_vehicles['buses'] += 1
        
        print("\nVideo Processing Statistics:")
        print(f"Total Frames: {total_vehicles['total_frames']}")
        print(f"Total Cars Detected: {total_vehicles['cars']}")
        print(f"Total Trucks Detected: {total_vehicles['trucks']}")
        print(f"Total Buses Detected: {total_vehicles['buses']}")
        print(f"Annotated video saved to: {video_out_dir}")
    else:
        print("No video file named", video_in, "found. Skipping video inference.")

    # ------------------ Class Counting ------------------
    def count_classes_from_results(results, class_names=None, target_classes=None):
        counts = {}
        if not results:
            return counts
        res = results[0]
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            return counts
        cls_idxs = boxes.cls.cpu().numpy().astype(int)
        for idx in cls_idxs:
            name = class_names[idx] if class_names is not None and idx < len(class_names) else str(idx)
            counts[name] = counts.get(name, 0) + 1
        if target_classes:
            return {k: counts.get(k, 0) for k in target_classes}
        return counts


    COCO_CLASSES = [
        'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light',
        'fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow',
        'elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee',
        'skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard',
        'tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich',
        'orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant','bed',
        'dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven',
        'toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush'
    ]


    # ------------------ Report ------------------
    report = {
        "model": "yolov8n.pt",
        "sample_counts": counts,
        "notes": "Using pretrained YOLOv8n and helmet detection models"
    }

    # Add helmet info - with proper None checking
    if helmet_model and sample_img is not None and Path(sample_img).exists():
        helmet_count = annotate_with_helmets(sample_img, "results/helmet_annotated.jpg")
        report["helmet_count_in_sample"] = helmet_count
        report["helmet_model"] = helmet_model_path
    else:
        report["helmet_count_in_sample"] = "not available"
        report["helmet_detection_status"] = "Skipped - No valid sample image or helmet model"

    # Add video statistics if available
    if Path(video_in).exists():
        report["video_statistics"] = total_vehicles

    ensure_dir("report")
    with open("report/summary.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nReport saved to report/summary.json")


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()
