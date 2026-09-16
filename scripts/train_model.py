from ultralytics import YOLO
import torch
import os

def main():
    print("=== YOLOv8 Pothole Detection Training (Stable Configuration) ===")
    
    # Change to data directory
    os.chdir('D:/pothole_detection_project/data')
    
    # Verify GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Use YOLOv8n model
    model = YOLO('yolov8n.pt')
    
    # Ultra-stable configuration for GTX 1650
    results = model.train(
        data='data.yaml',
        epochs=100,
        imgsz=416,
        batch=2,
        device=device,
        workers=0,                   # CRITICAL: No multiprocessing
        cache=False,                 # CRITICAL: Disable caching
        amp=True,
        project='../results',
        name='pothole_detection_stable',
        save=True,
        patience=20,
        save_period=10,
        rect=False,                  # Disable rectangular training
        single_cls=True              # Single class training (pothole only)
    )
    
    print("✅ Training completed!")
    print(f"📁 Best model saved at: ../results/pothole_detection_stable/weights/best.pt")
    
    # Quick evaluation
    try:
        metrics = model.val()
        print("📈 Final Performance:")
        print(f"mAP50: {metrics.box.map50:.1%}")
        print(f"Precision: {metrics.box.mp:.1%}")
        print(f"Recall: {metrics.box.mr:.1%}")
    except Exception as e:
        print(f"Evaluation completed, but metrics display failed: {e}")

if __name__ == "__main__":
    main()
