from ultralytics import YOLO

def test_single_image():
    print("=== Single Image Pothole Detection Test ===")
    
    # Load your trained model
    model = YOLO('../results/pothole_detection_stable/weights/best.pt')
    print("✅ Model loaded successfully")
    
    # Test on a specific image (replace with actual image name)
    image_path = '../data/valid/images/pothole_371.jpg'  # Change this filename
    
    # Run prediction
    results = model.predict(
        image_path, 
        conf=0.5,                    # 50% confidence threshold
        save=True,                   # Save annotated image
        show=True,                   # Display image with detections
        project='../single_test_results',
        name='single_image_test'
    )
    
    # Print detection details
    detections = len(results[0].boxes) if results[0].boxes is not None else 0
    print(f"\n📊 Results for: {image_path}")
    print(f"   Potholes detected: {detections}")
    
    if detections > 0:
        confidences = results[0].boxes.conf.cpu().numpy()
        boxes = results[0].boxes.xyxy.cpu().numpy()
        
        for i, (conf, box) in enumerate(zip(confidences, boxes)):
            x1, y1, x2, y2 = box
            print(f"   Pothole {i+1}: {conf:.2%} confidence at [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
    else:
        print("   No potholes detected")
    
    print(f"\n✅ Annotated image saved to: ../single_test_results/single_image_test/")

if __name__ == "__main__":
    test_single_image()
