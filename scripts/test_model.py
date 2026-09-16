from ultralytics import YOLO
import os

def test_pothole_detection():
    print("=== Testing YOLOv8 Pothole Detection Model ===")
    
    # Load your trained model
    model_path = '../results/pothole_detection_stable/weights/best.pt'
    model = YOLO(model_path)
    print(f"✅ Loaded trained model from: {model_path}")
    
    # Test on validation images (these weren't used for training)
    image_folder = '../data/valid/images'
    
    if os.path.exists(image_folder):
        # Get list of available images
        images = [f for f in os.listdir(image_folder) 
                 if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        print(f"Found {len(images)} images for testing")
        
        # Test on first 5 images as samples
        sample_images = images[:5]
        
        for i, img_name in enumerate(sample_images, 1):
            img_path = os.path.join(image_folder, img_name)
            print(f"\n[{i}/5] Testing: {img_name}")
            
            # Run prediction with confidence threshold of 50%
            results = model.predict(
                img_path, 
                conf=0.5,           # 50% confidence threshold
                save=True,          # Save annotated images
                project='../test_results',
                name='pothole_test'
            )
            
            # Print detection results
            detections = len(results[0].boxes) if results[0].boxes is not None else 0
            print(f"   Detected {detections} potholes")
            
            # Show confidence scores
            if detections > 0:
                confidences = results[0].boxes.conf.cpu().numpy()
                for j, conf in enumerate(confidences):
                    print(f"   Pothole {j+1}: {conf:.2%} confidence")
        
        print(f"\n✅ Testing completed!")
        print(f"📁 Annotated images saved to: ../test_results/pothole_test/")
        
    else:
        print(f"❌ Image folder not found: {image_folder}")

if __name__ == "__main__":
    test_pothole_detection()
