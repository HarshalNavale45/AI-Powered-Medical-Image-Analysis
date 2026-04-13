import os
import cv2
import numpy as np

def generate_dummy_data(base_path="data", samples=50):
    """
    Generates high-quality simulated medical images (X-rays) for pipeline testing.
    This ensures the project is 'Plug and Play' for recruiters or students.
    """
    classes = ['Normal', 'Pneumonia']
    splits = ['train', 'val', 'test']
    
    print(f"[INFO] 🏥 Generating simulated Medical Dataset in '{base_path}'...")
    
    for split in splits:
        for cls in classes:
            folder_path = os.path.join(base_path, split, cls)
            os.makedirs(folder_path, exist_ok=True)
            
            # Adjust samples per split
            if split == 'train':
                n_samples = samples
            elif split == 'val':
                n_samples = samples // 4
            else:
                n_samples = samples // 5

            for i in range(n_samples):
                img_path = os.path.join(folder_path, f"patient_{split}_{i}.jpg")
                if not os.path.exists(img_path):
                    # Create a more realistic simulated X-ray (grayscale with vertical 'rib' patterns)
                    img = np.zeros((224, 224), dtype=np.uint8)
                    
                    # Simulate lung fields (two darker areas)
                    cv2.ellipse(img, (70, 112), (40, 80), 0, 0, 360, 100, -1)
                    cv2.ellipse(img, (154, 112), (40, 80), 0, 0, 360, 100, -1)
                    
                    # Add noise to simulate X-ray grain
                    noise = np.random.normal(0, 15, (224, 224)).astype(np.uint8)
                    img = cv2.add(img, noise)
                    
                    # If Pneumonia, add cloudy patches
                    if cls == 'Pneumonia':
                        for _ in range(5):
                            center = (np.random.randint(50, 170), np.random.randint(50, 170))
                            cv2.circle(img, center, np.random.randint(10, 30), 180, -1)
                    
                    # Convert to 3-channel for MobileNetV2 compatibility
                    img_3ch = cv2.merge([img, img, img])
                    
                    # Add standard medical annotation
                    cv2.putText(img_3ch, f"PROV-SAMP: {cls.upper()}", (10, 210), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    
                    cv2.imwrite(img_path, img_3ch)

    print(f"[SUCCESS] Clinical Simulation Dataset created successfully.")

def log_diagnostic_event(message):
    """Utility for professional-looking terminal logging."""
    print(f"[LOG] {message}")
