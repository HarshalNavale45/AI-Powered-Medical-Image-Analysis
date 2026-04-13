import argparse
import os
import cv2
import numpy as np
import tensorflow as tf
import subprocess
import sys
from src.utils import generate_dummy_data, log_diagnostic_event
from src.data_prep import get_data_generators
from src.model import build_transfer_learning_model
from src.train import train_model
from src.evaluate import plot_training_history, evaluate_model
from src.explainability import generate_gradcam, save_and_display_gradcam, generate_clinical_report
from tensorflow.keras.models import load_model

def print_header():
    print("\n" + "="*60)
    print("      🏥 AI MEDICAL IMAGE ANALYSIS SYSTEM - v1.0")
    print("      Development: Industrial-Grade Student Portfolio")
    print("="*60 + "\n")

def main():
    parser = argparse.ArgumentParser(description="🏥 AI-Powered Medical Image Analysis System")
    parser.add_argument('--setup', action='store_true', help='Generate high-quality simulated data')
    parser.add_argument('--train', action='store_true', help='Train the deep learning model')
    parser.add_argument('--evaluate', action='store_true', help='Perform deep clinical evaluation')
    parser.add_argument('--predict', type=str, help='Path to an X-ray image for prediction')
    parser.add_argument('--explain', action='store_true', help='Generate heatmap for prediction')
    parser.add_argument('--dashboard', action='store_true', help='🚀 Launch the Premium Desktop Dashboard')
    
    args = parser.parse_args()
    print_header()

    if args.dashboard:
        print("[INFO] 🚀 Launching Premium Web Dashboard...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])

    elif args.setup:
        generate_dummy_data()
    
    elif args.train:
        if not os.path.exists('data/train'):
            print("[ERROR] ❌ Data folders not set up. Run 'python main.py --setup' first.")
            return
            
        train_gen, val_gen, _ = get_data_generators('data/')
        model = build_transfer_learning_model()
        
        history = train_model(model, train_gen, val_gen, epochs=5) 
        plot_training_history(history)
        
        # Auto-evaluate after training
        _, _, test_gen = get_data_generators('data/')
        evaluate_model(model, test_gen)
        print("[SUCCESS] ✅ Full Training & Evaluation Pipeline Complete.")

    elif args.evaluate:
        if not os.path.exists('models/best_medical_model.keras'):
            print("[ERROR] ❌ No trained model found.")
            return
        
        _, _, test_gen = get_data_generators('data/')
        model = load_model('models/best_medical_model.keras')
        evaluate_model(model, test_gen)

    elif args.predict:
        if not os.path.exists('models/best_medical_model.keras'):
            print("[ERROR] ❌ Model not found. Please train first.")
            return
            
        print(f"[INFO] 🧠 Analyzing Diagnostic Input: {args.predict}...")
        model = load_model('models/best_medical_model.keras')
        
        img = cv2.imread(args.predict)
        if img is None:
            print("[ERROR] ❌ Invalid image path.")
            return
            
        img_resized = cv2.resize(img, (224, 224))
        img_normalized = img_resized / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        prediction = model.predict(img_batch)[0][0]
        diagnosis = "PNEUMONIA DETECTED" if prediction > 0.5 else "NORMAL (HEALTHY)"
        confidence = prediction if prediction > 0.5 else (1 - prediction)
        
        print("\n" + "🩺 " + "-"*50)
        print("          AI DIAGNOSTIC FINDINGS")
        print("-"*52)
        print(f"  FILE: {os.path.basename(args.predict)}")
        print(f"  RESULT: {diagnosis}")
        print(f"  CONFIDENCE: {confidence*100:.2f}%")
        print("-"*52)

        if args.explain:
            heatmap = generate_gradcam(model, img_batch, 'Conv_1')
            save_and_display_gradcam(args.predict, heatmap)
            generate_clinical_report(args.predict, diagnosis, confidence, heatmap)

    else:
        parser.print_help()
        print("\n[TIP] Launch the dashboard with: python main.py --dashboard")

if __name__ == '__main__':
    main()
