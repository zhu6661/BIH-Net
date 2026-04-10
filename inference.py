import torch
import os
import numpy as np
from VstepT4gai import CombinedModel, Config

def run_inference():
    # 1. Setup device and configuration
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Initialize the model 
    # Ensure dimensions match your BIH-Net architectural parameters
    model = CombinedModel(
        mri_shape=config.mri_img_size,
        clinical_feature_dim=15, 
        num_classes=2
    ).to(device)
    
    # 3. Define path to the uploaded .pt file
    model_path = os.path.join('weights', 'checkpoint_best.pt')
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    # 4. Load weights
    try:
        checkpoint = torch.load(model_path, map_location=device)
        # Handle different saving formats (full checkpoint dict vs state_dict only)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        print(f"Model successfully loaded from {model_path}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 5. Generate dummy multimodal input for demonstration
    # MRI: (Batch, Channel, D, H, W)
    dummy_mri = torch.randn(1, 1, 64, 64, 64).to(device)
    # Clinical: (Batch, Features)
    dummy_clinical = torch.randn(1, 15).to(device)
    # graph Embedding: (Batch, Embedding_Dim)
    dummy_graph = torch.randn(1, 128).to(device)

    # 6. Forward pass
    with torch.no_grad():
        output, _ = model(dummy_mri, dummy_clinical, dummy_kg)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        
        print("-" * 30)
        print(f"Inference complete.")
        print(f"Healthy Probability: {probabilities[0][0]:.4f}")
        print(f"PD Probability: {probabilities[0][1]:.4f}")
        print(f"Predicted Class: {'PD' if predicted_class == 1 else 'Healthy'}")
        print("-" * 30)

if __name__ == "__main__":
    run_inference()
