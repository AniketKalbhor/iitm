import os
import torch
import numpy as np
from PIL import Image
import datetime
import torchvision.transforms as transforms
from transformers import CLIPVisionModel

# Import your SCAFET-style feature detector (make sure this file exists)
from feature_detector import enhanced_climate_processor

def enhanced_pooler_encoder(date_str):
    """
    Extract both CLIP features AND physical climate features for a given date.
    """
    # Path to the RGB image for the given date
    rgb_path = f"rgb_images/rgb_image_{date_str}.png"
    if not os.path.exists(rgb_path):
        print(f"Skipped {date_str}: '{rgb_path}' not found.")
        return

    # CLIP model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711])
    ])

    # Load and preprocess the image
    image = Image.open(rgb_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Extract CLIP pooled features
    with torch.no_grad():
        outputs = model(image_tensor)
        pooled_features = outputs.pooler_output.squeeze().cpu().numpy()  # Shape: (1024,)

    # Extract physical features using SCAFET-style detection
    physical_features = enhanced_climate_processor(date_str)

    # Combine both feature types
    combined_features = {
        'clip_features': pooled_features,
        'physical_features': physical_features,
        'date': date_str
    }

    # Save combined features
    os.makedirs("combined_features", exist_ok=True)
    np.save(f"combined_features/{date_str}_all_features.npy", combined_features)
    print(f"Saved combined features for {date_str}.")

    return combined_features

if __name__ == "__main__":
    # Example: process a single date, or loop as needed
    date_str = "2024-06-17"
    enhanced_pooler_encoder(date_str)
