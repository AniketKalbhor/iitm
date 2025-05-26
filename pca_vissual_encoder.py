import torch
from PIL import Image
import os
from transformers import CLIPVisionModel, CLIPImageProcessor
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import datetime

def process_image(date, model, transform, device, output_dir):
    input_file = f"rgb_images/rgb_image_{date}.png"
    if not os.path.exists(input_file):
        print(f"Skipped {date}: '{input_file}' not found.")
        return

    image = Image.open(input_file).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor, output_hidden_states=True)
    features = outputs.hidden_states[-1].squeeze().cpu().numpy()

    patch_features = features[1:]  # remove CLS token
    pca = PCA(n_components=3)
    reduced = pca.fit_transform(patch_features)
    vis = reduced.reshape(16, 16, 3)
    vis = (vis - vis.min()) / (vis.max() - vis.min())

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{date}_features_vis.png")
    plt.imshow(vis)
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711])
    ])
    output_dir = "visual_features"

    # Loop over 2024
    start_date = datetime.date(2024, 1, 1)
    end_date = datetime.date(2024, 12, 31)
    current_date = start_date

    while current_date <= end_date:
        date_str = current_date.isoformat()
        process_image(date_str, model, transform, device, output_dir)
        current_date += datetime.timedelta(days=1)

if __name__ == "__main__":
    main()
