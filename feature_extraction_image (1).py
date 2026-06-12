import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import torchvision.transforms as T
from torchvision.models import vit_b_16

# === Paths ===
image_root = r"F:/ai_image_dataset"
output_dir = r"F:/ai_image_features2"
os.makedirs(output_dir, exist_ok=True)

# === CLIP Setup ===
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

# === DINOv2 Setup ===
dino_model = vit_b_16(weights='IMAGENET1K_V1')
dino_model.eval()

# === Transform for DINO ===
dino_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

clip_features = []
dino_features = []
labels = []
image_names = []

class_map = {"real": 0, "fake": 1}

for label_name in os.listdir(image_root):
    label_path = os.path.join(image_root, label_name)
    if not os.path.isdir(label_path):
        continue
    label = class_map[label_name]
    
    for img_file in tqdm(os.listdir(label_path), desc=f"Processing {label_name}"):
        img_path = os.path.join(label_path, img_file)
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            print(f"Skipped corrupted image: {img_path}")
            continue

        # CLIP feature
        inputs = clip_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            clip_output = clip_model.get_image_features(**inputs)
        clip_feat = clip_output[0].cpu().numpy()

        # DINO feature
        dino_input = dino_transform(image).unsqueeze(0)
        with torch.no_grad():
            dino_output = dino_model(dino_input)
        dino_feat = dino_output[0].cpu().numpy()

        clip_features.append(clip_feat)
        dino_features.append(dino_feat)
        labels.append(label)
        image_names.append(img_file)

# === Save Features ===
np.save(os.path.join(output_dir, "clip_features.npy"), np.array(clip_features))
np.save(os.path.join(output_dir, "dino_features.npy"), np.array(dino_features))
np.save(os.path.join(output_dir, "labels.npy"), np.array(labels))
np.save(os.path.join(output_dir, "image_names.npy"), np.array(image_names))

print("✅ Feature extraction completed.")