import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
from torchvision.models.vision_transformer import vit_b_16 , ViT_B_16_Weights
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ViT-B/16 (DINOv2 equivalent via torchvision weights)
weights = ViT_B_16_Weights.DEFAULT
model = vit_b_16(weights=weights).to(device)
model.eval()

# Transform setup
transform = weights.transforms()

# Path setup
input_dir = "F:/ai_video_frames"
output_dir = "F:/dinov2_features"
os.makedirs(output_dir, exist_ok=True)

output_feature_path = os.path.join(output_dir, "features.npy")
output_label_path = os.path.join(output_dir, "labels.npy")

splits = ['train', 'val', 'test']
labels_map = {'real': 0, 'fake': 1}

features = []
labels = []

for split in splits:
    for label_name, label_id in labels_map.items():
        split_path = os.path.join(input_dir, split, label_name)
        if not os.path.exists(split_path):
            continue

        for video_folder in tqdm(os.listdir(split_path), desc=f"{split}/{label_name}"):
            video_path = os.path.join(split_path, video_folder)
            if not os.path.isdir(video_path):
                continue

            frame_features = []
            for frame_file in sorted(os.listdir(video_path))[:16]:
                frame_path = os.path.join(video_path, frame_file)
                try:
                    image = Image.open(frame_path).convert("RGB")
                    img_tensor = transform(image).unsqueeze(0).to(device)
                    with torch.no_grad():
                        feat = model(img_tensor)
                    frame_features.append(feat.squeeze(0).cpu().numpy())
                except Exception as e:
                    print(f"Skipping frame {frame_path}: {e}")

            if frame_features:
                avg_feature = np.mean(frame_features, axis=0)
                features.append(avg_feature)
                labels.append(label_id)

# Save to .npy
np.save(output_feature_path, np.array(features))
np.save(output_label_path, np.array(labels))

print("✅ DINOv2 feature extraction completed.")
print(f"Saved features to: {output_feature_path}")
print(f"Saved labels to: {output_label_path}")