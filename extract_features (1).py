import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import clip
import torchvision.transforms as T
import torchvision.models as models
import timm  # for DINOv2

# Paths
frames_root = "F:/ai_video_frames"
dino_save_root = "features_dino"
clip_save_root = "features_clip"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load DINOv2
dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device).eval()
dino_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

# Load CLIP
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

# Create output folders
os.makedirs(dino_save_root, exist_ok=True)
os.makedirs(clip_save_root, exist_ok=True)

for class_name in os.listdir(frames_root):
    class_path = os.path.join(frames_root, class_name)
    if not os.path.isdir(class_path):
        continue

    dino_class_out = os.path.join(dino_save_root, class_name)
    clip_class_out = os.path.join(clip_save_root, class_name)
    os.makedirs(dino_class_out, exist_ok=True)
    os.makedirs(clip_class_out, exist_ok=True)

    for video_id in tqdm(os.listdir(class_path), desc=f"Processing {class_name}"):
        video_folder = os.path.join(class_path, video_id)
        dino_out_file = os.path.join(dino_class_out, f"{video_id}.npy")
        clip_out_file = os.path.join(clip_class_out, f"{video_id}.npy")

        # ✅ Pause & Resume logic
        if os.path.exists(dino_out_file) and os.path.exists(clip_out_file):
            continue  # already processed

        dino_feats, clip_feats = [], []

        for frame_file in sorted(os.listdir(video_folder)):
            frame_path = os.path.join(video_folder, frame_file)
            try:
                img = Image.open(frame_path).convert("RGB")

                # DINOv2
                dino_input = dino_transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    dino_feat = dino_model(dino_input).cpu().numpy().squeeze()
                dino_feats.append(dino_feat)

                # CLIP
                clip_input = clip_preprocess(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    clip_feat = clip_model.encode_image(clip_input).cpu().numpy().squeeze()
                clip_feats.append(clip_feat)

            except Exception as e:
                print(f"❌ Error with {frame_path}: {e}")
                continue

        if dino_feats:
            np.save(dino_out_file, np.mean(dino_feats, axis=0))
        if clip_feats:
            np.save(clip_out_file, np.mean(clip_feats, axis=0))