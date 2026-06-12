import os
import cv2
import gradio as gr
import numpy as np
import torch
import joblib
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import clip
from transformers import BlipProcessor, BlipForConditionalGeneration
from gtts import gTTS
import tempfile

# Load model and device
model = joblib.load("saved_model_lgb.pkl")
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

# Load DINOv2
dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device).eval()
dino_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

# Load BLIP
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

def extract_features(img):
    img = img.convert("RGB")
    dino_tensor = dino_transform(img).unsqueeze(0).to(device)
    clip_tensor = clip_preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        dino_feat = dino_model(dino_tensor).cpu().numpy().squeeze()
        clip_feat = clip_model.encode_image(clip_tensor).cpu().numpy().squeeze()
    return np.concatenate([dino_feat, clip_feat])

def extract_evenly_spaced_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = int(total_frames / fps) if fps else 0  # in seconds
    frames = []
    for sec in range(duration):
        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        ret, frame = cap.read()
        if not ret:
            continue
        rgb = cv2.cvtColor(cv2.resize(frame, (224, 224)), cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(rgb))
    cap.release()
    return frames

def create_confidence_plot(probas):
    plt.figure(figsize=(4, 3))
    bars = plt.bar(["Real", "AI-generated"], probas * 100, color=["green", "red"])
    plt.ylim(0, 100)
    plt.ylabel("Confidence (%)")
    for bar, prob in zip(bars, probas * 100):
        plt.text(bar.get_x() + bar.get_width()/2, prob + 1, f"{prob:.1f}%", ha="center")
    plt.tight_layout()
    return plt

def clean_and_format_captions(captions):
    seen = set()
    cleaned = []
    for cap in captions:
        cap = cap.strip().strip('.')
        if cap.lower() not in seen and cap != "":
            seen.add(cap.lower())
            cleaned.append(cap)
    return ', '.join(cleaned) + '.'

def generate_video_caption(frames, every=10):
    raw_captions = []
    for i in range(0, len(frames), every):
        frame = frames[i].resize((512, 512))
        inputs = blip_processor(images=frame, return_tensors="pt").to(device)
        out = blip_model.generate(**inputs, max_new_tokens=30)
        cap = blip_processor.decode(out[0], skip_special_tokens=True)
        raw_captions.append(cap)
    return clean_and_format_captions(raw_captions)

def text_to_speech(text):
    tts = gTTS(text=text)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp.name)
    return tmp.name

def process_video(video_path):
    if not video_path or not os.path.exists(video_path):
        return "No video", 0.0, [], None, "", None
    frames = extract_evenly_spaced_frames(video_path)
    if not frames:
        return "Failed to extract", 0.0, [], None, "", None
    features = [extract_features(img) for img in frames]
    video_feat = np.mean(features, axis=0).reshape(1, -1)
    pred = model.predict(video_feat)[0]
    probas = model.predict_proba(video_feat)[0]
    label = "Real" if pred == 0 else "AI-generated"
    confidence = round(probas[pred] * 100, 2)
    graph = create_confidence_plot(probas)
    caption = generate_video_caption(frames)
    audio_path = text_to_speech(caption)
    return label, confidence, frames, graph, caption, audio_path

def process_image(img: Image.Image):
    if img is None:
        return "No image", 0.0, None, None, None, None
    features = extract_features(img).reshape(1, -1)
    pred = model.predict(features)[0]
    probas = model.predict_proba(features)[0]
    label = "Real" if pred == 0 else "AI-generated"
    confidence = round(probas[pred] * 100, 2)
    resized = img.resize((512, 512))
    inputs = blip_processor(images=resized, return_tensors="pt").to(device)
    cap_ids = blip_model.generate(**inputs, max_new_tokens=30)
    caption = blip_processor.decode(cap_ids[0], skip_special_tokens=True).strip().strip('.') + '.'
    graph = create_confidence_plot(probas)
    audio_path = text_to_speech(caption)
    return label, confidence, img, graph, caption, audio_path

with gr.Blocks() as demo:
    gr.Markdown("##  AI Media Detection & Moderation System")

    with gr.Tab(" Video"):
        with gr.Row():
            upload = gr.Video(label=" Upload Video", source="upload", type="filepath", format="mp4")
            webcam = gr.Video(label=" Webcam", source="webcam", type="filepath", format="mp4")
        selected = gr.State()
        gr.Button(" Select").click(lambda u, w: w if w else u, [upload, webcam], selected)
        gr.Button(" Predict").click(process_video, inputs=selected, outputs=[
            gr.Text(label="Prediction"),
            gr.Number(label="Accuracy (%)"),
            gr.Gallery(label="Sample Frames", columns=6),
            gr.Plot(label="Confidence Graph"),
            gr.Textbox(label="Generated Prompt / Explanation"),
            gr.Audio(label=" Prompt Audio", autoplay=True)
        ])

    with gr.Tab(" Image"):
        image_input = gr.Image(type="pil", label="Upload Image")
        gr.Button("Classify Image").click(process_image, inputs=image_input, outputs=[
            gr.Text(label="Prediction"),
            gr.Number(label="Confidence (%)"),
            gr.Image(label="Image Preview"),
            gr.Plot(label="Confidence Graph"),
            gr.Textbox(label="Generated Caption"),
            gr.Audio(label=" Caption Audio", autoplay=True)
        ])

demo.launch(share=True)
