import cv2
import os
import shutil

# === Config ===
input_folder = r"F:\add_new_vid"
output_base_folder = r"F:\add_new_frames"
max_frames = 300

# === Prepare output folder ===
os.makedirs(output_base_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if not filename.lower().endswith(".mp4"):
        continue

    video_path = os.path.join(input_folder, filename)
    video_name = os.path.splitext(filename)[0]
    output_folder = os.path.join(output_base_folder, video_name)
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[✗] Failed to open {filename}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    if duration <= 0 or total_frames == 0:
        print(f"[✗] Skipping {filename} (invalid duration or frame count)")
        cap.release()
        continue

    time_interval_sec = duration / max_frames
    saved_count = 0
    last_good_frame = None

    for i in range(max_frames):
        timestamp_ms = i * time_interval_sec * 1000  # target timestamp in ms
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)
        ret, frame = cap.read()

        # If fail, try small offset
        if not ret:
            for offset_ms in [-20, 20, -40, 40]:  # try earlier/later
                cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms + offset_ms)
                ret, frame = cap.read()
                if ret:
                    print(f"[!] Recovered frame at { (timestamp_ms+offset_ms)/1000:.2f}s with offset {offset_ms}ms")
                    break

        # If still fail, duplicate last frame
        if not ret and last_good_frame is not None:
            frame = last_good_frame.copy()
            print(f"[!] Could not read frame at {timestamp_ms/1000:.2f}s → duplicated last good frame")

        if ret or frame is not None:
            output_path = os.path.join(output_folder, f"frame_{i:04d}.jpg")
            cv2.imwrite(output_path, frame)
            last_good_frame = frame
            saved_count += 1
        else:
            print(f"[X] Could not get any frame at index {i}")

    cap.release()
    print(f"[✓] Saved {saved_count} frames from {filename} → {output_folder}")

print("✅ All videos processed (always 300 frames each)")