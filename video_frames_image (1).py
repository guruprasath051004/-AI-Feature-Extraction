import os
import shutil

def copy_and_rename_images(src_folder, dest_folder, start_index=0):
    os.makedirs(dest_folder, exist_ok=True)
    count = start_index

    for root, dirs, files in os.walk(src_folder):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                src_path = os.path.join(root, file)
                ext = os.path.splitext(file)[1]
                dst_path = os.path.join(dest_folder, f"image_{count}{ext}")
                shutil.copy2(src_path, dst_path)
                count += 1

    return count

# Define source and destination
real_src = r"F:/ai_video_frames/real_2500"
fake_src = r"F:/ai_video_frames/ai_2500"

real_dest = r"F:/final_dataset/real"
fake_dest = r"F:/final_dataset/fake"

# Copy real then fake, ensuring unique naming
next_index = copy_and_rename_images(real_src, real_dest, start_index=0)
copy_and_rename_images(fake_src, fake_dest, start_index=next_index)

print("✅ All frames renamed and copied uniquely to real/ and fake/ folders.")