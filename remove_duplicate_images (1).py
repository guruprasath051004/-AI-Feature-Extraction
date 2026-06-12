import os
import cv2
import numpy as np
from tqdm import tqdm

def average_hash(image, hash_size=8):
    # Convert to grayscale and resize
    resized = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (hash_size, hash_size))
    avg = resized.mean()
    return ''.join(['1' if pixel > avg else '0' for row in resized for pixel in row])

def remove_duplicates_keep_last(image_folder):
    hashes = {}
    images = sorted(os.listdir(image_folder))

    for image_name in tqdm(images, desc="Processing images"):
        image_path = os.path.join(image_folder, image_name)

        try:
            img = cv2.imread(image_path)
            if img is None:
                continue

            hash_val = average_hash(img)

            # If already exists, delete the previous (to keep only latest one)
            if hash_val in hashes:
                try:
                    os.remove(hashes[hash_val])
                except Exception as e:
                    print(f"Failed to delete {hashes[hash_val]}: {e}")

            # Save the current image as the latest with this hash
            hashes[hash_val] = image_path

        except Exception as e:
            print(f"Error processing {image_name}: {e}")

remove_duplicates_keep_last("F:/final_dataset/fake")