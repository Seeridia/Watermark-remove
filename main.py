from PIL import Image
import numpy as np
import os
from tqdm import tqdm

image_folder = 'input_images'
output_folder = 'output_images'

if not os.path.exists(image_folder):
    os.makedirs(image_folder)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

image_files = [f for f in os.listdir(image_folder) if f.endswith(('png', 'jpg', 'jpeg'))]
if not image_files:
    print("No images found in the folder.")
    exit()

target_color_range = [(178, 250), (178, 250), (178, 250)]

def is_target_color(pixel):
    return all(target_color_range[k][0] <= pixel[k] <= target_color_range[k][1] for k in range(3))

for image_file in tqdm(image_files, desc="Processing images"):
    image_path = os.path.join(image_folder, image_file)
    image = Image.open(image_path).convert('RGB')
    pixels = np.array(image)
    height, width = pixels.shape[:2]
    first_pixel_color = pixels[0, 0].copy()
    new_pixels = pixels.copy()

    mask = np.all([(target_color_range[k][0] <= pixels[:, :, k]) & (pixels[:, :, k] <= target_color_range[k][1]) for k in range(3)], axis=0)

    for i in range(height):
        for j in range(width):
            if mask[i, j]:
                neighbors = pixels[max(0, i-1):min(height, i+2), max(0, j-1):min(width, j+2)].reshape(-1, 3)
                neighbors_mask = np.all([(target_color_range[k][0] <= neighbors[:, k]) & (neighbors[:, k] <= target_color_range[k][1]) for k in range(3)], axis=0)
                if np.any(neighbors_mask):
                    new_pixels[i, j] = first_pixel_color
                else:
                    new_pixels[i, j] = neighbors[~neighbors_mask].mean(axis=0).astype(int)

    new_image = Image.fromarray(new_pixels.astype('uint8'), 'RGB')
    new_image.save(os.path.join(output_folder, image_file))
