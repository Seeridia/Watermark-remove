from PIL import Image
import numpy as np
import os
from tqdm import tqdm

image_folder = 'input_images'
if not os.path.exists(image_folder):
    os.makedirs(image_folder)
image_files = [f for f in os.listdir(image_folder) if f.endswith(('png', 'jpg', 'jpeg'))]
if not image_files:
    print("No images found in the folder.")
    exit()
image = Image.open(os.path.join(image_folder, image_files[0])).convert('RGB')
pixels = np.array(image)

height, width = pixels.shape[:2]


target_color_range = [(217, 250), (217, 250), (217, 250)]


first_pixel_color = pixels[0, 0].copy()


new_pixels = pixels.copy()


for image_file in tqdm(image_files, desc="Processing images", position=0):
    image_path = os.path.join(image_folder, image_file)

    image = Image.open(image_path).convert('RGB')
    pixels = np.array(image)

    height, width = pixels.shape[:2]

    first_pixel_color = pixels[0, 0].copy()

    new_pixels = pixels.copy()

    for i in tqdm(range(height), desc=f"Processing rows of {image_file}", leave=False, ncols=100, position=1):
        for j in range(width):
            if all(target_color_range[k][0] <= pixels[i, j][k] <= target_color_range[k][1] for k in range(3)):

                neighbors = []
                for x in range(max(0, i-1), min(height, i+2)):
                    for y in range(max(0, j-1), min(width, j+2)):
                        if (x, y) != (i, j):
                            neighbors.append(pixels[x, y])
                neighbors = np.array(neighbors)

                has_target_color = np.any([all(target_color_range[k][0] <= neighbor[k] <= target_color_range[k][1] for k in range(3)) for neighbor in neighbors])
                if has_target_color:
                    new_pixels[i, j] = first_pixel_color
                else:
                    avg_color = neighbors.mean(axis=0).astype(int)
                    new_pixels[i, j] = avg_color
    if not os.path.exists('output_images'):
        os.makedirs('output_images')
    new_image = Image.fromarray(new_pixels.astype('uint8'), 'RGB')
    new_image.save(os.path.join('output_images', image_file))

new_image = Image.fromarray(new_pixels.astype('uint8'), 'RGB')