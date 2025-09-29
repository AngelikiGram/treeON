
import cv2
import os
import numpy as np
from pathlib import Path

list_folders = ['DSM'] # Cl_colors', 'test']
supersample = False 
supersample_trunk = False # True 

if supersample:
    additive_folder = "supersample"
elif supersample_trunk:
    additive_folder = "supersample-trunk"

# Set input and output folder
input_folder_base = r"C:\Users\mmddd\Documents\network-tree-gen\landmarks_austria\outputs"

output_folder_base = os.path.join(input_folder_base, "CROPPED")
os.makedirs(output_folder_base, exist_ok=True)

for i in range(len(list_folders)):
    input_folder = os.path.join(input_folder_base, list_folders[i])

    output_folder = os.path.join(output_folder_base, list_folders[i])
    os.makedirs(output_folder, exist_ok=True)

    if supersample or supersample_trunk:
        input_folder = os.path.join(input_folder, additive_folder)

        output_folder = os.path.join(output_folder, additive_folder)
        os.makedirs(output_folder, exist_ok=True)

    # Image formats to process
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp"}

    def crop_and_white_bg(image, threshold=10):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Create a binary mask: 1 = content, 0 = black background
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        
        coords = cv2.findNonZero(mask)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            cropped = image[y:y+h, x:x+w]
            cropped_mask = mask[y:y+h, x:x+w]

            # Create white background
            white_bg = np.ones_like(cropped) * 255

            # Apply mask to keep only foreground, paste on white
            cropped_result = np.where(cropped_mask[:, :, None] == 255, cropped, white_bg)
            return cropped_result
        else:
            return image  # No content found

    # Get parent folder name
    parent_folder_name = os.path.basename(input_folder)

    # Process each image
    for filename in os.listdir(input_folder):
        ext = Path(filename).suffix.lower()
        
        if ext == ".png": #  
            filepath = os.path.join(input_folder, filename)
            image = cv2.imread(filepath)

            if image is not None:
                result = crop_and_white_bg(image)
                
                # Add parent folder name as prefix to the filename
                new_filename = f"{parent_folder_name}_{filename}"
                save_path = os.path.join(output_folder, new_filename)
                
                cv2.imwrite(save_path, result)
                print(f"Cropped and saved: {save_path}")
            else:
                print(f"Failed to load image: {filename}")
        else:
            print(f"Skipping: {filename} (not a PNG with 72 or 31)")
