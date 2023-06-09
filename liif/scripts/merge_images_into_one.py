import os
from PIL import Image

def merge_images(image_folder):
    images = []
    for filename in os.listdir(image_folder):
        if (filename.endswith(".jpg") or filename.endswith(".png")) and 'gt' in filename:
            img_path = os.path.join(image_folder, filename)
            img = Image.open(img_path)
            images.append((filename, img))

    num_images = len(images)
    if num_images == 0:
        print("No matching images found in the folder.")
        return

    images.sort(key=lambda x: x[0])  # Sort images based on filenames

    max_images_per_row = 5
    num_rows = (num_images + max_images_per_row - 1) // max_images_per_row
    max_width = max_images_per_row * max(img.width for _, img in images)
    max_height = num_rows * max(img.height for _, img in images)

    merged_image = Image.new('RGB', (max_width, max_height), color='white')

    x_offset = 0
    y_offset = 0

    for _, img in images:
        merged_image.paste(img, (x_offset, y_offset))

        x_offset += img.width
        if x_offset >= max_width:
            x_offset = 0
            y_offset += img.height

    output_path = os.path.join(image_folder, "0_0merged_image.jpg")
    merged_image.save(output_path)
    print("Merged image saved at:", output_path)

# Example usage
folder_path = "test_images/"
merge_images(folder_path)
