import os
from PIL import Image
import cv2
import numpy as np

def reorder_list_with_indices(input_list, index_list):
    if len(input_list) != len(index_list):
        raise ValueError("Input list and index list must have the same length")

    return [input_list[i] for i in index_list]

def sort_by_edginess(images):

    num_edges_list = []

    for _, img in images:

        image_np = np.asarray(img)

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

        # Convert the grayscale image to 8-bit unsigned integer
        gray_image = np.uint8(gray_image * 255.0)

        # Apply Canny edge detection using OpenCV
        edge_map = cv2.Canny(gray_image, threshold1=100, threshold2=200)

        num_edges_list.append(np.sum(edge_map))

    indices = np.argsort(num_edges_list).tolist()
    
    sorted_images = reorder_list_with_indices(images, indices)
    #print(sorted_images)

    return sorted_images


def merge_images(image_folder):
    images = []
    for filename in os.listdir(image_folder):
        if (filename.endswith(".jpg") or filename.endswith(".png")): #and 'gt' in filename:
            img_path = os.path.join(image_folder, filename)
            img = Image.open(img_path)
            images.append((filename, img))

    num_images = len(images)
    if num_images == 0:
        print("No matching images found in the folder.")
        return
    
    images.sort(key=lambda x: x[0])  # Sort images based on filenames
    
    # TODO: rmeove
    images = sort_by_edginess(images)

    max_images_per_row = 6
    num_rows = (num_images + max_images_per_row - 1) // max_images_per_row
    spacing = 0 #40  # Adjust this value to control the spacing between images

    max_width = max_images_per_row * (max(img.width for _, img in images) + spacing)
    max_height = num_rows * (max(img.height for _, img in images))

    merged_image = Image.new('RGB', (max_width, max_height), color='white')

    x_offset = 0
    y_offset = 0

    for _, img in images:
        merged_image.paste(img, (x_offset, y_offset))

        x_offset += img.width + spacing  # Add spacing between images
        if x_offset >= max_width:
            x_offset = 0
            y_offset += img.height + spacing  # Add spacing between rows

    output_path = os.path.join(image_folder, "merged_image.jpg")
    merged_image.save(output_path)
    print("Merged image saved at:", output_path)

# Example usage
#folder_path = "test_images/compare_crops/merge-0859-x18/"
folder_path = "test_edge_maps/patches-eb3"
merge_images(folder_path)
