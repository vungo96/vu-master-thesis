from PIL import Image
import numpy as np

def compute_error_map(image1_path, image2_path, output_path):
    # Open the images using PIL
    image1 = Image.open(image1_path).convert("RGB")
    image2 = Image.open(image2_path).convert("RGB")

    # Convert the images to numpy arrays
    image1_arr = np.array(image1)
    image2_arr = np.array(image2)

    # Compute the absolute difference between the images
    error_map = np.abs(image1_arr.astype(np.float32) - image2_arr.astype(np.float32))

    # Calculate the maximum value across color channels
    max_error = np.max(error_map, axis=2)

    # Create a PIL image from the error map
    error_map_img = Image.fromarray(max_error.astype(np.uint8))

    # Save the error map
    error_map_img.save(output_path)

# Example usage
image1_path = "path/to/image1.png"
image2_path = "path/to/image2.png"
output_path = "path/to/error_map.png"

compute_error_map(image1_path, image2_path, output_path)
