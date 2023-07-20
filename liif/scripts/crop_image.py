from PIL import Image
import os

def crop_image(input_path, center_x, center_y, patch_size_x, patch_size_y, output_folder, tag):
    # Open the image
    image = Image.open(input_path)

    # Calculate the coordinates for cropping the patch
    left = center_x - patch_size_x // 2
    top = center_y - patch_size_y // 2
    right = left + patch_size_x
    bottom = top + patch_size_y

    # Crop the image
    cropped_image = image.crop((left, top, right, bottom))

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the cropped image with the given tag
    output_path = os.path.join(output_folder, f"{tag}.png")
    cropped_image.save(output_path)

if __name__ == "__main__":
    # Example usage:
    input_path = "output.png"
    center_x = 128  # Replace with the desired center x-coordinate of the patch
    center_y = 128  # Replace with the desired center y-coordinate of the patch
    patch_size_x = 256  # Replace with the desired patch size in the x-direction
    patch_size_y = 256  # Replace with the desired patch size in the y-direction
    output_folder = "test_images"  # Replace with the desired output folder path
    tag = "0-test-crop-image"  # Replace with the desired tag for the cropped image

    crop_image(input_path, center_x, center_y, patch_size_x, patch_size_y, output_folder, tag)
