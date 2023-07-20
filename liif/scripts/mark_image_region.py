from PIL import Image, ImageDraw
import os

def mark_patch_with_box(input_path, center_x, center_y, patch_size_x, patch_size_y, output_folder, tag, width):
    # Open the image
    image = Image.open(input_path)

    # Calculate the coordinates for cropping the patch
    left = center_x - patch_size_x // 2
    top = center_y - patch_size_y // 2
    right = left + patch_size_x
    bottom = top + patch_size_y

    # Draw a red rectangle around the patch
    draw = ImageDraw.Draw(image)
    draw.rectangle([left, top, right, bottom], outline="red", width=width)

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the modified image with the given tag
    output_path = os.path.join(output_folder, f"{tag}.png")
    image.save(output_path)

if __name__ == "__main__":
    # Example usage:
    input_path = "output.png"
    center_x = 128  # Replace with the desired center x-coordinate of the patch
    center_y = 128  # Replace with the desired center y-coordinate of the patch
    patch_size_x = 128  # Replace with the desired patch size in the x-direction
    patch_size_y = 128  # Replace with the desired patch size in the y-direction
    output_folder = "test_images"  # Replace with the desired output folder path
    tag = "0-test-crop-image"  # Replace with the desired tag for the cropped image
    width=4

    mark_patch_with_box(input_path, center_x, center_y, patch_size_x, patch_size_y, output_folder, tag, width)
