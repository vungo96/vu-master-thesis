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
    input_path = "../../../mngo_datasets/load/div2k/DIV2K_valid_HR/0806.png"
    #input_path = "../../../mngo_datasets/load/benchmark/Urban100/HR/img004.png"
    # input_path = "test_images/compare_crops/001-manga109-x6-AkkeraKanjinchou-GT-big-crop.png"
    center_x = 420 # 1040 + 64# 128+32 #1660  # 550 # 528  # 360 + 64 #175  
    center_y = 470 # 128 + 40 - 48 # 128 #815 # 1100 # 970 +128 # 64 # 860  
    patch_size_x = 128 #128 #256 # 512 # 512 # 128 # 256
    patch_size_y = 128 #128 #256 # 512 # 512 # 128 # 256  
    output_folder = "test_images"  
    tag = "0000-test-crop-image"  
    width=10

    mark_patch_with_box(input_path, center_x, center_y, patch_size_x, patch_size_y, output_folder, tag, width)
