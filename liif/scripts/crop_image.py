from PIL import Image
import os

def crop_image(input_path, center_x, center_y, patch_size_x, patch_size_y, output_folder, tag):
    # Open the image
    image = Image.open(input_path)

    # shave = 24
    # width, height = image.size
    # print(image.size)
    # cropped_image = image.crop((shave, shave, width - shave, height - shave))
    # # shaved_image = image[..., shave:-shave, shave:-shave]
    # print(cropped_image.size)

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
    #input_path = "../../../mngo_datasets/load/div2k/DIV2K_valid_HR/0863.png"
    #input_path = "../../../mngo_datasets/load/benchmark/Manga109/HR/AkkeraKanjinchou.png"
    input_path = "test_images/compare_crops/0863-x18-GT-marked.png"
    tag = "0863-x18-GT-big-crop"  
    center_x = 550 # 400 +128 crocodile # 360 + 64 manga109 # 175  div2k-0821
    center_y = 1100 # 970 +128 #64 # 860 
    patch_size_x = 512 # 512 #128 # 256 
    patch_size_y = 512 # 512 #128 #256
    center_x = 700
    center_y = 1100
    patch_size_x = 1400  
    patch_size_y = 1400  
    output_folder = "test_images/compare_crops"  

    crop_image(input_path, center_x, center_y, patch_size_x, patch_size_y, output_folder, tag)
