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
    #input_path = "../../../mngo_datasets/load/div2k/DIV2K_valid_HR/epoch-img-0_pred.png"
    #input_path = "../../../mngo_datasets/load/benchmark/Urban100/HR/img004.png"
    scales = [2, 3, 4, 6, 8, 12, 18, 24, 30]
    tags = ["lsdir-1to4-edge-crop", "lsdir-1to4-traditional", "lsdir-1toMax-edge-crop", "lsdir-1toMax",
            "div2k-1toMax-edge-crop", "div2k-1toMax", "div2k-1to4-edge-crop", "div2k-1to4-traditional"]

    for scale in scales:
        for tag in tags:
            input_path = "test_images-x/test_images-x"+str(scale)+"/_test-chosen-" + \
                str(scale)+"_" + tag + "/epoch-img-28_pred.png"
            input_path = "test-images-ciaosr/chosen-x" + \
                str(scale) + "/butterfly.png"
            tag = "ciaosr"
            tag = "butterfly-x" + str(scale) + "-" + tag
            folder = "**merge-butterfly"
            # center_x = # 420 palms # 750 urban100 # 1040 + 64 x18 0821 #400 monarch #128+32 set5 #1660 birds # 550 jaguar # 400 +128 crocodile # 360 + 64 manga109 # 175  div2k-0821 x24
            # center_y = # 470  # 500 # 128 + 40 - 48  #256 + 30 #128 #815 # 1100 # 970 +128 -5 #64 # 860
            # patch_size_x = # 128  # 200 # 128 + 20 #128 #128 #256 # 512 # 512 #128 # 256
            # patch_size_y = # 128  # 200 # 128 + 20 #128 #128 #256 # 512 # 512 #128 #256
            center_x = 128
            center_y = 128
            patch_size_x = 245
            # 128  # 200 # 128 + 20 #128 #128 #256 # 512 # 512 #128 #256
            patch_size_y = patch_size_x
            #center_x = 600
            #center_y = 600
            #patch_size_x = 1000
            #patch_size_y = 1000
            output_folder = "test_images/compare_crops/" + folder

            crop_image(input_path, center_x, center_y, patch_size_x,
                       patch_size_y, output_folder, str(scale) + "-" + tag)
