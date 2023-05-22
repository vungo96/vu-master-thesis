import os
from PIL import Image
from tqdm import tqdm

save_folder = 'concatenated_images'
image_folder1 = './_test-div2k-2__train_edsr-baseline-lte-variable-input_sample-2304-scale-1to4-batch-16-last'
image_folder2 = './_test-div2k-2__train_edsr-baseline-lte-variable-input-lsdir2_sample-4096-scale-1toMax-batch-32-inputs-48-lsdir-last'

# the number of epochs passed until we save images -> set in test.py
num_epochs = 1

filenames1 = sorted(os.listdir(image_folder1))
filesnames2 = sorted(os.listdir(image_folder2))

#if len(filenames1) != len(filesnames2):
#    raise ValueError('Number of images in folders do not match')

images1 = [Image.open(os.path.join(image_folder1, filename)) for filename in filenames1]
images2 = [Image.open(os.path.join(image_folder2, filename)) for filename in filesnames2]

im = images1[0]
total_width = 4 * im.size[0]
total_height = im.size[1]

new_im = Image.new('RGB', (total_width, total_height))

x_offset = 0
counter = 0

# stack together input -> gt -> pred1 -> pred2
for i in range(len(filenames1)):
    im = images1[i]
    if i % 3 == 1:
        im = im.resize((total_width // 4, total_height), resample=Image.BICUBIC)
    new_im.paste(im, (x_offset,0))
    x_offset += im.size[0]

    if i % 3 == 2:
        im = images2[i]
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]
        
        new_im.save(os.path.join('.', save_folder, str(counter*num_epochs) + '_stacked.png'))
        new_im.close()
        if i + 1 < len(filenames1):
            im = images1[i+1]
            total_width = 4 * im.size[0]
            new_im = Image.new('RGB', (total_width, total_height))
            x_offset = 0
            counter += 1
