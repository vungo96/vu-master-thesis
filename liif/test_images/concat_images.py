import os
from PIL import Image
from tqdm import tqdm

save_folder = 'concatenated_images'
image_folder1 = './_test-celebAHQ-16-1024_liif'
image_folder2 = './_test-celebAHQ-16-1024_liif_glean'
size = 1024
# the number of epochs passed until we save images -> set in test.py
num_epochs = 1

filenames1 = sorted(os.listdir(image_folder1))
filesnames2 = sorted(os.listdir(image_folder2))

#if len(filenames1) != len(filesnames2):
#    raise ValueError('Number of images in folders do not match')

images1 = [Image.open(os.path.join(image_folder1, filename)).resize((size, size)) for filename in filenames1]
images2 = [Image.open(os.path.join(image_folder2, filename)).resize((size, size)) for filename in filesnames2]

total_width = size * 4
max_height = size

new_im = Image.new('RGB', (total_width, max_height))

x_offset = 0
counter = 0

# stack together input -> gt -> pred1 -> pred2
for i in range(len(filenames1)):
    im = images1[i]
    new_im.paste(im, (x_offset,0))
    x_offset += im.size[0]

    if i % 3 == 2:
        im = images2[i]
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]
        
        new_im.save(os.path.join('.', save_folder, str(counter*num_epochs) + '_stacked.png'))
        new_im.close()
        new_im = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        counter += 1
