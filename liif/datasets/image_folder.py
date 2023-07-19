import os
import json
from PIL import Image

import pickle
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from datasets import register
from PIL import ImageFile
from utils import get_edge_map


@register('image-folder')
class ImageFolder(Dataset):

    def __init__(self, root_path, root_path2=None, split_file=None, split_key=None, first_k=None,
                 repeat=1, cache='none', sharded=None, edge_map=None):
        self.repeat = repeat
        self.cache = cache
        self.edge_map = edge_map

        ImageFile.LOAD_TRUNCATED_IMAGES = True

        if split_file is None:
            if sharded is None:
                filenames = sorted(os.listdir(root_path))
            else:
                # Get list of all subdirectories in root_path
                subdirs = [os.path.join(root_path, d) for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]

                # Get list of all image files in subdirectories
                filenames = []
                for subdir in subdirs:
                    filenames += [os.path.join(subdir, f) for f in os.listdir(subdir) if f.endswith('.jpg') or f.endswith('.png')]

                # Sort the list of file names alphabetically
                filenames = sorted(filenames)
        else:
            with open(split_file, 'r') as f:
                filenames = json.load(f)[split_key]

        # add another dataset
        if root_path2 is not None:
                print("Add additional datatset.")
                filenames2 = sorted(os.listdir(root_path2))
                filenames.extend(filenames2)
    
        if first_k is not None:
            filenames = filenames[:first_k]

        self.files = []
        self.edge_maps = []
        
        for i, filename in enumerate(filenames):
            if root_path2 is not None and i >= len(filenames)-len(filenames2):
                root_path = root_path2

            if sharded is None:
                file = os.path.join(root_path, filename)
            else:
                file = filename

            if cache == 'none':
                self.files.append(file)

            elif cache == 'bin':
                bin_root = os.path.join(os.path.dirname(root_path),
                    '_bin_' + os.path.basename(root_path))
                if not os.path.exists(bin_root):
                    os.mkdir(bin_root)
                    print('mkdir', bin_root)
                bin_file = os.path.join(
                    bin_root, filename.split('.')[0] + '.pkl')
                if not os.path.exists(bin_file):
                    with open(bin_file, 'wb') as f:
                        pickle.dump(imageio.imread(file), f)
                    print('dump', bin_file)
                self.files.append(bin_file)

            elif cache == 'in_memory':
                img = transforms.ToTensor()(
                    Image.open(file).convert('RGB'))
                self.files.append(img)
                if self.edge_map is not None:
                    self.edge_maps.append(get_edge_map(img, save_dir='test_images/'))


    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]

        if self.cache == 'none':
            return transforms.ToTensor()(Image.open(x).convert('RGB'))

        elif self.cache == 'bin':
            with open(x, 'rb') as f:
                x = pickle.load(f)
            x = np.ascontiguousarray(x.transpose(2, 0, 1))
            x = torch.from_numpy(x).float() / 255
            return x

        elif self.cache == 'in_memory':
            if self.edge_map is not None:
                edge_map = self.edge_maps[idx % len(self.files)]
                edge_map = torch.unsqueeze(torch.from_numpy(edge_map), dim=0)
                return torch.cat([x, edge_map], dim=0)
            return x


@register('paired-image-folders')
class PairedImageFolders(Dataset):

    def __init__(self, root_path_1, root_path_2, **kwargs):
        self.dataset_1 = ImageFolder(root_path_1, **kwargs)
        self.dataset_2 = ImageFolder(root_path_2, **kwargs)

    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, idx):
        return self.dataset_1[idx], self.dataset_2[idx]
