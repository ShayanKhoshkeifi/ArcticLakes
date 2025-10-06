import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
import config
import numpy as np

class ArcticLabeledImagePatchDataset(Dataset):
    def __init__(self, image_dir, label_dir, grid_file, patch_size, stride, transform=None, normalize=True):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.grid_file = grid_file

        with open(grid_file, 'r') as file:
            self.grids = [line.strip() for line in file]
            print(f"Dataset created for Grids: {self.grids}")

        self.image_paths = [f"{os.path.join(self.image_dir, config.tile + '_' + config.date + '_' + grid + '_' + config.image_file_suffix)}" for grid in self.grids]
        self.label_paths = [f"{os.path.join(self.label_dir, config.tile + '_' + config.date + '_' + grid + '_' + config.label_file_suffix)}" for grid in self.grids]
        assert len(self.image_paths) == len(self.label_paths), "Number of images and label masks do not match."
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform
        self.normalize = normalize
        self.patch_info = self.__generate_patch_info() # Stores (image_idx, start_x, start_y) for each patch

    def __generate_patch_info(self):

        patch_info = []
        for img_idx, img_path in enumerate(self.image_paths):
            img_array = np.load(img_path)
            num_channels, img_height, img_width = img_array.shape

            for y in range(0, img_height - self.patch_size[0] + 1, self.stride[0]):
                for x in range(0, img_width - self.patch_size[1] + 1, self.stride[1]):
                    patch_info.append((img_idx, x, y))
        return patch_info

    def __len__(self):
        return len(self.patch_info)

    def __getitem__(self, idx):
        img_idx, start_x, start_y = self.patch_info[idx]
        img_path = self.image_paths[img_idx]
        label_path = self.label_paths[img_idx]

        img_array = np.load(img_path)

        # Ensure only 10 channels
        img_array = img_array[:10]  # (10, H, W)
        image_patch = img_array[:, start_x:start_x+self.patch_size[1], start_y:start_y+self.patch_size[0]]

        if self.normalize:
            # Normalize data
            means = np.load(config.means)
            stds = np.load(config.stds)
            for channel in range(image_patch.shape[0]):
                image_patch[channel] = (image_patch[channel] - means[channel])/stds[channel]

        image_patch = torch.from_numpy(image_patch)

        if self.transform:
            image_patch = self.transform(image_patch)              
                        
        label_patch = np.load(label_path)
        label_patch = label_patch[start_x:start_x+self.patch_size[1], start_y:start_y+self.patch_size[0]]
        
        # Binarize labels using threshold
        label_patch = (label_patch >= config.water_occurrence_threshold).astype(np.float32)
        
        label_patch = torch.from_numpy(label_patch)


        if self.transform:
            label_patch = self.transform(label_patch)

        return image_patch, label_patch


if __name__ == "__main__":
    print("Testing dataloader")
    train_dataset = ArcticLabeledImagePatchDataset(config.IMAGE_GRIDS_PATH, config.LABEL_GRIDS_PATH, config.train_grids, config.patch_size, config.stride)
    train_loader = DataLoader(train_dataset, config.batch_size, shuffle=True)

    for idx, (images, labels) in enumerate(train_loader):
        if idx == 0:
            print("Unique values in label[0]:", torch.unique(labels[0]))
        if (idx % 100 == 0):
            print(images.shape)
            print(labels.shape)
            print(images[0])
            print(labels[0])
            

        