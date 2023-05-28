import models

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from PIL import Image

import numpy as np
from matplotlib import pyplot as plt

""" u-net convnext mse """
path_to_unet_convnext_mse = 'unet_conv_models/trained_unet_model_convnext_mse.pth'

""" u-net convnext bce """
path_to_unet_convnext_bce = 'unet_conv_models/trained_unet_model_convnext_bce.pth'

""" u-net vanilla mse """
path_to_unet_vanilla_mse = 'unet_models/trained_unet_model_mse.pth'

""" u-net vanilla bce """
path_to_unet_vanilla_bce = 'unet_models/trained_unet_model_bce.pth'


class Dataset_rgb(Dataset):
    def __init__(self, ids):
        self.ids = ids

    def transform(self, train_data, train_labels):
        return TF.to_tensor(train_data), TF.to_tensor(train_labels)

    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, index):
        id = self.ids[index]

        Image.MAX_IMAGE_PIXELS = None
        X = Image.open(f"test_images/test_image_{id}.tif").convert('RGB')

        patch_size = 512
        batch_size = 8

        # split images into smaller images of size patch_size x patch_size
        X_patch = [TF.to_tensor(X.crop((i, j, i+patch_size, j+patch_size))) for i in range(0, X.width, patch_size) for j in range(0, X.height, patch_size)]
        
        # create list of batches
        X_batches = [torch.stack(X_patch[i:i+batch_size]) for i in range(0, len(X_patch), batch_size)]
        
        return X_batches
    
class Dataset_rgb(Dataset):
    def __init__(self, ids):
        self.ids = ids

    def transform(self, train_data, train_labels):
        return TF.to_tensor(train_data), TF.to_tensor(train_labels)

    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, index):
        id = self.ids[index]

        Image.MAX_IMAGE_PIXELS = None
        X = Image.open(f"test_images/test_image_{id}.tif")

        patch_size = 512
        batch_size = 8

        # split images into smaller images of size patch_size x patch_size
        X_patch = [TF.to_tensor(X.crop((i, j, i+patch_size, j+patch_size))) for i in range(0, X.width, patch_size) for j in range(0, X.height, patch_size)]
        
        # create list of batches
        X_batches = [torch.stack(X_patch[i:i+batch_size]) for i in range(0, len(X_patch), batch_size)]
        
        return X_batches


params = {"batch_size": 1, # batch size should be one to avoid re-batching of already batched data. Dataset class returns list of batched data
          "shuffle": True,
          "num_workers": 4,}


all_ids = range(1)

test_data_rgb = Dataset_rgb(all_ids)
test_generator_rgb = DataLoader(test_data_rgb, **params)

test_data = Dataset(all_ids)
test_generator = DataLoader(test_data, **params)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the trained models
""" unet convnext mse """
unet_convnext_bce_model = models.unet_convnext().to(device)
unet_convnext_bce_model.load_state_dict(torch.load(path_to_unet_convnext_bce, map_location=device))

""" unet convnext bce """
unet_convnext_mse_model = models.unet_convnext().to(device)
unet_convnext_mse_model.load_state_dict(torch.load(path_to_unet_convnext_mse, map_location=device))

""" unet vanilla mse """
unet_vanilla_mse_model = models.unet().to(device)
unet_vanilla_mse_model.load_state_dict(torch.load(path_to_unet_vanilla_mse, map_location=device))

""" unet vanilla bce """
unet_vanilla_bce_model = models.unet().to(device)
unet_vanilla_bce_model.load_state_dict(torch.load(path_to_unet_vanilla_bce, map_location=device))


def test_step(model, dataset):
    model.eval()
    test_images = []
    test_outputs = []
    with torch.no_grad():
        for batches in dataset:
            for batch in batches:
                # print(batch)
                test_image = batch[0].to(device)
                test_output = model(test_image)
                test_images.append(test_image.detach().cpu())
                test_outputs.append(test_output.detach().cpu())
    return test_images, test_outputs


test_images_convnext_mse, test_output_convnext_mse = test_step(unet_convnext_mse_model, test_generator_rgb)
test_images_convnext_bce, test_output_convnext_bce = test_step(unet_convnext_bce_model, test_generator_rgb)
test_images_vanilla_mse, test_output_vanilla_mse = test_step(unet_vanilla_mse_model, test_generator)
test_images_vanilla_bce, test_output_vanilla_bce = test_step(unet_vanilla_bce_model, test_generator)


def concat_batch(list_of_batches):
    concat_list = []
    for i in range(0, len(list_of_batches), 8): #8 is batch size
        concat_tensor = torch.cat(list_of_batches[i:i+8], dim=0)
        concat_list.append(concat_tensor)
    return concat_list

mse_convnext_concat_batch_test_images = concat_batch(test_images_convnext_mse)
mse_convnext_concat_batch_test_labels = concat_batch(test_images_convnext_mse)
mse_convnext_concat_batch_test_outputs = concat_batch(test_images_convnext_mse)

bce_convnext_concat_batch_test_images = concat_batch(test_images_convnext_bce)
bce_convnext_concat_batch_test_labels = concat_batch(test_images_convnext_bce)
bce_convnext_concat_batch_test_outputs = concat_batch(test_images_convnext_bce)

mse_vanilla_concat_batch_test_images = concat_batch(test_images_vanilla_mse)
mse_vanilla_concat_batch_test_labels = concat_batch(test_images_vanilla_mse)
mse_vanilla_concat_batch_test_outputs = concat_batch(test_images_vanilla_mse)

bce_vanilla_concat_batch_test_images = concat_batch(test_images_vanilla_bce)
bce_vanilla_concat_batch_test_labels = concat_batch(test_images_vanilla_bce)
bce_vanilla_concat_batch_test_outputs = concat_batch(test_images_vanilla_bce)

# todo reconstruct images.
def reconstruct(list_of_batches, orig_size):
    list_of_img = []
    for batch in list_of_batches:
        
        patch_size = batch.shape[-1]
        grid_size = orig_size // patch_size
        num_channels = batch.shape[1]

        X = torch.zeros(orig_size, orig_size, num_channels)
        for i in range(grid_size):
            for j in range(grid_size):
                X[j*patch_size:(j+1)*patch_size, i*patch_size:(i+1)*patch_size] = batch[i*grid_size+j].permute(1,2,0)
        
        list_of_img.append(X)
    return list_of_img



mse_convnext_re_test_images =  reconstruct(mse_convnext_concat_batch_test_images, 15360)
mse_convnext_re_test_labels =  reconstruct(mse_convnext_concat_batch_test_labels, 15360)
mse_convnext_re_test_outputs = reconstruct(mse_convnext_concat_batch_test_outputs, 15360)

bce_convnext_re_test_images =  reconstruct(bce_convnext_concat_batch_test_images, 15360)
bce_convnext_re_test_labels =  reconstruct(bce_convnext_concat_batch_test_labels, 15360)
bce_convnext_re_test_outputs = reconstruct(bce_convnext_concat_batch_test_outputs, 15360)

mse_vanilla_re_test_images =  reconstruct(mse_vanilla_concat_batch_test_images, 15360)
mse_vanilla_re_test_labels =  reconstruct(mse_vanilla_concat_batch_test_labels, 15360)
mse_vanilla_re_test_outputs = reconstruct(mse_vanilla_concat_batch_test_outputs, 15360)

bce_vanilla_re_test_images =  reconstruct(bce_vanilla_concat_batch_test_images, 15360)
bce_vanilla_re_test_labels =  reconstruct(bce_vanilla_concat_batch_test_labels, 15360)
bce_vanilla_re_test_outputs = reconstruct(bce_vanilla_concat_batch_test_outputs, 15360)


for i in range(len(mse_convnext_re_test_images)):
    plt.imshow(mse_convnext_re_test_outputs[i], cmap='gray')
    plt.savefig(f'test_output/mse_convnext_test_output_{i}.png', dpi=1000, bbox_inches='tight')
    plt.close()

for i in range(len(bce_convnext_re_test_images)):
    plt.imshow(bce_convnext_re_test_outputs[i], cmap='gray')
    plt.savefig(f'test_output/bce_convnext_test_output_{i}.png', dpi=1000, bbox_inches='tight')
    plt.close()

for i in range(len(mse_vanilla_re_test_images)):
    plt.imshow(mse_vanilla_re_test_outputs[i], cmap='gray')
    plt.savefig(f'test_output/mse_vanilla_test_output_{i}.png', dpi=1000, bbox_inches='tight')
    plt.close()

for i in range(len(bce_vanilla_re_test_images)):
    plt.imshow(bce_vanilla_re_test_outputs[i], cmap='gray')
    plt.savefig(f'test_output/bce_vanilla_test_output_{i}.png', dpi=1000, bbox_inches='tight')
    plt.close()