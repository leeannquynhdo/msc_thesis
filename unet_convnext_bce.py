
import torch
import torch.nn as nn
import torchvision


from torch.utils.data import DataLoader
from torch.utils.data import Dataset, random_split
import torchvision.transforms.functional as TF
import torch.optim as optim

import sys
import datetime

from PIL import Image

import numpy as np
from matplotlib import pyplot as plt

#############################################################################################################################

# train the model?

should_train = True

path_to_trained_model = 'unet_conv_models/trained_unet_model_convnext_bce.pth'
path_to_train_loss = 'unet_conv_models/unet_convnext_all_train_losses_bce.txt'
path_to_val_loss = 'unet_conv_models/unet_convnext_all_val_losses_bce.txt'


#############################################################################################################################


from torchvision.models import ConvNeXt_Tiny_Weights

model_tiny = torchvision.models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)


# U-net up-convolution
class convolution(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_c)
        
        self.relu = nn.ReLU()
        
    def forward(self, data):
        x = self.conv1(data)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        return x


# U-net decoder
class decoder(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = convolution(out_c + out_c, out_c)
        
    def forward(self, data, skip): # skip connections
        x = self.up(data)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x

# # Model definition: U-net using pretrained Convnet weights (tiny)

class unet_convnext(nn.Module):
    def __init__(self):
        super().__init__()

        """ Encoding """
        self.encoder = model_tiny.features # model: convnext tiny weights

        # downsampling
        self.stem = self.encoder[0]  # 192
        self.down1 = self.encoder[2] # 384
        self.down2 = self.encoder[4] # 768
        self.down3 = self.encoder[6] # 1536

        # convnext weights
        self.cn1 = self.encoder[1]
        self.cn2 = self.encoder[3]
        self.cn3 = self.encoder[5]
        self.cn4 = self.encoder[7]
        
        """ Decoding """

        """for convnext tiny"""
        self.decode1 = decoder(768, 384)
        self.decode2 = decoder(384, 192)
        self.decode3 = decoder(192, 96)

        self.last = nn.Sequential(
            nn.Conv2d(96, 1, kernel_size=1, padding=0),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
            )


    def forward(self, data):
        """ Encoding """
        s0 = self.stem(data) 
        cn1 = self.cn1(s0)
        s1 = self.down1(cn1) 
        cn2 = self.cn2(s1)
        s2 = self.down2(cn2)
        cn3 = self.cn3(s2)
        s3 = self.down3(cn3)

        """ Decoding """
        d1 = self.decode1(s3, s2)
        d2 = self.decode2(d1, s1)
        d3 = self.decode3(d2, s0)

        """ Classifier """
        output = self.last(d3)

        return torch.sigmoid(output)


class Dataset(Dataset):
    def __init__(self, ids):
        self.ids = ids

    def transform(self, train_data, train_labels):
        return TF.to_tensor(train_data), TF.to_tensor(train_labels)

    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, index):
        id = self.ids[index]

        X = Image.open(f"train_images/train_image_{id}.tif").convert('RGB')
        y = Image.open(f"train_labels/train_label_{id}.tif")


        patch_size = 512
        batch_size = 8

        # split images into smaller images of size patch_size x patch_size
        X_patch = [TF.to_tensor(X.crop((i, j, i+patch_size, j+patch_size))) for i in range(0, X.width, patch_size) for j in range(0, X.height, patch_size)]
        y_patch = [TF.to_tensor(y.crop((i, j, i+patch_size, j+patch_size))) for i in range(0, y.width, patch_size) for j in range(0, y.height, patch_size)]

        # create list of batches
        X_batches = [torch.stack(X_patch[i:i+batch_size]) for i in range(0, len(X_patch), batch_size)]
        y_batches = [torch.stack(y_patch[i:i+batch_size]) for i in range(0, len(y_patch), batch_size)]

        batch_ids = [(batch, id) for batch in zip(X_batches, y_batches)]
        
        return batch_ids


# ### Create train, validation and test set

params = {"batch_size": 1, # batch size should be one to avoid re-batching of already batched data. Dataset class returns batched data
          "shuffle": True,
          "num_workers": 4,}

all_ids = range(90)

# Define the split lengths
train_len = int(len(all_ids) * 0.6)
val_len = int(len(all_ids) * 0.2)
test_len = len(all_ids) - train_len - val_len

# Use random_split to split the dataset
train_data, val_data, test_data = random_split(
    Dataset(all_ids),
    [train_len, val_len, test_len]
)

training_generator= DataLoader(train_data, **params)
validation_generator = DataLoader(val_data, **params)
test_generator = DataLoader(test_data, **params)


# ## Training

print("GPU?:", torch.cuda.is_available())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
convnext_model = unet_convnext().to(device)
loss_func = nn.BCELoss().to(device)
optimizer = optim.Adam(convnext_model.parameters(), lr=0.0001)


# ### Functions to train model
def training_step(model, dataset):
    model.train()
    running_loss = 0

    for batches in dataset:
        for batch_pair, batch_id in batches:
            optimizer.zero_grad()
            train_images = batch_pair[0].to(device)
            train_labels = batch_pair[1].to(device)
            outputs = model(train_images.squeeze(0))
            loss = loss_func(outputs, train_labels.squeeze(0))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # train_loss_list.append(running_loss / len(dataset))
    return running_loss / len(dataset)

def validation_step(model, dataset):
    model.eval()
    validation_loss = 0
    with torch.no_grad():
        for batches in dataset:
            for batch_pair, batch_id in batches:
                val_images = batch_pair[0].to(device)
                val_labels = batch_pair[1].to(device)
                val_outputs = model(val_images.squeeze(0))
                validation_loss += loss_func(val_outputs, val_labels.squeeze(0)).item()
    return validation_loss / len(dataset)

def train_until_convergence(model, train_set, val_set, epochs, patience):
    best_val_loss = np.inf
    no_improvement = 0
    time_diff = 0
    train_loss_list = []
    val_loss_list = []

    total_start = datetime.datetime.now()

    for epoch in range(epochs):
        start_time = datetime.datetime.now()
        sys.stdout.write("\rCurrently at epoch: " + str(epoch+1) + ". Estimated time remaining: {}\n".format(time_diff*(epochs - epoch)))

        train_loss = training_step(model, train_set)
        val_loss = validation_step(model, val_set)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

        print(f"Epoch {epoch+1}: \t Training Loss: {train_loss}, \t Validation Loss: {val_loss}")

        end_time = datetime.datetime.now()
        time_diff = end_time - start_time

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement = 0
        else:
            no_improvement += 1

        if no_improvement >= patience:
            print(f"No improvement in validation loss for {patience} epochs. Stopping training...")
            break

    total_end = datetime.datetime.now()
    total_time = total_end-total_start
    print("Total running time for unet with convnext pretrained weights using BCELoss", total_time)
    
    return model, train_loss_list, val_loss_list


if should_train:

    # # Train the model LET's FuCKING SKRRRrrRT

    trained_model, train_losses, val_losses = train_until_convergence(convnext_model, training_generator, validation_generator, epochs=500, patience=500)
    # save the trained model
    torch.save(trained_model.state_dict(), path_to_trained_model)
    # print('train losses:', train_losses)


    # write loss to file
    with open(path_to_train_loss, 'w') as f:
        for loss in train_losses:
            f.write("%s\n" % loss)
    
    with open(path_to_val_loss, 'w') as f:
        for loss in val_losses:
            f.write("%s\n" % loss)



# load the trained model
trained_model = unet_convnext().to(device)
trained_model.load_state_dict(torch.load(path_to_trained_model)) # path to trained model

def test_step(model, dataset):
    model.eval()
    test_loss = 0
    test_images = []
    test_labels = []
    test_outputs = []
    test_loss_list = []
    with torch.no_grad():
        for batches in dataset:
            for batch_pair, batch_id in batches:
                test_image = batch_pair[0].to(device)
                test_label = batch_pair[1].to(device)
                test_output = model(test_image.squeeze(0))
                
                test_images.append(test_image.squeeze(0).detach().cpu())
                test_labels.append(test_label.squeeze(0).detach().cpu())
                test_outputs.append(test_output.detach().cpu())
                test_loss += loss_func(test_output, test_label.squeeze(0)).item()
                test_loss_list.append(test_loss / len(dataset))
    return test_images, test_labels, test_outputs, test_loss_list


test_images, test_labels, test_outputs, test_loss = test_step(trained_model, test_generator)

## with 64 batches of size [16, num_c, 256, 256], each image of made from 16 consecutive tensors in the outputs
## new batch size[8, num_c, 512, 512] 

# batch tensors so that one tensor contains a full image

def concat_batch(list_of_batches):
    concat_list = []
    for i in range(0, len(list_of_batches), 8):
        concat_tensor = torch.cat(list_of_batches[i:i+8], dim=0)
        concat_list.append(concat_tensor)
    return concat_list

concat_batch_test_images = concat_batch(test_images)
concat_batch_test_labels = concat_batch(test_labels)
concat_batch_test_outputs = concat_batch(test_outputs)

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

re_test_images = reconstruct(concat_batch_test_images, 4096)
re_test_labels = reconstruct(concat_batch_test_labels, 4096)
re_test_outputs = reconstruct(concat_batch_test_outputs, 4096)

for i in range(len(re_test_images)):
    fig, ax = plt.subplots(1,3,figsize=(15,5))
    ax[0].imshow(re_test_images[i], cmap='gray')
    ax[0].set_title('Image')
    ax[1].imshow(re_test_labels[i], cmap='gray')
    ax[1].set_title('Ground truth')
    ax[2].imshow(re_test_outputs[i], cmap='gray')
    ax[2].set_title('Prediction')
    plt.savefig(f'unet_conv_models/unet_conv_test_result_bce_{i}.png', dpi=500, bbox_inches='tight')
    plt.close()
    # plt.show()

## load train and val loss
train_losses = np.loadtxt(path_to_train_loss)
val_losses = np.loadtxt(path_to_val_loss)


loss_x = list(range(len(train_losses)))
plt.plot(loss_x, train_losses, label='Train loss')
plt.plot(loss_x, val_losses, label='Validation loss')
plt.title('Train / Validation loss')
plt.legend()
plt.savefig(f'unet_conv_models/unet_conv_train_val_loss_bce.png', dpi=500, bbox_inches='tight')
plt.close()

predictions = torch.stack(re_test_outputs)
labels = torch.stack(re_test_labels)

# assuming predictions and labels are PyTorch tensors with shape [4, 4096, 4096, 1]
predictions = predictions.squeeze().view(-1)
labels = labels.squeeze().view(-1)


predictions = (predictions > 0.5).float()
print(len(predictions))

accuracy = torch.eq(predictions, labels).sum().item() / len(predictions)
with open(f'unet_conv_models/unet_conv_accuracy_bce.txt', 'w') as f:
    f.write("%s\n" % accuracy)
