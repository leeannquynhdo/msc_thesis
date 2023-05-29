
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

from torchvision.models import ConvNeXt_Tiny_Weights

model_tiny = torchvision.models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)

#############################################################################################################################

# train the model?

should_train = False

path_to_trained_model = 'lon_unet_models_normal_batch/trained_lon_unet_model.pth'
path_to_train_loss = 'lon_unet_models_normal_batch/lon_unet_train_losses.txt'
path_to_val_loss = 'lon_unet_models_normal_batch/lon_unet_val_losses.txt'


#############################################################################################################################

class HistActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def sigmoid(self, x):
        return 1 / (1 + torch.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    def forward(self, x):
        return self.sigmoid_derivative(x)

class Hist(nn.Module):
    def __init__(self,nBins=10,KSize=(3,3),WSize=(3,3)):
        super().__init__()
        self.nBins = nBins
        self.b = nn.Parameter(torch.randn(nBins)).to(device)
        self.K = nn.Parameter(torch.randn(1,1,*KSize)).to(device) # kernel init
        self.W = nn.Parameter(torch.randn(1,1,*WSize)).to(device) # kernel init
        self.act = HistActivation()
        # reordering to save time in forward()
        self.V = torch.cat([self.W for i in range(nBins)],dim=0)
        self.bias = self.b.view(1,nBins,1,1)

    def forward(self, I):
        IK = nn.functional.conv2d(I.to(device), self.K, None, stride=1, padding=1)
        X = nn.functional.conv2d(self.act(self.bias - IK), self.V, None, padding='same', groups=self.nBins)
        return X
    
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
    
class encoder(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = convolution(in_c, out_c)
        
    def forward(self, data):
        x = self.conv(data)
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


class LON(nn.Module):
    def __init__(self, nKernels, nBins, nOut):
        super().__init__()
        self.convs = nn.ModuleList([Hist(nBins) for i in range(nKernels)])
        self.lin = nn.Linear(nKernels*nBins,nOut)

    def forward(self, X):
        X = [h(X) for h in self.convs]
        X = torch.cat(X,1)
        X = torch.permute(self.lin(torch.permute(X,(0,2,3,1))),(0,3,1,2))
        return X
    
class LON_UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d((2,2))
        
        """ Encoding """
        self.en1 = LON(2,10,64)
        self.en2 = encoder(64, 128)
        self.en3 = encoder(128, 256)
        self.en4 = encoder(256, 512)

        
        # """ Bottleneck """
        self.bottle = convolution(512, 1024)
        
        # """ Decoding """
        self.de1 = decoder(1024, 512)
        self.de2 = decoder(512, 256)
        self.de3 = decoder(256, 128)
        self.de4 = decoder(128, 64)
        
        """ Classifier """
        self.last = nn.Conv2d(64, 1, kernel_size=1, padding=0)
        
    
    def forward(self, data):
        """ Encoding """
        s1 = self.en1(data)
        p1 = self.pool(s1)
        s2 = self.en2(p1)
        p2 = self.pool(s2)
        s3 = self.en3(p2)
        p3 = self.pool(s3)
        s4 = self.en4(p3)
        p4 = self.pool(s4)
        
        """ Bottleneck """
        b = self.bottle(p4)
        
        """ Decoding """
        d1 = self.de1(b, s4)
        d2 = self.de2(d1, s3)
        d3 = self.de3(d2, s2)
        d4 = self.de4(d3, s1)
        
        """ Classifier """
        outs = self.last(d4)
        
        return torch.sigmoid(outs)



class Dataset(Dataset):
    def __init__(self, ids):
        self.ids = ids

    def transform(self, train_data, train_labels):
        return TF.to_tensor(train_data), TF.to_tensor(train_labels)

    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, index):
        id = self.ids[index]

        X = Image.open(f"train_images_lon/train_image_{id}.tif")
        y = Image.open(f"train_labels_lon/train_label_{id}.tif")

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
unet_lon_model = LON_UNet().to(device)
loss_func = nn.MSELoss().to(device)
optimizer = optim.Adam(unet_lon_model.parameters(), lr=0.001)


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

        if (epoch%50==0): #save every model every 50th iter
            torch.save(model.state_dict(), f'lon_unet_models_normal_batch/lon_unet_model_epoch_{epoch+1}.pth' )

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
    print("Total running time for unet_lon", total_time)
    
    return model, train_loss_list, val_loss_list


if should_train:

    # # Train the model LET's FuCKING SKRRRrrRT

    trained_model, train_losses, val_losses = train_until_convergence(unet_lon_model, training_generator, validation_generator, epochs=500, patience=500)
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
trained_model = LON_UNet().to(device)
trained_model.load_state_dict(torch.load(path_to_trained_model, map_location=device)) # path to trained model

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
# print('test_images[0].shape:',test_images[0].shape)
# test_images[0].shape: torch.Size([4, 1, 256, 256])

# batch tensors so that one tensor contains a full image

# thus grid is 16x16

def concat_batch(list_of_batches):
    concat_list = []
    for i in range(0, len(list_of_batches), 8):
        concat_tensor = torch.cat(list_of_batches[i:i+8], dim=0)
        concat_list.append(concat_tensor)
    return concat_list

concat_batch_test_images = concat_batch(test_images)
concat_batch_test_labels = concat_batch(test_labels)
concat_batch_test_outputs = concat_batch(test_outputs)

print('concat_batch_test_images[0].shape:', concat_batch_test_images[0].shape)

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
    ax[0].set_title('image')
    ax[1].imshow(re_test_labels[i], cmap='gray')
    ax[1].set_title('ground truth')
    ax[2].imshow(re_test_outputs[i], cmap='gray')
    ax[2].set_title('prediction')
    plt.savefig(f'lon_unet_models_normal_batch/lon_unet_test_result{i}.png', dpi=500, bbox_inches='tight')
    plt.close()
    plt.show()

## load train and val loss
train_losses = np.loadtxt(path_to_train_loss)
val_losses = np.loadtxt(path_to_val_loss)


loss_x = list(range(len(train_losses)))
plt.plot(loss_x, train_losses, label='Train loss')
plt.plot(loss_x, val_losses, label='Validation loss')
plt.title('Train / Validation loss')
plt.legend()
plt.savefig(f'lon_unet_models_normal_batch/lon_unet_train_val_loss.png', dpi=500, bbox_inches='tight')
plt.close()

predictions = torch.stack(re_test_outputs)
labels = torch.stack(re_test_labels)

# assuming predictions and labels are PyTorch tensors with shape [4, 4096, 4096, 1]
predictions = predictions.squeeze().view(-1)
labels = labels.squeeze().view(-1)


from sklearn.metrics import roc_curve, roc_auc_score
fpr, tpr, ts = roc_curve(labels.int().view(-1).numpy(), predictions.view(-1).numpy())
auc = roc_auc_score(labels.int().view(-1).numpy(), predictions.view(-1).numpy())

plt.title('U-Net BCE ROC curve')
plt.plot(fpr, tpr, label=f'AUC score = {auc}')
plt.legend()
plt.savefig('unet_models/roc_curve_bce.png', dpi=500, bbox_inches='tight')
plt.close()

predictions = (predictions > 0.5).float()

print("U-Net LON 90 images. normal batch")

accuracy = torch.eq(predictions, labels).sum().item() / len(predictions)
# with open(f'unet_conv_models/unet_conv_accuracy_bce.txt', 'w') as f:
#     f.write("%s\n" % accuracy)
print("accuracy:", accuracy)

def calculate_iou(pred, target):
    intersection = torch.logical_and(pred, target).sum()
    union = torch.logical_or(pred, target).sum()
    iou = intersection.float() / union.float()
    return iou

iou = calculate_iou(predictions, labels)
print("iou:", iou)

from sklearn.metrics import f1_score

f1 = f1_score(labels.int(), predictions.int())
print("f1:", f1)