import torch
from torch import nn

from torch.utils.data import DataLoader
from torch.utils.data import Dataset, random_split
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import torch.optim as optim

import sys
import datetime

from PIL import Image

import numpy as np
from matplotlib import pyplot as plt


should_train = False

path_to_trained_model = 'line_unet/line_models/line_cnn_model.pth'
path_to_train_loss = 'line_unet/line_models/line_cnn_train_losses.txt'
path_to_val_loss = 'line_unet/line_models/line_cnn_val_losses.txt'


class encoder(nn.Module):
    def __init__(self,nBins=10,KSize=(3,3),WSize=(3,3)):
        super().__init__()
        self.nBins = nBins
        self.b = nn.Parameter(torch.randn(nBins)).to(device)
        self.K = nn.Parameter(torch.randn(1,1,*KSize)).to(device) # learnable kernel init
        self.W = nn.Parameter(torch.randn(1,1,*WSize)).to(device) # kernel init
        self.act = nn.ReLU()
        # reordering to save time in forward()
        self.V = torch.cat([self.W for i in range(nBins)],dim=0)
        self.bias = self.b.view(1,nBins,1,1)

    def forward(self, I):
        IK = nn.functional.conv2d(I.to(device), self.K, None, stride=1, padding=1)
        X = nn.functional.conv2d(self.act(self.bias-IK), self.V, None, padding='same', groups=self.nBins)
        return X

class CNN(nn.Module):
    def __init__(self,nKernels,nBins,nOut):
        super().__init__()
        self.convs = nn.ModuleList([encoder(nBins) for i in range(nKernels)])
        self.lin = nn.Linear(nKernels*nBins,nOut)

    def forward(self, X):
        X = [h(X) for h in self.convs]
        X = torch.cat(X,1)
        X = torch.permute(self.lin(torch.permute(X,(0,2,3,1))),(0,3,1,2))
        return X


class Dataset(Dataset):
    def __init__(self, ids):
        self.ids = ids

    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, index):
        id = self.ids[index]

        X = TF.to_tensor(Image.open(f"line_unet/line_images/img_{id}.png"))
        y = TF.to_tensor(Image.open(f"line_unet/line_images/gms_{id}.png"))
        
        return X, y


params = {"batch_size": 1, 
          "shuffle": True,
          "num_workers": 4,}

all_ids = range(500)

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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_model = CNN(2,10,1)
cnn_model.to(device)
loss_func = nn.MSELoss().to(device)
optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)


def training_step(model, dataset):
    model.train()
    running_loss = 0

    for batch_pair in dataset:
        optimizer.zero_grad()
        train_images = batch_pair[0].to(device)
        train_labels = batch_pair[1].to(device)
        outputs = model(train_images)
        loss = loss_func(outputs, train_labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataset)

def validation_step(model, dataset):
    model.eval()
    validation_loss = 0
    with torch.no_grad():
        for batch_pair in dataset:
            val_images = batch_pair[0].to(device)
            val_labels = batch_pair[1].to(device)
            val_outputs = model(val_images)
            validation_loss += loss_func(val_outputs, val_labels).item()
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
    print("Total running time for lines cnn unet:", total_time)
    
    return model, train_loss_list, val_loss_list


if should_train:
    trained_model, train_losses, val_losses = train_until_convergence(cnn_model, training_generator, validation_generator, epochs=500, patience=500)
    # save the trained model
    torch.save(trained_model.state_dict(), path_to_trained_model)
    
    # write loss to file
    with open(path_to_train_loss, 'w') as f:
        for loss in train_losses:
            f.write("%s\n" % loss)
    
    with open(path_to_val_loss, 'w') as f:
        for loss in val_losses:
            f.write("%s\n" % loss)


# load the trained model
trained_model = cnn_model.to(device)
trained_model.load_state_dict(torch.load(path_to_trained_model, map_location=device))


def test_step(model, dataset):
    model.eval()
    test_loss = 0
    test_images = []
    test_labels = []
    test_outputs = []
    test_loss_list = []
    with torch.no_grad():
        for batch_pair in dataset:
            test_image = batch_pair[0].to(device)
            test_label = batch_pair[1].to(device)
            test_output = model(test_image)
            
            test_images.append(test_image.detach().cpu())
            test_labels.append(test_label.detach().cpu())
            test_outputs.append(test_output.detach().cpu())
            test_loss += loss_func(test_output, test_label).item()
            test_loss_list.append(test_loss / len(dataset))
    return test_images, test_labels, test_outputs, test_loss_list


test_images, test_labels, test_outputs, test_loss = test_step(trained_model, test_generator)

for i in range(len(test_images)):
    fig, ax = plt.subplots(1,3,figsize=(15,5))
    ax[0].imshow(test_images[i].squeeze(0).permute(1,2,0), cmap='gray')
    ax[0].set_title('image')
    ax[1].imshow(test_labels[i].squeeze(0).permute(1,2,0), cmap='gray')
    ax[1].set_title('ground truth')
    ax[2].imshow(test_outputs[i].squeeze(0).permute(1,2,0), cmap='gray')
    ax[2].set_title('prediction')
    plt.savefig(f'line_unet/line_models/test_output/cnn_test_result_{i}.png', dpi=500, bbox_inches='tight')
    plt.close()


train_losses = np.loadtxt(path_to_train_loss)
val_losses = np.loadtxt(path_to_val_loss)


loss_x = list(range(len(train_losses)))
plt.plot(loss_x, train_losses, label='Train loss')
plt.plot(loss_x, val_losses, label='Validation loss')
plt.title('Train / Validation loss')
plt.legend()
plt.show()
plt.savefig('line_unet/line_models/line_cnn_train_val_loss.png', dpi=500, bbox_inches='tight')
plt.close()

predictions = torch.stack(test_outputs)
labels = torch.stack(test_labels)

# from sklearn.metrics import roc_curve, roc_auc_score
# fpr, tpr, ts = roc_curve(labels.int().view(-1).numpy(), predictions.view(-1).numpy())
# auc = roc_auc_score(labels.int().view(-1).numpy(), predictions.view(-1).numpy())

# plt.title('CNN lines ROC curve')
# plt.plot(fpr, tpr, label=f'AUC score = {auc}')
# plt.legend()
# plt.savefig('line_unet/line_models/cnn_roc_curve.png', dpi=500, bbox_inches='tight')
# plt.close()

# predictions = (predictions > 0.5)
# print('predictions:', predictions)
# print('predictions min:', predictions.min())
# print('predictions max:', predictions.max)

# # predictions = (predictions > 0.5)

# print("CNN lines")

# accuracy = torch.eq(predictions, labels).sum().item() / len(predictions)

# print("accuracy:", accuracy)

# def calculate_iou(pred, target):
#     intersection = torch.logical_and(pred, target).sum()
#     union = torch.logical_or(pred, target).sum()
#     iou = intersection.float() / union.float()
#     return iou

# iou = calculate_iou(predictions, labels)
# print("iou:", iou)

# from sklearn.metrics import f1_score

# f1 = f1_score(labels.int(), predictions.int())
# print("f1:", f1)