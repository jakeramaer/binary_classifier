import torch
import torchvision
import torchsummary
import matplotlib.pyplot as plt
import json
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from torchsummary import summary

from torchvision import models
import torch.nn.functional as F
from torch.nn import Linear

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load config
f = open('./config.json')
config = json.load(f)

# Hyperparameters
data_dir = config['data']['data_dir']
val_size = config['data']['validation_size']
num_workers = config['data']['num_workers']
batch_size = config['training']['batch_size']
num_epochs = config['training']['num_epochs']
lr = config['training']['lr']

# Create train + val dataloaders
train_dataset = ImageFolder(data_dir + '/train', transform = transforms.Compose([transforms.ToTensor()]))
train_size = len(train_dataset) - val_size 
train_data,val_data = random_split(train_dataset,[train_size, val_size], generator=torch.Generator().manual_seed(42)) # Manual seed ensures replicable results
train_dl = DataLoader(train_data, batch_size, shuffle = True, num_workers = num_workers, pin_memory = True)
val_dl = DataLoader(val_data, batch_size, shuffle = True, num_workers = num_workers, pin_memory = True)

# Set up pretrained VGG16 model, freeze weights and add trainable binary output layer
opt_func = torch.optim.Adam
model = models.vgg16_bn(pretrained=True)
for param in model.parameters():
   param.requires_grad = False
model.classifier[6] = Linear(4096, 2)
for param in model.classifier[6].parameters():
    param.requires_grad = True 
model.to(device)
if config['model']['print_summary']:
    summary(model, torch.Size([3, 128, 128]))

def print_epoch_results(epoch, result):
    print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
        epoch, result['train_loss'], result['val_loss'], result['val_acc']))

def print_training_graphs(history):
    figure, axis = plt.subplots(2)
    # Plot accuracies
    accuracies = [x['val_acc'] for x in history]
    axis[0].plot(accuracies, '-x')
    axis[0].set_xlabel('epoch')
    axis[0].set_ylabel('accuracy')
    axis[0].set_title('Accuracy vs. No. of epochs');

    # Plot losses
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    axis[1].plot(train_losses, '-bx')
    axis[1].plot(val_losses, '-rx')
    axis[1].set_xlabel('epoch')
    axis[1].set_ylabel('loss')
    axis[1].legend(['Training', 'Validation'])
    axis[1].set_title('Loss vs. No. of epochs');

    return plt

def accuracy(outputs, labels):
    # Assess binary accuracy between predictions and ground truth
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def fit(epochs, lr, model, train_loader, val_loader, opt_func):
    # Fit model to the training data
    history = []
    optimizer = opt_func(model.parameters(),lr)
    for epoch in range(epochs):

        # Train step
        model.train()
        train_losses = []
        for batch in train_loader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            out = model(images)                  # Generate predictions
            loss = F.cross_entropy(out, labels) # Calculate loss
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        # Val step
        model.eval()
        outputs = []
        with torch.no_grad():
            for batch in val_loader:
                images, labels = batch
                images, labels = images.to(device), labels.to(device)
                out = model(images)                    # Generate predictions
                loss = F.cross_entropy(out, labels)   # Calculate loss
                acc = accuracy(out, labels)           # Calculate accuracy
                outputs.append({'val_loss': loss.detach(), 'val_acc': acc})

        # Metrics
        epoch_loss = torch.stack([x['val_loss'] for x in outputs]).mean() # Combine losses
        epoch_acc = torch.stack([x['val_acc'] for x in outputs]).mean() # Combine accuracies
        result = {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
        result['train_loss'] = torch.stack(train_losses).mean().item()
        print_epoch_results(epoch, result)
        history.append(result)
    
    return history

#fitting the model on training data and record the result after each epoch
history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)
if config['training']['print_training_graphs']:
    print_training_graphs(history)
if config['training']['save_model']:
    torch.save(model, './models/VGG16-model.pth')