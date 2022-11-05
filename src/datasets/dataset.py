import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split

class ImageDataset():
    def __init__(self, config):
        #train and test data directory
        self.data_dir = "./data/train"
        test_data_dir = "./data/test"


#load the train and test data
dataset = ImageFolder(data_dir,transform = transforms.Compose([transforms.ToTensor()
]))
test_dataset = ImageFolder(test_data_dir,transforms.Compose([transforms.ToTensor()
]))

img, label = dataset[0]
print(img.shape,label)

print("Follwing classes are there : \n",dataset.classes)

def display_img(img,label):
    print(f"Label : {dataset.classes[label]}")
    plt.imshow(img.permute(1,2,0))
    plt.show()

display_img(*dataset[0])

batch_size = 128
val_size = 2000
train_size = len(dataset) - val_size 

train_data,val_data = random_split(dataset,[train_size,val_size])
print(f"Length of Train Data : {len(train_data)}")
print(f"Length of Validation Data : {len(val_data)}")

#output
#Length of Train Data : 12034
#Length of Validation Data : 2000

#load the train and validation into batches.
train_dl = DataLoader(train_data, batch_size, shuffle = True, num_workers = 4, pin_memory = True)
val_dl = DataLoader(val_data, batch_size*2, num_workers = 4, pin_memory = True)