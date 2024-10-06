!pip install kaggle
from google.colab import files
files.upload()
# Upload your API key here
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d shravankumar9892/image-colorization
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
import torch
import os
import random
import cv2
import tarfile
import numpy as np
import torchvision.transforms as tt
import matplotlib.pyplot as plt
import torch.nn as nn
from tqdm.notebook import tqdm
from torch import cat
from torch.autograd import Variable
#Uncomment it to unzip the file
#!unzip image-colorization.zip

dataset = [np.load('./ab/ab/ab1.npy'),np.load('./I/gray_scale.npy')] # training on ab1 , can use ab2 , ab3 alos
images = []
for image,image_gray in dataset:
    image = image.astype(np.float64)
    image[:, :, 0] *= 255 / 100
    image[:, :, 1] += 128
    image /= 255
    image_gray = image_gray.astype(np.float64)
    image_gray[:, :, 0] *= 255 / 100
    image_gray[:, :, 1] += 128
    image_gray /= 255
    torch_lab_image = torch.from_numpy(np.array(np.transpose(image, (2, 0, 1)),np.transpose(image_gray, (2, 0, 1)))).float()
    images.append(torch_lab_image)

# Making a custom dataloader
class Dataset(torch.utils.data.Dataset):
    def __len__(self):
        return len(images)

    def __getitem__(self, index):
        img,grey_image = images[index]
        return img,grey_image
    
ds = Dataset()
dl = torch.utils.data.DataLoader(lab_ds, batch_size=32,
                  shuffle=True, num_workers=2)

# Helper function
def show_img(img,msg):
    print(msg)
    figure,ax=plt.subplot(figsize=(12,12))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(img.permute(1,2,0))
# Method to get gpu if available
def get_default_device():
    if torch.cuda.is_available():
        print("Working with GPU....")
        return torch.device('cuda')
    else:
        print("Working with CPU....")
        return torch.device('cpu')
# Method to move data(tensor or model) to device
def to_device(data,device):
    # Checking if it is a list of tensors so that all of them can be transferred to device
    if isinstance(data, list):
        return [to_device(x,device) for x in data]

    # Else returning data after moving it to device using to() function
    return data.to(device, non_blocking=True)
# CLass to transfer dataloaders from cpu to device
class DeviceDataLoader():

    # Constructor
    def __init__(self,dl,device):
        self.dl=dl
        self.device=device

    # Function which yields batch in device upon beng used in loop
    def __iter__(self):
        for b in self.dl:
            yield to_device(b,self.device)

    # Returns no. of batches
    def __len__(self):
        return len(self.dl)
# Getting default device for execution
device = get_default_device()
dl=DeviceDataLoader(dl,device)
# Define a utility function to be used at the time of layer creation, returns 2 convolutional layers along with activation and batch size
def conv_layer(in_size,out_size,is_leaky=False):
    if is_leaky: # Use LeakyReLU()
        return nn.Sequential(
            nn.Conv2d(in_size,out_size,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_size),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_size,out_size,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_size),
            nn.LeakyReLU(0.2)
        )
    else: # Use ReLU()
        return nn.Sequential(
            nn.Conv2d(in_size,out_size,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_size),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_size,out_size,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_size),
            nn.LeakyReLU(0.2)
        )
# Defining a function to return a sequence of frequently occuring sequence in architechture
def deconv(in_size,out_size):
    return nn.Sequential(
        nn.ConvTranspose2d(in_size,out_size,3, 2, 1, 1),
        nn.ReLU()
    )
# Defining architechture of generator(Baseline architechture in research paper)
class Base(nn.Module):
    def __init__(self,is_leaky):
        super(Base, self).__init__()
        # input 1,256,256
        self.conv1=conv_layer(1,64,is_leaky)
        self.conv2=conv_layer(64,128,is_leaky)
        self.conv3=conv_layer(128,256,is_leaky)
        self.conv4=conv_layer(256,512,is_leaky)
        self.conv5=conv_layer(512,512,is_leaky)
        self.deconv1=deconv(512,512)
        self.deconv1=deconv(512,256)
        self.deconv2=deconv(256,128)
        self.deconv3=deconv(128,64)
        self.conv6=conv_layer(512,512,False)
        self.conv7=conv_layer(512,256,False)
        self.conv8=conv_layer(256,128,False)
        self.conv9=conv_layer(128,64,False)
        self.conv10=nn.Conv2d(64,2,1) # Kernel of size 1,1
        self.pool=nn.MaxPool2d(2)

    def forward(self,xb):
        x1=self.conv1(xb)
        x2=self.conv2(self.pool(x1))
        x3=self.conv3(self.pool(x2))
        x4=self.conv4(self.pool(x3))
        x5=self.conv5(self.pool(x4))
        x = self.conv6(torch.cat((x4, self.up1(x5)), 1))
        x = self.conv7(torch.cat((x4, self.up1(x)), 1))
        x = self.conv8(torch.cat((x4, self.up1(x)), 1))
        x = self.conv9(torch.cat((x4, self.up1(x)), 1))
        x = self.conv10(x)
        x=nn.Tanh(x)
        return x
load=False # Variable to check if training is required
Generator=Base(True)
try:
    Generator.load_state_dict(torch.load('colorize-image-gen.pth'))
    load=True
    print("Pre-trained model found!")
except Exception:
    print("No pre-trained model found , training from scratch")
    load=False
to_device(Generator,device)
# Defining architechture of discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1=conv_layer(3,64,True)
        self.conv2=conv_layer(64,128,True)
        self.conv3=conv_layer(128,256,True)
        self.conv4=conv_layer(256,512,True)
        self.conv5=conv_layer(512,512,True)
        self.clasfr1=nn.Linear(512*2*2,64)
        self.classfr2=nn.Linear(64,1)
        self.pool=nn.MaxPool2d(2)

    def forward(self,xb):
        x1=self.conv1(xb)
        x2=self.conv2(self.pool(x1))
        x3=self.conv3(self.pool(x2))
        x4=self.conv4(self.pool(x3))
        x5=self.conv5(self.pool(x4))
        x5=nn.Flatten(x5)
        x6=self.conv6(self.pool(x5))
        x6=nn.Sigmoid(x6)
        return x6
Discriminator=Discriminator()
try:
    Discriminator.load_state_dict(torch.load('colorize-image-dis.pth'))
    load=True
    print("Pre-trained model found!")
except Exception:
    print("No pre-trained model found , training from scratch")
    load=False
to_device(Discriminator,device)
d_opt = torch.optim.Adam(Discriminator.parameters(), betas=(0.5, 0.999), lr=0.0002)
g_opt = torch.optim.Adam(Generator.parameters(), betas=(0.5, 0.999), lr=0.0002) # Hyperparameters from research paper
epochs=50
if(not load):
    def fit(epochs):
        torch.cuda.empty_cache()
        Discriminator.train()
        Generator.train()
        smooth=0.1
        lost_g=0
        lost_d=0
        g_lambda=100 # factor to be multiplied with diff b/w real and fake images for generator loss
        for epoch in range(epochs):
         for index, batch,batch_grey in tqdm(enumerate(lab_dl)):
            l_images=batch_grey # Extracting first channel about luminescence
            c_images=batch # Extracting channels containing color info
            mean = torch.Tensor([0.5]).to(device)
            # Transforming into range of -0.5 to 0.5
            l_images=l_images-mean.expand_as(l_images)
            l_images=2*l_images
            c_images=c_images-mean.expand_as(c_images)
            c_images=2*c_images
            batch_size=batch.shape[0] # Fetching batch size from batch of loader
            to_device([l_images,c_images],device)
            l_images=Variable(l_images)
            c_images=Variable(c_images)
            print(l_images.shape)
            fake_images=Generator(l_images)
            # Training discriminator
            d_opt.zero_grad()
            pred=Discriminator(cat([l_images,c_images] ,1))
            real_loss_d=nn.BCELoss(pred,to_device(((1 - smooth) * torch.ones(batch_size)),device))
            pred=Discriminator(cat([l_images,fake_images], 1))
            fake_loss_d=nn.BCELoss(pred,to_device(( torch.zeros(batch_size)).cuda(),device))
            loss_d=real_loss_d+fake_loss_d
            loss_d.backward()
            d_opt.step()
            # Training generator
            g_opt.zero_grad()
            out=Discriminator(cat([l_images,c_images]), 1)
            fake_loss_g=nn.BCELoss(out,to_device((torch.ones(batch_size)).cuda(),device)) # Generated images must yield one in discriminator
            image_loss_g=g_lambda*nn.L1Loss(fake_images,c_images)
            loss_g=fake_loss_g+image_loss_g
            loss_g.backward()
            lost_d+=loss_d
            lost_g+=loss_g
            g_opt.step()
            print('[%d, %5d] d_loss: %.5f g_loss: %.5f' %
            (epoch + 1, index + 1, lost_d, lost_g)) # Printing loss
            lost_d = 0.0
            lost_g = 0.0
         torch.save(Discriminator.state_dict(),'colorize-image-dis.pth')
         torch.save(Generator.state_dict(),'colorize-image-gen.pth')
        print("Training Finished");
fit(epochs)

# Function for evaluation over entire dataset
def eval():
    Generator.eval()
    Discriminator.eval()
    running_loss=0.0
    steps=0
    for index,batch,batch_grey in tqdm(enumerate(lab_dl)):
        steps+=1
        to_device(batch,device)
        l_images=batch_grey # Extracting first channel about luminescence
        c_images=batch # Extracting channels containing color info
        mean = torch.Tensor([0.5])
        to_device(mean,device)
        # Transforming into range of -0.5 to 0.5
        l_images=l_images-mean.expand_as(l_images)
        l_images=2*l_images
        c_images=c_images-mean.expand_as(c_images)
        c_images=2*c_images
        l_images=Variable(l_images)
        c_images=Variable(c_images)
        fake_images=Generator(l_images)
        loss = nn.L1Loss(fake_images,c_images)
        running_loss+=loss
    print("Absolute mean error:",running_loss/steps)


# Helper function to colorize individual images
def show_generated_images(image):
    to_device(image,device)
    l_image=image[ 0, :, :]
    c_image=image[ 1:, :, :]
    fake_image=Generator(l_image)
    to_device([l_image,c_image,fake_image], device)
    show_img(image, "RBG Image:")
    show_img(l_image, "BW Image:")
    show_img(cat([l_image,fake_image],1), "Generated:")
ans=(input("Evaluate the model over complete dataset ?"))
if ans=="yes":
    eval()
elif not(ans=="no"):
    print("Enter correct expression")
print("Individual result:")
img=random.choice(lab_images)
show_generated_images(img)