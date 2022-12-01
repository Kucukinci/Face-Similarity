
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import torch
from torch import optim
from config import Config
from dataset import SiameseNetworkDataset
from model_builder import SiameseNetwork
from loss import ContrastiveLoss

training_dir = "/Users/ahmet/PycharmProjects/Intern/data/faces/training"
testing_dir = "/Users/ahmet/PycharmProjects/Intern/data/faces/testing"
train_batch_size = 32
train_number_epochs = 20

folder_dataset = dset.ImageFolder(root=Config.training_dir)
siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                        transform=transforms.Compose([transforms.Resize((100,100)),
                                                                      transforms.ToTensor()]))
train_dataloader = DataLoader(siamese_dataset,
                        shuffle=True,
                        num_workers=0,
                        batch_size=Config.train_batch_size)



mps_device = torch.device("mps")
net = SiameseNetwork().to(mps_device)
criterion = ContrastiveLoss()
optimizer = optim.Adam(net.parameters(),lr = 0.0005 )


def train():
    counter =[]
    loss_history = []
    iteration_number = 0
    for epoch in range(0, Config.train_number_epochs):
        for i, data in enumerate(train_dataloader, 0):
            img0, img1, label = data
            img0, img1, label = img0.to(mps_device), img1.to(mps_device), label.to(mps_device)
            optimizer.zero_grad()
            output1, output2 = net(img0, img1)
            loss_contrastive = criterion(output1, output2, label)
            loss_contrastive.backward()
            optimizer.step()
            if i % 10 == 0:
                print("Epoch number {}\n Current loss {}\n".format(epoch, loss_contrastive.item()))
                iteration_number += 10
                counter.append(iteration_number)
                loss_history.append(loss_contrastive.item())
    return net

model = train()
torch.save(model.state_dict(), "model.pt")
print("Model Saved Successfully")



