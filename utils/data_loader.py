from torchvision import datasets,transforms
from torch.utils.data import DataLoader,random_split
import os

##Loading the dataset
def get_data_loaders(data_dir,batch_size=32):
    transform = transforms.Compose(
    [
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
        mean=[0.485, 0.456, 0.406],      
        std=[0.229, 0.224, 0.225]
        )
    ]
)
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    
    train_dataset= datasets.ImageFolder(root=train_dir,transform=transform)
    val_dataset = datasets.ImageFolder(root=val_dir,transform=transform)
    
    train_loader = DataLoader(train_dataset,batch_size=32,shuffle=True);
    val_loader = DataLoader(val_dataset,batch_size=32,shuffle=False)
    return train_loader,val_loader
