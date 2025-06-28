from torchvision import datasets,transforms
from torch.utils.data import DataLoader,random_split

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
    dataset = datasets.ImageFolder(root=data_dir,transform=transform)

    #Splitting the dataset into 80% training and 20% validation
    train_size = int(0.8*len(dataset))
    val_size = len(dataset) - train_size
    train_dataset,val_dataset = random_split(dataset,[train_size,val_size])

    train_loader = DataLoader(train_dataset,batch_size=32,shuffle=True);
    val_loader = DataLoader(val_dataset,batch_size=32)
    return train_loader,val_loader
