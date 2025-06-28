import torch
import torch.nn as nn

class WasteClassifierNN(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1)
        
        dummy_input = torch.zeros(1,3,224,224)
        dummy_output = self.forward_features(dummy_input)
        flattened_size = dummy_output.view(-1).shape[0]
        self.fc1 = nn.Linear(flattened_size,128)
        self.fc2 = nn.Linear(128,10)
    
    def forward_features(self,x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        return x
    def forward(self,x):
       x = self.forward_features(x)
       x = x.view(x.size(0),-1)
       x = torch.relu(self.fc1(x))
       x = self.fc2(x)
       return x