import torch
import torch.nn as nn

class WasteClassifierNN(nn.Module):
    def __init__(self, input_shape=(3, 128, 128), dropout=0.3):
        super(WasteClassifierNN, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout)
        )
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout)
        )
        self.conv_block5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout)
        )

        # Flatten size calculation
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            print("🧪 Dummy input shape:", dummy.shape)
            dummy = self._forward_conv_debug(dummy)
            self.flatten_size = dummy.view(1, -1).shape[1]
            print(f"✅ Flatten size computed: {self.flatten_size}")

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 12)
        )

    def _forward_conv_debug(self, x):
        print("➡️ Before conv_block1:", x.shape)
        x = self.conv_block1(x)
        print("✅ After conv_block1:", x.shape)

        x = self.conv_block2(x)
        print("✅ After conv_block2:", x.shape)

        x = self.conv_block3(x)
        print("✅ After conv_block3:", x.shape)

        x = self.conv_block4(x)
        print("✅ After conv_block4:", x.shape)

        x = self.conv_block5(x)
        print("✅ After conv_block5:", x.shape)

        return x

    def _forward_conv(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = self.fc_layers(x)
        return x
