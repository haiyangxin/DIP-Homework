import torch.nn as nn

# class FullyConvNetwork(nn.Module):

#     def __init__(self):
#         super().__init__()
#          # Encoder (Convolutional Layers)
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(3, 8, kernel_size=4, stride=2, padding=1),  # Input channels: 3, Output channels: 8
#             nn.BatchNorm2d(8),
#             nn.ReLU(inplace=True)
#         )
#         ### FILL: add more CONV Layers
        
#         # Decoder (Deconvolutional Layers)
#         ### FILL: add ConvTranspose Layers
#         ### None: since last layer outputs RGB channels, may need specific activation function

#     def forward(self, x):
#         # Encoder forward pass
        
#         # Decoder forward pass
        
#         ### FILL: encoder-decoder forward pass

#         output = ...
        
#         return output
    

class FullyConvNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        
        # Encoder (Convolutional Layers)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1),  # Input channels: 3, Output channels: 8
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=4, stride=2, padding=1),  # Input channels: 8, Output channels: 16
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=4, stride=2, padding=1),  # Input channels: 8, Output channels: 16
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # Input channels: 16, Output channels: 32
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),  # Input channels: 32, Output channels: 64
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5)
        )
        
        # Decoder (Deconvolutional Layers)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),  # Output channels: 32
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # Output channels: 16
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1),   # Output channels: 8
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(64, 16, kernel_size=4, stride=2, padding=1),   # Output channels: 8
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),    # Output channels: 3 for RGB
            nn.Tanh()  # Since last layer outputs RGB, we use Sigmoid for normalized output
        )

    def forward(self, x):
        # Encoder forward pass
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        
        # Decoder forward pass
        d1 = self.deconv1(x5)
        d2 = self.deconv2(d1)
        d3 = self.deconv3(d2)
        d4 = self.deconv4(d3)
        output = self.deconv5(d4)
        
        return output
