from torch import nn

class Net(nn.Module):

    def __init__(self, n_classes=10):
        super().__init__()
        
        # Input is [B, 3, 64, 64]
        self.convolutions = nn.Sequential( 
            nn.Conv2d(3, 32, kernel_size = 3, padding = 1), #[B, 32, 64, 64]
            nn.ReLU(),  #[B, 32, 64, 64]
            nn.Conv2d(32,64, kernel_size = 3, stride = 1, padding = 1),  #[B, 64, 64, 64]
            nn.ReLU(),  #[B, 64, 64, 64]
            nn.MaxPool2d(2,2),  #[B, 64, 32, 32]
        
            nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1), #[B, 128, 32, 32]
            nn.ReLU(), #[B, 64, 32, 32]
            nn.Conv2d(128 ,128, kernel_size = 3, stride = 1, padding = 1), #[B, 128, 32, 32]
            nn.ReLU(), #[B, 128, 32, 32]
            nn.MaxPool2d(2,2), #[B, 128, 16, 16]
            nn.Dropout(0.2), #To Prevent Overfitting

            nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1), #[B, 128256, 16, 16]
            nn.ReLU(), #[B, 256, 16, 16]
            nn.Conv2d(256,256, kernel_size = 3, stride = 1, padding = 1), #[B, 256, 16, 16]
            nn.ReLU(), #[B, 256, 16, 16]
            nn.MaxPool2d(2,2), #[B, 256, 8, 8]
            nn.Dropout(0.15) #To Prevent Overfitting
        )

        # Input will be reshaped from [B, 256, 8, 8] to [B, 256*8*8] for fully connected layers
        self.fully_connected = nn.Sequential(
            nn.Linear(256*8*8, 16), # [B, 16]
            nn.ReLU(inplace=True), # [B, 16]
            nn.Linear(16, n_classes), # [B, n_classes]
        )


    def forward(self, img):

        # Apply convolution operations
        x = self.convolutions(img)

        # Reshape
        x = x.view(x.size(0), -1)

        # Apply fully connected operations
        x = self.fully_connected(x)

        return x