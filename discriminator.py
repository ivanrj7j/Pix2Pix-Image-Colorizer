from torch.nn import Module
from torch.nn import Conv2d, LeakyReLU, Dropout, BatchNorm2d, Sequential
from torch import cat

class DiscBlock(Module):
    def __init__(self, inChannels:int, outChannels:int, stride:int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layers = Sequential(
            Conv2d(inChannels, outChannels, kernel_size=4, stride=stride, padding=1, padding_mode="reflect", bias=False),
            BatchNorm2d(outChannels),
            LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.layers.forward(x)
    

class Discriminator(Module):
    def __init__(self, inChannels:int=3, outChannels:int=None, features:list[int]=[64, 128, 256, 512], *args, **kwargs) -> None:
        """
        Discriminator model predicts if the given image is real or not.
        Since the discriminator is used as a loss function for the generator, the discriminator takes in the original image and the predicted image/real image as input and
        
        Keyword arguments:
        inChannels -- number of channels of the input image
        outChannels -- number of channels of the output image (default: inChannels)
        features: a list of number of features of each block in the middle of the discriminator
        Return: None
        """
        
        super().__init__(*args, **kwargs)

        self.inChannels = inChannels

        if outChannels is None:
            outChannels = inChannels

        self.outChannels = outChannels
        # specifying the number of channels of the input and output images 

        self.layer1 = Sequential(
            Conv2d(self.inChannels+self.outChannels, features[0], 4, 2, 1, padding_mode="reflect"),
            LeakyReLU(0.2)
        )

        self.middleLayers = []
        for i in range(1, len(features)):
            stride = 2 if (i == len(features) - 1) else 1
            self.middleLayers.append(DiscBlock(features[i-1], features[i], stride))
            
        self.middleLayers = Sequential(*self.middleLayers)
        # procedurally making middle layers 

        self.finalLayer = Conv2d(features[-1], 1, 4, 1, padding_mode="reflect")


    def forward(self, x, y):
        combined = cat((x, y), 1)
        
        out = self.layer1(combined)
        out = self.middleLayers(out)
        return self.finalLayer(out)
    


if __name__ == "__main__":
    import torch
    discriminator = Discriminator(1, 2)
    x, y = torch.randn((8, 1, 256, 256)), torch.randn((8, 2, 256, 256))

    out = discriminator(x, y)
    print(out.shape)