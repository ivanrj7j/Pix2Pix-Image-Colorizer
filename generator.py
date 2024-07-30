from torch.nn import Module
from torch.nn import Sequential
from torch.nn import ConvTranspose2d, Tanh, ReLU, LeakyReLU, Conv2d, BatchNorm2d, Dropout
from torch import cat


class GenBlock(Module):
    def __init__(self, inChannels:int, outChannels:int, dropout:bool=False, up:bool=False, activation:str="leaky", padding:int=1, kernelSize:int=4,  *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        layers = [
            Conv2d(inChannels, outChannels, kernelSize, 2, padding=padding, bias=False, padding_mode="reflect")           
        ]

        if up:
            layers = [ConvTranspose2d(inChannels, outChannels, kernelSize, 2, padding=padding, bias=False)]

        layers.append(BatchNorm2d(outChannels))

        if activation == "leaky":
            layers.append(LeakyReLU(0.2))
        else:
            layers.append(ReLU())

        if dropout:
            layers.append(Dropout(0.5))

        self.layers = Sequential(*layers)

    def forward(self, x):
        return self.layers.forward(x)


class Generator(Module):
    def __init__(self, inChannels:int, outChannels:int=None, features:int=64, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        """
        Generator model generates the desired output image.
        The generator takes in a latent vector and produces the corresponding image
        
        Keyword arguments:
        inChannels -- number of channels of the input latent vector
        outChannels -- number of channels of the output image (default: inChannels)
        features -- the (base) number of features in the middle layers
        Return: None
        """

        self.inChannels = inChannels

        if outChannels is None:
            outChannels = inChannels

        self.outChannels = outChannels
        # specifying the number of channels of the input and output images 


        self.initDown = Sequential(
            Conv2d(inChannels, features, 4, 2, 1, padding_mode="reflect"),
            LeakyReLU(0.2)
        )

        self.down1 = GenBlock(features, features*2)
        self.down2 = GenBlock(features*2, features*4)
        self.down3 = GenBlock(features*4, features*8)
        self.down4 = GenBlock(features*8, features*8)
        self.down5 = GenBlock(features*8, features*8)
        self.down6 = GenBlock(features*8, features*8)
        
        self.bottleNeck = Sequential(
            Conv2d(features*8, features*8, 4, 2, 1),
            ReLU()
        )
        # downscaling 

        self.up1 = GenBlock(features * 8, features * 8, up=True, activation="relu", dropout=True)
        self.up2 = GenBlock(features * 16, features * 8, up=True, activation="relu", dropout=True)
        self.up3 = GenBlock(features * 16, features * 8, up=True, activation="relu", dropout=True)
        self.up4 = GenBlock(features * 16, features * 8, up=True, activation="relu")
        self.up5 = GenBlock(features * 16, features * 4, up=True, activation="relu")
        self.up6 = GenBlock(features * 8, features * 2, up=True, activation="relu")
        self.up7 = GenBlock(features*4, features, up=True, activation="relu")

        self.final = Sequential(
            ConvTranspose2d(features*2, self.outChannels, 4, 2, 1),
            Tanh()
        )

    def forward(self, x):
        d0 = self.initDown(x)
        d1 = self.down1(d0)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        bottleNeck = self.bottleNeck(d6)

        u1 = self.up1(bottleNeck)
        u2 = self.up2(cat((u1, d6), 1))
        u3 = self.up3(cat((u2, d5), 1))
        u4 = self.up4(cat((u3, d4), 1))
        u5 = self.up5(cat((u4, d3), 1))
        u6 = self.up6(cat((u5, d2), 1))
        u7 = self.up7(cat((u6, d1), 1))

        return self.final(cat((u7, d0), 1))
        


if __name__ == "__main__":
    import torch
    import time

    gen = Generator(1, 2).to("cuda")
    x= torch.randn((64, 1, 256, 256)).to("cuda")
    start = time.time()
    output = gen(x)
    print(f"Time taken: {time.time() - start} seconds")
    print(output.shape)