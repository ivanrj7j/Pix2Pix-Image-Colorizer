import torch
from generator import Generator
from PIL import Image
from kornia.color import lab_to_rgb, rgb_to_lab
import numpy as np
from torchvision.utils import save_image


device = "cuda" if torch.cuda.is_available() else "cpu"

model = Generator(1, 2)
model.load_state_dict(torch.load("checkpoints/gen.pth"))
model.to(device)

def colorizeImage(imagePath:str, save=False, outputName:str=None):
    image = Image.open(imagePath)
    image = image.resize((256, 256))
    image = torch.tensor(np.array(image)).unsqueeze(0)/255
    image = torch.mean(image, -1)
    image = torch.reshape(image, (1, 1, 256, 256))
    image = image.to(device)

    with torch.no_grad():
        ab = model(image)*128
        l = image*100
        output = lab_to_rgb(torch.cat((l, ab), 1)).cpu()
    
    if save:
        save_image(output, outputName)

    return output