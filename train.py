import config
from generator import Generator
from discriminator import Discriminator
from dataloader import getDataLoader
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import GradScaler, autocast
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss, L1Loss
from torch import ones_like, zeros_like
import torch
import os
from torchvision.utils import save_image
from kornia.color import lab_to_rgb
import time
import json


def trainStep(l, ab, gen:Generator, disc:Discriminator, discOpt:Adam, genOpt:Adam, genScaler:GradScaler, discScaler:GradScaler, bce:BCEWithLogitsLoss, l1:L1Loss):
        l = l.to(config.device)
        ab = ab.to(config.device)
        # sending the data to the device 

        with autocast(config.device, torch.float16):
            generatedAB = gen.forward(l)
            # generating image 
            discReal = disc.forward(l, ab)
            discGenerated = disc.forward(l, generatedAB.detach())
            # discriminator predictions. here .detach method is used to avoid graph issues later 
            discRealLoss = bce.forward(discReal, ones_like(discReal))
            discGeneratedLoss = bce.forward(discGenerated, zeros_like(discGenerated))
            discLoss = (discRealLoss + discGeneratedLoss)/2
            # computing loss 

        disc.zero_grad()
        discOpt.zero_grad()
        discScaler.scale(discLoss).backward()
        discScaler.step(discOpt)
        discScaler.update()

        with autocast(config.device, torch.float16):
            discPredictions = disc.forward(l, generatedAB)
            genFakeLoss = bce.forward(discPredictions, ones_like(discPredictions))
            l1Loss = l1.forward(generatedAB, ab) * config.l1Lambda
            genLoss = genFakeLoss + l1Loss
            # computing loss 

        gen.zero_grad()
        genOpt.zero_grad()
        genScaler.scale(genLoss).backward()
        genScaler.step(genOpt)
        genScaler.update()

        return genLoss.item(), discLoss.item()


def train(train:DataLoader, test:DataLoader, gen:Generator, disc:Discriminator, discOpt:Adam, genOpt:Adam):
    bce = BCEWithLogitsLoss()
    l1 = L1Loss()

    discScaler = GradScaler(config.device)
    genScaler = GradScaler(config.device)

    history = []

    gen.train()
    disc.train()

    for epoch in range(1, config.epochs+1):
        epochStart = time.time()
        params = {}
        params["genTotalLoss"] = 0
        params["discTotalLoss"] = 0

        looper = tqdm(train, leave=False, desc=f"[{epoch}/{config.epochs}]")

        for i, (l, ab) in enumerate(looper):
            gl, dl = trainStep(l, ab, gen, disc, discOpt, genOpt, genScaler, discScaler, bce, l1)
            params["genTotalLoss"] += gl/config.epochs
            params["discTotalLoss"] += dl/config.epochs

            looper.set_postfix(params)
        
        elapsed = round(time.time() - epochStart)
        print(f"[{epoch}/{config.epochs}] discLoss: {params['discTotalLoss']} genLoss: {params['genTotalLoss']} ({elapsed}s)")
        history.append(params)

        if epoch % config.checkPointEvery == 0:
            torch.save(generator.state_dict(), os.path.join(config.checkpointDirectory, config.genCheckPointTemplate.format(t=epoch)))
            torch.save(discriminator.state_dict(), os.path.join(config.checkpointDirectory, config.discCheckPointTemplate.format(t=epoch)))

        with torch.no_grad():
            for x, y in test:
                x = x.to(config.device)
                images = gen.forward(x)

                x += 1
                x *= 50
                images *= 128

                rgb = lab_to_rgb(torch.cat((x, images), 1))
                save_image(rgb.cpu(), os.path.join(config.outputDirectory, f"generated_{epoch}.png"))
                break
        



    return history, generator, discriminator



if __name__ == "__main__":
    generator = Generator(1, 2).to(config.device)
    discriminator = Discriminator(1, 2).to(config.device)

    discOptim = Adam(discriminator.parameters(), config.lr, config.betas)
    genOptim = Adam(generator.parameters(), config.lr, config.betas)

    trainDataLoader = getDataLoader(config.trainFolder, batchSize=config.batchSize, numWorks=config.numWorkers)
    testDataLoader = getDataLoader(config.testFolder, batchSize=config.batchSize, numWorks=config.numWorkers)

    history, tranedGen, trainedDisc = train(trainDataLoader, testDataLoader, generator, discriminator, discOptim, genOptim)
    torch.save(tranedGen.state_dict(), os.path.join(config.checkpointDirectory, config.genCheckPointTemplate.format(t="final")))
    torch.save(trainedDisc.state_dict(), os.path.join(config.checkpointDirectory, config.discCheckPointTemplate.format(t="final")))

    with open("history.json", "w") as f:
        json.dump(history, f)