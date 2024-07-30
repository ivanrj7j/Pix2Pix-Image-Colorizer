import os
import torch

trainFolder = os.path.join("data", "train")
testFolder = os.path.join("data", "test")
# folder paths 

epochs = 250
batchSize = 8
lr = 2e-4
betas = (0.5, 0.999)
l1Lambda = 100
# training config 

device = "cuda" if torch.cuda.is_available() else "cpu"
# device 

numWorkers = 8
# number of workers to load data in dataloader 

checkpointDirectory = "checkpoints"
genCheckPointTemplate = "gen-{t}.pth"
discCheckPointTemplate = "disc-{t}.pth"
checkPointEvery = 5
outputDirectory = "outputs"
# checkpoint configuration 


if not os.path.exists(checkpointDirectory):
    os.mkdir(checkpointDirectory)

if not os.path.exists(outputDirectory):
    os.mkdir(outputDirectory)
