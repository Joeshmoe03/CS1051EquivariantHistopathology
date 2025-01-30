from monai.losses import DiceCELoss, DiceLoss, MaskedDiceLoss
from monai.networks.utils import one_hot
from albumentations.pytorch import ToTensorV2
import albumentations as A
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch
import os
from comet_ml import Experiment
import sys
import matplotlib.pyplot as plt
import cv2
import csv
import json
import argparse
from tqdm import tqdm

path = os.getcwd()
par_path = os.path.abspath(os.pardir)
sys.path.append(par_path)

from model.equivariantUnet import EqUnetVariant
from util.dataloader import OcelotDatasetLoaderV1

experiment = Experiment(
    api_key="Fl7YrvyVQDhLRYuyUfdHS3oE8",
    project_name="deep-learning",
    workspace="joeshmoe03",
)

def main(args):
    # where to find data
    dataDir = args.dataDir

    # data manifests for indexing
    trainDataManifest = os.path.join(dataDir, 'train_data.csv')
    valDataManifest = os.path.join(dataDir, 'val_data.csv')
    testDataManifest = os.path.join(dataDir, 'test_data.csv')

    # we only apply augmentations for non-equivariant
    trainTransforms = A.Compose([#A.ElasticTransform(p=0.2),
                                 #A.HorizontalFlip(p=0.5),
                                 #A.RandomRotate90(p=0.5),
                                 #A.Resize(256, 256),# TODO: change image size to 256x256
                                 A.Normalize(mean = 0.0, std=1, always_apply=True),
                                 ToTensorV2()])
    valTestTransforms = A.Compose([A.Normalize(mean = 0.0, std=1, always_apply=True),
                                   ToTensorV2()])   

    # custom format for loading data
    trainData = OcelotDatasetLoaderV1(dataDir=dataDir, dataManifest=trainDataManifest, transforms=trainTransforms, trainMode='train', multiclass=True)
    valData = OcelotDatasetLoaderV1(dataDir=dataDir, dataManifest=valDataManifest, transforms=valTestTransforms, trainMode='val', multiclass=True)

    batch_size = args.batchSize
    num_workers = args.numWorkers

    # pass to built-in pytorch dataloader
    trainLoader = DataLoader(trainData, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valLoader = DataLoader(valData, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EqUnetVariant(img_channels=args.imgChannel, out_channels=args.outputChannel, in_channels=args.inputChannel).to(device)
    print(f"Using device: {device}")

    # training parameters passed as args
    lr = args.learningRate
    n_epochs = args.epochs
    out_channels = args.outputChannel
    img_channels = args.imgChannel
    input_channels = args.inputChannel

    # loss function and optimizer
    optim = Adam(model.parameters(), lr=lr)
    loss = DiceCELoss(sigmoid=True) if out_channels == 1 else DiceCELoss(softmax=True, to_onehot_y=True)

    train_losses = []
    val_losses = [] 
    best_val_loss = float('inf')

    # training loop
    for epoch in range(n_epochs):
        model.train()
        running_train_loss = 0.0 
        for batch in tqdm(trainLoader):
            _, cellImg, _, cellMask = batch
            cellImg = cellImg.to(device)
            cellMask = cellMask.to(device).long()

            optim.zero_grad()
            pred = model(cellImg)  # Output shape: (B, 3, H, W)

            train_loss = loss(pred, cellMask.unsqueeze(1))  # Expecting cellMask to be (B, H, W)
            train_loss.backward()
            optim.step()
            optim.zero_grad()

            running_train_loss += train_loss.item()

        train_loss = running_train_loss / len(trainLoader)
        train_losses.append(train_loss)
        experiment.log_metric("train_loss", train_loss, step=epoch)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(valLoader):
                _, cellImg, _, cellMask = batch
                cellImg = cellImg.to(device)
                cellMask = cellMask.to(device).long()

                pred = model(cellImg)
                val_loss = loss(pred, cellMask.unsqueeze(1))
                val_loss += val_loss.item()

        val_loss /= len(valLoader)
        val_losses.append(val_loss)

        print(f'Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        experiment.log_metric("val_loss", val_loss, step=epoch)

        # save model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_dir = os.path.join('output', f"lr{str(lr)}", 'equiv1024equalparamnoaug')
        
        # Create the directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Save the model to the file within the directory
        save_path = os.path.join(save_dir, 'model.pth')
        torch.save(model.state_dict(), save_path)

    # save losses to csv
    loss_dir = os.path.join('output', f"lr{str(lr)}", 'equiv1024equalparamnoaug')
    if not os.path.exists(loss_dir):
        os.makedirs(loss_dir)

    loss_file = os.path.join(loss_dir, 'losses.txt')
    with open(loss_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['train_loss', 'val_loss'])
        for train, val in zip(train_losses, val_losses):
            writer.writerow([train, val])

    return train_losses, val_losses

if __name__ == "__main__":
    print("Current dir: ", os.getcwd())
    defaultDataDir = os.path.join('data', 'ocelot2023_v1.0.1')

    parser = argparse.ArgumentParser(description='Train equivariant U-Net')
    parser.add_argument('-imgch'              ,type=int  , action="store", dest='imgChannel'   , default=3               )
    parser.add_argument('-inch'               ,type=int  , action="store", dest='inputChannel'   , default=64            )
    parser.add_argument('-ouch'              ,type=int  , action="store", dest='outputChannel'   , default=3             )
    parser.add_argument('-lr'               ,type=float, action="store", dest='learningRate'     , default=1e-4          )
    parser.add_argument('-nepoch'           ,type=int  , action="store", dest='epochs'           , default=100           )
    parser.add_argument('-batchSize'        ,type=int  , action="store", dest='batchSize'        , default=5             )
    parser.add_argument('-dataDir'          ,type=str  , action="store", dest='dataDir'          , default=defaultDataDir)
    parser.add_argument('-numWorkers'       ,type=int  , action="store", dest='numWorkers'       , default=3             )
    args = parser.parse_args()
    main(args)
