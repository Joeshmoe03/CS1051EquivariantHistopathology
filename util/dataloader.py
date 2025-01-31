import os
from torch.utils.data import Dataset
from PIL import Image
import json
import numpy as np
import torch
import cv2
import csv
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

def generate_cell_masks(annDir, partitions = ['train', 'val', 'test'], radius = 30):
    '''
    Generates cell masks from the cell annotations (set of coordinates + some radius) in the annotations directory.
    Cell masks are saved in the same directory as the cell annotations, but in a subdirectory called cell_masks.
    '''
    for partition in partitions:
        annCellDir = os.path.join(annDir, partition, 'cell')
        cellAnnFiles = os.listdir(annCellDir)
        for cellAnnFile in cellAnnFiles:

            # read in the csv: contains coordinates of cells and their class
            cellAnnPath = os.path.join(annCellDir, cellAnnFile)

            # maybe be empty csv if no cells
            if os.path.getsize(cellAnnPath) > 0:
                cellAnn = pd.read_csv(cellAnnPath).to_numpy()
            else:
                cellAnn = np.empty((0, 0, 0))

            # generate the mask with circles of bounding box around each cell and color based on class
            mask = np.zeros((1024, 1024))
            for cell in cellAnn:
                x, y, lbl = cell

                # color based on class: handles unexpected classes by setting to background
                if int(lbl) == 1 or int(lbl) == 2:
                    color = int(lbl)
                else:
                    color = 0
                cv2.circle(mask, (int(x), int(y)), radius, color, -1)
    
            # save the mask to annotations/partition/cell_masks as a .png
            maskPath = os.path.join(annDir, partition, 'cell_masks', cellAnnFile.replace('csv', 'png'))

            if not os.path.exists(os.path.dirname(maskPath)):
                os.makedirs(os.path.dirname(maskPath))

            cv2.imwrite(maskPath, mask)

class OcelotDatasetLoaderV1(Dataset):
    def __init__(self, dataDir: str, dataManifest: str, trainMode = 'train', transforms=None, multiclass=True):
        '''
        Custom dataset loader for the Ocelot dataset. Trainmode should be one of ['train', 'val', 'test'],
        and always with corresponding dataManifest (trainManifest, valManifest, testManifest) or else bad indexing things will happen.
        This dataset should be treated as multiclass given background, cancer, and non-cancer classes, but it can be treated as binary if ignoring background
        in some scenarios.
        '''
        self.dataManifest = pd.read_csv(dataManifest)

        # Sanity check on right directory
        if not os.path.exists(dataDir):
            raise FileNotFoundError('dataDir does not exist')
        
        # Sanity check on existing dataManifest
        if not os.path.exists(dataManifest):
            raise FileNotFoundError('dataManifest does not exist')
        
        # Sanity check on valid trainMode
        if trainMode not in ['train', 'val', 'test']:
            raise ValueError('trainMode must be one of ["train", "val", "test"]')

        # args
        self.transforms = transforms
        self.dataDir = dataDir
        self.multiclass = multiclass
        self.trainMode = trainMode

        # metadata.json contains the coordinates of the cell images in the tissue images
        self.jsonObject = json.load(open(os.path.join(dataDir, 'metadata.json')))

        # convert manifest to list of file names and shuffle in case weird ordering of images
        self.dataManifest = list(pd.read_csv(dataManifest, header=None).loc[:,0])
        self.dataManifest = np.random.permutation(self.dataManifest)
        print(f'Found {len(self.dataManifest)} images for {trainMode} mode...')

        # path stuff (see ocelot data format stuff)
        self.tissImgAbsPath = os.path.join(self.dataDir, 'images', self.trainMode, 'tissue')
        self.cellImgAbsPath = os.path.join(self.dataDir, 'images', self.trainMode, 'cell')
        self.tissAnnAbsPath = os.path.join(self.dataDir, 'annotations', self.trainMode, 'tissue')
        self.cellAnnAbsPath = os.path.join(self.dataDir, 'annotations', self.trainMode, 'cell_masks')

    def __len__(self):
        return len(self.dataManifest)
    
    def __getitem__(self, idx):

        # uses the manifest to get a sample pair
        image_name = self.dataManifest[idx]

        tissImg = cv2.imread(os.path.join(self.tissImgAbsPath, image_name))
        cellImg = cv2.imread(os.path.join(self.cellImgAbsPath, image_name))
        tissAnn = cv2.imread(os.path.join(self.tissAnnAbsPath, image_name.replace('.jpg','.png')), 0)
        cellAnn = cv2.imread(os.path.join(self.cellAnnAbsPath, image_name.replace('.jpg','.png')), 0)

        # apply transforms to images
        if self.transforms:
            tsampleTissue = self.transforms(image=tissImg, mask=tissAnn)
            tsampleCell = self.transforms(image=cellImg, mask=cellAnn)
            tissImg = tsampleTissue['image']
            cellImg = tsampleCell['image']
            tissMask = tsampleTissue['mask']
            cellMask = tsampleCell['mask']
        
        return tissImg, cellImg, tissMask, cellMask
