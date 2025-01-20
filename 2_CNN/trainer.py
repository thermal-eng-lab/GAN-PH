# train CNN model
# Surrogate model for estimating the specific surface area

# -------- import library -------- #
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
import torch

# -------- Functions -------- #
class Trainer():
    def __init__(self, model, optimizer, criterion, epochs, device, dl_train, dl_valid):
        self.model = model          # network model
        self.optimizer = optimizer  # optimizer
        self.criterion = criterion  # loss function
        self.epochs = epochs        # number of epochs
        self.device = device        # device, cpu or gpu
        self.dl_train = dl_train    # training data loader
        self.dl_valid = dl_valid    # validation data loader
        self.save_epoch = 10        # save model every 10 epochs


    def train(self):
        """ 
        Function to train the model
        """
        for epoch in range(self.epochs):

            # ----- Training ----- #
            self.model.train()
            running_loss = 0.0

            for i, data in enumerate(self.dl_train):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = torch.sqrt(self.criterion(outputs, labels))
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            # ----- Validation ----- #
            self.model.eval()
            valid_loss = 0.0
            with torch.no_grad():
                for i, data in enumerate(self.dl_valid):
                    inputs, labels = data[0].to(self.device), data[1].to(self.device)
                    outputs = self.model(inputs)
                    loss = torch.sqrt(self.criterion(outputs, labels))
                    valid_loss += loss.item()

            print(
                "[{:03}/{:03}] Train loss: {:.4f} Valid loss: {:.4f}".format(
                    epoch+1, self.epochs, running_loss/len(self.dl_train), valid_loss/len(self.dl_valid)
            ))

            file = open("log.dat", "a")
            file.write("[{:03}/{:03}] Train loss: {:.4f} Valid loss: {:.4f}\n".format(
                epoch+1, self.epochs, running_loss/len(self.dl_train), valid_loss/len(self.dl_valid)
            ) + "\n")
            file.close()

            if (epoch+1) % self.save_epoch == 0:
                os.makedirs("save_model", exist_ok=True)
                torch.save(self.model.state_dict(), "save_model/model_{:03}epoch.pth".format(epoch+1))
    
        print("Finished Training")
    
        return  
