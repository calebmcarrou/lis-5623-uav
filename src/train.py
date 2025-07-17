#######################################################################################
# train.py
#######################################################################################

import torch
import torch.nn as nn
import torch.optim as optim

from cnn import CNN
from UAVDataset import UAVDataset, create_dataloaders

class Trainer:

    def __init__(self):
        self.cnn = CNN()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.cnn.parameters(),
                                   lr=0.001,
                                   momentum=0.9)
        self.epochs = 10
        self.val_epochs = 5

    def train(self, train_dl):
        """Train our model on UAV dataset

        Args:
            train_dl: (DataLoader) our dataloader

        Returns:
            None
        """
        print('Beginning Training with ' + str(self.epochs) + 'epochs.')
        for epoch in self.epochs:
            rloss = 0.0 # running loss
            for idx, data in enumerate(train_dl):
                image, label, bbox = data
                self.optimizer.zero_grad() # zero gradients
                # get model outputs and backprop
                outputs = self.cnn(image)
                loss = self.criterion(outputs, label)
                loss.backward()
                self.optimizer.step()
                rloss += loss.item() # update epoch loss
                # every 100 batches in epoch, print stats
                if idx % 100 == 99:
                    print('Epoch: ' + str(self.epoch + 1) + ', Loss: ' + str(rloss))
                    rloss = 0.0
        print('Training Finished! Saving model...')
        torch.save(self.cnn.state_dict(), 'data/models/uav.pth')

if __name__ == "__main__":
    print('Setting device to CPU')
    device = torch.device('cpu')
    print('Initiating Dataloaders')
    train_dl, test_dl = create_dataloaders(batch_size=16)
    print('Creating CNN Model')
    T = Trainer()
    