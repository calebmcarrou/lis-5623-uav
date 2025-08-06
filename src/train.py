#######################################################################################
# train.py
#######################################################################################

import logging
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
        self.val_epochs = 2

    def train(self, train_dl):
        """Train our model on UAV dataset

        Args:
            train_dl: (DataLoader) our dataloader

        Returns:
            None
        """
        logging.info('Beginning Training with ' + str(self.epochs) + ' epochs.')
        for epoch in range(self.epochs):
            logging.info('Starting Epoch: '+str(epoch))
            rloss = 0.0 # running loss
            for idx, data in enumerate(train_dl):
                image, label = data
                self.optimizer.zero_grad() # zero gradients
                # get model outputs and backprop
                outputs = self.cnn(image)
                loss = self.criterion(outputs, label)
                loss.backward()
                self.optimizer.step()
                rloss += loss.item() # update epoch loss
                # every 100 batches in epoch, print stats
                logging.info('Epoch: ' + str(epoch) + ', Batch: '+str(idx+1)+', Loss: ' + str(rloss/(idx+1)))
            rloss = 0 # reset after epoch
            # save every epoch
            torch.save(self.cnn.state_dict(), 'data/models/uav_epoch_'+str(epoch)+'.pth')
            logging.info('Saved Checkpoints model!')
        logging.info('Training Finished! Saving model...')

    def eval(self, test_dl, model_ckpt: str):
        """Test the model after it's trained"""
        model = CNN()
        model.load_state_dict(torch.load(model_ckpt, weights_only=True)) # load trained weights
        TP = FP = FN = 0 #setup for precision calc
        with torch.no_grad():
            for data in test_dl:
                image, label = data
                outputs = model(image)
                _, predicted=torch.max(outputs, 1)
                #bene nota: 0 is uav so positive is 0
                TP += ((predicted == 0) & (label == 0)).sum().item()
                FP += ((predicted == 0) & (label == 1)).sum().item()
                FN += ((predicted == 1) & (label == 0)).sum().item()
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        print('Model evaluated:\nPrecision: '+str(precision)+' | Recall: '+str(recall))

if __name__ == "__main__":
    logging.basicConfig(
        filename='val_log.log',
        filemode='w',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO  
    )
    logging.info('Setting device to CPU')
    device = torch.device('cpu')
    logging.info('Initiating Dataloaders')
    train_dl, test_dl = create_dataloaders(batch_size=16)
    logging.info('Creating CNN Model')
    T = Trainer()
    #T.train(train_dl)
    T.eval(test_dl, 'data/models/uav_epoch_9.pth')