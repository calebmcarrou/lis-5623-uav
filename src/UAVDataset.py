import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class UAVDataset(Dataset):

    def __init__(self, subset: str):
        ### creates a dataset loader for the UAV dataset
        self.idx = 0
        self.img_size = 640
        # we need to normalize our values. why not just do N(0,1)? well, you could
        # but industry standard is to use RGB specs from your dataset or other
        # standards such as ImageNet
        self.stdize = {'mean': [0.655, .67, .67],
                         'sd': [0.221, 0.205, 0.224]}
        # set up for train or val
        if subset == 'train':
            img_path, lab_path = 'data/dataset/images/train', 'data/dataset/labels/train'
        else:
            img_path, lab_path = 'data/dataset/images/val', 'data/dataset/labels/val'
        self.images = [os.path.join(img_path, f) for f in os.listdir(img_path)]
        self.labels = [os.path.join(lab_path, f) for f in os.listdir(lab_path)]

        # resize all to (640x640) and make tensor
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(self.stdize['mean'], self.stdize['sd'])
        ])

    def __len__(self):
        return(len(self.images))

    def __getitem__(self, idx):
        """returns next image

        Returns
        -------
            truth: (tuple) (img, bbox)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # read image
        img = cv2.imread(self.images[idx])
        height, width, _ = img.shape # need this to convert yolo to pixel
        # massage label out to get class (always 0) and bbox
        with open(self.labels[idx], 'r') as f:
            label = f.read()
        # just grab the first line if more than one for this exericse
        label = label.split('\n')[0]
        label = [float(i) for i in label.split(' ')]
        category = int(label[0])
        # convert bbox from yolo to bbox size
        # yolo is (x_center, y_center, width, height) as fraction [0,1]
        x, y, w, h = label[1:] # yolo format as fraction
        x *= width # multiply all by real image to get actual values
        y *= height
        w *= width
        h *= height
        # turn to top-left corner for patch
        x1 = max(0, int(x - w / 2))
        y1 = max(0, int(y - h / 2))
        x2 = min(width, int(x + w / 2))
        y2 = min(height, int(y + h / 2))

        patch = img[y1:y2, x1:x2] # create crop around bbox
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        patch = cv2.resize(patch, (32, 32)) # resize to 32x32
        patch = self.transform(patch)

        return patch, category

def create_dataloaders(batch_size: int=16):
    """Create the train & test dataloaders

    Args:
        batch_size (int) batch size for loaders
    
    Returns:
        train_dataloader (DataLoader)
        test_dataloader (DataLoader)
    """
    # create sets
    train_data = UAVDataset(subset='train')
    test_data = UAVDataset(subset='test')
    # create dataloaders
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader

if __name__ == "__main__":
    '''
    U = UAVDataset('train')
    img, label = U.__getitem__(0)
    print(label)
    fig, ax = plt.subplots()
    ax.imshow(img.permute(1,2,0))
    plt.savefig('data/test.png')
    '''
    train_dl, test_dl = create_dataloaders()