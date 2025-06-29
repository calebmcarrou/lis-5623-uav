#######################################################################################
# misc.py
#######################################################################################

import os
import random
import shutil

def copy_yolo(names, subset):
    source = 'data/uav/drone_dataset_yolo/dataset_txt'
    dest = 'data/dataset'
    # move files
    for idx, b in enumerate(names):
        # source
        img_src = os.path.join(source, b + '.jpg')
        label_src = os.path.join(source, b + '.txt')
        # destination
        img_dst = os.path.join(dest, 'images', subset, b + '.jpg')
        label_dst = os.path.join(dest,  'labels', subset, b + '.txt')
        # copy over
        shutil.copy2(img_src, img_dst)
        shutil.copy2(label_src, label_dst)


def create_yolo_struct():
    train_split = 0.8
    source = 'data/uav/drone_dataset_yolo/dataset_txt'
    # get all pair names (.jpg and .txt are both length 4 so [:-4])
    base_names = [f[:-4] for f in os.listdir(source) if f.endswith('.jpg')]
    random.shuffle(base_names)

    # get train/val split
    sidx = int(len(base_names) * train_split)
    train_data = base_names[:sidx]
    val_data = base_names[sidx:]

    # copy over
    print('Creating Training Data Structure')
    copy_yolo(train_data, 'train')
    print('Creating Validation Data Structure')
    copy_yolo(val_data, 'val')


if __name__ == "__main__":
    # print(create_yolo_struct())