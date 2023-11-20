# this script is to organize folder

#Data/folderX/size/cell_type -> Data/size/train/cell_type
# folder1 -> train
# folder2 -> val
# folder3 -> test

import pandas as pd
import os
import shutil
import glob
import argparse

patch_size = [20,50,100,200]
for _size in patch_size:
    train_dir = f'/home/bear/Documents/harry/CellPathoX-dev/Data/folder_1/{_size}'
    train_img_dirs = glob.glob(f'{train_dir}/*/*')
    train_label = [int(_dir.split('/')[-2]) for _dir in train_img_dirs]
    split  = ['train']*len(train_img_dirs)

    train_df = pd.DataFrame({'image_dir':train_img_dirs, 'label':train_label, 'split':split})
    

    val_dir = f'/home/bear/Documents/harry/CellPathoX-dev/Data/folder_2/{_size}'
    val_img_dirs = glob.glob(f'{val_dir}/*/*')
    val_label = [int(_dir.split('/')[-2]) for _dir in val_img_dirs]
    split  = ['val']*len(val_img_dirs)

    val_df = pd.DataFrame({'image_dir':val_img_dirs, 'label':val_label, 'split':split})

    test_dir = f'/home/bear/Documents/harry/CellPathoX-dev/Data/folder_3/{_size}'
    test_img_dirs = glob.glob(f'{test_dir}/*/*')
    test_label = [int(_dir.split('/')[-2]) for _dir in test_img_dirs]
    split  = ['test']*len(test_img_dirs)

    test_df = pd.DataFrame({'image_dir':test_img_dirs, 'label':test_label, 'split':split})

    df = pd.concat([train_df, val_df, test_df], axis = 0)

    df.to_csv(f'/home/bear/Documents/harry/CellPathoX-dev/Data/{_size}.csv', index = False)
    




