import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colors
import argparse

import os 
from scipy import ndimage as ndi
from skimage import (
    color, feature, filters, measure, morphology, segmentation, util
)
from skimage.color import label2rgb


from tqdm import tqdm
class extract_annotation:
    def __init__(self, image_npy_dir, types_npy_dir, masks_npy_dir):
        self.image_npy_dir = image_npy_dir
        self.types_npy_dir = types_npy_dir
        self.masks_npy_dir = masks_npy_dir
        

        # load data in the most efficient way, 
        # we only need one image and its corresponding label and mask
        
        self.tissue_type = np.load(self.types_npy_dir)
        
        self.image = np.load(self.image_npy_dir)
        self.mask = np.load(self.masks_npy_dir)

        


        
        
        #self.mask = np.argmax(self.mask, axis = -1)
        #self.mask = np.expand_dims(self.mask, axis = -1)

        

        self.cmap = colors.ListedColormap(['black', 'red', 'blue', 'green', 'yellow', 'white'])
        self.bounds = [0,1,2,3,4,5]
        #self.norm = colors.BoundaryNorm(self.bounds, self.cmap.N)

    def extract_cells(self, _image, _center, radius):
        """
        extract the cell with the given center and radius
        given a list of cropped images 
        """
        image_crops = []
        for _radius in radius:
            _patch = _image[_center[0] - _radius: _center[0] + _radius,
                             _center[1] - _radius: _center[1] + _radius]
            
            
            assert _patch.shape[0] == _patch.shape[1], "the patch is not a square"
            assert _patch.shape[2] == 3, "the patch should have 3 channels"

            image_crops.append(_patch)

        return image_crops
    
    def extract(self, radius = [100,50,25,10], save_dir = None):
        ### first extract the instance for each cell
        ### find the center for the cells
        ### check if we are able  to extract the cell with the max radius
        ### if yes, then extract the cell with the all the radius
        ### if no, then skip the cell

        # return is stored in a dictionary

        self.data = {}
        self.data['name'] = []
        for _radius in radius:
            self.data[f'size_{_radius*2}'] = []
        self.data['category'] = []
        

        for _index in tqdm(range(self.image.shape[0])):
            
            _image = self.image[_index]
            _mask_orig = self.mask[_index]

            # create a color map for the mask
        
            # convert the npy image to uint8
            _image = _image.astype(np.uint8)
            
            # convert (256,256,3) to (256,256)
            _mask = np.squeeze(_mask_orig)

            
            # how many instance in the mask, any value greater than , because the 0 is not a instance and 1 is the background
            _instance = np.unique(_mask)
            # remove 0 and 1 
            if len(_instance) >= 2:
                assert _instance[0] == 0, "the first value should be 0"
                assert _instance[1] == 1, "the second value should be 1"
                _instance = _instance[2:]

            elif len(_instance) == 1:
                # show the image
                

                assert _instance[0] == 0, "the first value should be 0"
                continue
            else:
                continue
            # assert _instance[:2] is 0 and 1
            
            
            # extract the instance
            for _instance_index in range(len(_instance)):
                _instance_value = _instance[_instance_index]
                # extract the instance
                _instance_mask = np.where(_mask == _instance_value, 1, 0)
                # check the value is in which channel 0 -4, 
                sum_value = np.sum(_instance_mask, axis = (0,1))
                # find the channel with the max value
                _channel = np.argmax(sum_value)
                # extract the instance
                _instance_mask = _instance_mask[:,:, _channel]
                # find the center of the instance, the mean of the x and y value when the value is 1
                _center = np.where(_instance_mask == 1)
                _center = np.mean(_center, axis = 1)

                # check if we are able to extract the cell with the max radius

                ## make them int
                _center = (int(_center[0]), int(_center[1]))
                # check if we are able to extract the cell with the max radius
    
                max_radius = max(radius)
                
                if _center[0] - max_radius >= 0 and _center[0] + max_radius < _image.shape[0] and _center[1] - max_radius >= 0 and _center[1] + max_radius < _image.shape[0]:
                    

                    cropped_images_list = self.extract_cells(_image, _center, radius)
                    self.data['name'].append(str(_instance_index) + '_' + str(_instance) +'_'+ str(_center)[0] + '_' + str(_center)[1])
                    # add the center

                    for _num in range(len(radius)):
                        self.data[f'size_{radius[_num]*2}'].append(cropped_images_list[_num])

                        _name = str(_channel) + '_' + str(_instance_index) +'_'+ str(_center[0])+ '_' + str(_center[1])
                        _category = _channel
                        _size = radius[_num]*2
                        _image = cropped_images_list[_num]

                        # create the directory if not exist


                        self.save(_size, _image, _name, _category, save_dir = save_dir)

                    self.data['category'].append(_channel)
                else:
                    pass
        return self.data

                


    def save(self, size, image, name, category,save_dir= None):
        assert save_dir is not None, "please specify the save directory"
        # save each image by name + size, under the save_dir/size/category
        # save the data
        
        # size 
        _dir = save_dir + '/' + str(size)
        if not os.path.exists(_dir):
            os.makedirs(_dir)
        _dir = f'{_dir}/{category}'
        if not os.path.exists(_dir):
            os.makedirs(_dir)
        _dir = f'{_dir}/{name}.png'
        plt.imsave(_dir, image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_npy_dir', type = str, default = 'images.npy', help = 'the directory of the image npy file')
    parser.add_argument('--types_npy_dir', type = str, default = 'types.npy', help = 'the directory of the types npy file')
    parser.add_argument('--masks_npy_dir', type = str, default = 'masks.npy', help = 'the directory of the masks npy file')
    parser.add_argument('--image_folder', type = str, default = 'images', help = 'the directory of the image folder')
    args = parser.parse_args()

    extract_annotation = extract_annotation(args.image_npy_dir, args.types_npy_dir, args.masks_npy_dir)
    metatada = extract_annotation.extract(save_dir=args.image_folder)

    # how many datapoint we have
    print('How many datapoint we have')
    print(len(metatada['name']))


    # how to ran this file
    # example 
    
    # python utils/extract_cells_correct.py --image_npy_dir ../Dataset/pannuke/Fold_2/images/fold2/images.npy --types_npy_dir ../Dataset/pannuke/Fold_2/images/fold2/types.npy --masks_npy_dir ../Dataset/pannuke/Fold_2/masks/fold2/masks.npy --image_folder ./Data/folder_2
    # python utils/extract_cells_correct.py --image_npy_dir ../Dataset/pannuke/Fold_3/images/fold3/images.npy --types_npy_dir ../Dataset/pannuke/Fold_3/images/fold3/types.npy --masks_npy_dir ../Dataset/pannuke/Fold_3/masks/fold3/masks.npy --image_folder ./Data/folder_3
    # python utils/extract_cells_correct.py --image_npy_dir ../Dataset/pannuke/Fold_1/images/fold1/images.npy --types_npy_dir ../Dataset/pannuke/Fold_1/images/fold1/types.npy --masks_npy_dir ../Dataset/pannuke/Fold_1/masks/fold1/masks.npy --image_folder ./Data/folder_1

            
