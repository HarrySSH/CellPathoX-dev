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

from tqdm import tqdm
class extract_annotation:
    def __init__(self, image_npy_dir, types_npy_dir, masks_npy_dir):
        self.image_npy_dir = image_npy_dir
        self.types_npy_dir = types_npy_dir
        self.masks_npy_dir = masks_npy_dir
        

        # load data in the most efficient way, 
        # we only need one image and its corresponding label and mask
        self.image = np.load(self.image_npy_dir)
        self.tissue_type = np.load(self.types_npy_dir)
        self.mask = np.load(self.masks_npy_dir)
        self.mask = np.argmax(self.mask, axis = -1)
        self.mask = np.expand_dims(self.mask, axis = -1)

        self.cmap = colors.ListedColormap(['black', 'red', 'blue', 'green', 'yellow', 'white'])
        self.bounds = [0,1,2,3,4,5]
        self.norm = colors.BoundaryNorm(self.bounds, self.cmap.N)

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



    def extract(self, radius = [100,50,25,10]):
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

        # print image shape
        print(f"the shape of the image is {self.image.shape}")

        for _index in tqdm(range(self.image.shape[0])):
            _image = self.image[_index]
            _mask_orig = self.mask[_index]

                    # create a color map for the mask
        

        

            # convert the npy image to uint8
            _image = _image.astype(np.uint8)
            
            # convert (256,256,3) to (256,256)
            _mask = np.squeeze(_mask_orig)
            

            # make a segmentation as binary, background is 0, foreground is 1
            _mask = np.where(_mask == 5, 0, 1)

            
            cells = _mask.copy()
            distance = ndi.distance_transform_edt(cells)
            local_max_coords = feature.peak_local_max(distance, min_distance=7)
            local_max_mask = np.zeros(distance.shape, dtype=bool)
            local_max_mask[tuple(local_max_coords.T)] = True
            markers = measure.label(local_max_mask)

            segmented_cells = segmentation.watershed(-distance, markers, mask=cells)

            ### get them by instance
            ### get the center for each instance
            ### get the radius for each instance
            ### extract the cell for each instance
            ### store the data in a dictionary
            ### return the dictionary
            
            segmented_cells_by_instance = measure.label(segmented_cells)

            number_of_instances = np.max(segmented_cells_by_instance)
            # getting each object
            for _instance in range(1, number_of_instances + 1):
                # get the center for each instance
                _center = np.where(segmented_cells_by_instance == _instance)

                # catogory of the cell
                category = np.unique(_mask_orig[segmented_cells])
                # use the category with the most pixels
                category = np.argmax(np.bincount(category))
                
                _center = np.mean(_center, axis = 1)
                _center = np.round(_center)
                _center = tuple(_center)
                ## make them int
                _center = (int(_center[0]), int(_center[1]))
                # check if we are able to extract the cell with the max radius
                max_radius = np.max(radius)
                
                if _center[0] - max_radius >= 0 and _center[0] + max_radius < _image.shape[0] and _center[1] - max_radius >= 0 and _center[1] + max_radius < _image.shape[0]:
                    

                    cropped_images_list = self.extract_cells(_image, _center, radius)
                    self.data['name'].append(str(_index) + '_' + str(_instance) +'_'+ str(_center)[0] + '_' + str(_center)[1])
                    # add the center

                    for _num in range(len(radius)):
                        self.data[f'size_{radius[_num]*2}'].append(cropped_images_list[_num])

                    self.data['category'].append(category)
                else:
                    pass
                
            
        return self.data
    
    def save(self, save_dir= None):
        assert save_dir is not None, "please specify the save directory"
        # save each image by name + size, under the save_dir/size/category
        # save the data
        for _index in tqdm(range(len(self.data['name']))):
            _name = self.data['name'][_index]
            _category = self.data['category'][_index]
            for _key in self.data.keys():
                if _key != 'name' and _key != 'category':
                    _size = _key.split('_')[1]
                    _size = int(_size)
                    _image = self.data[_key][_index]
                    # create the directory if not exist
                    _dir = f'{save_dir}/{_size}'
                    if not os.path.exists(_dir):
                        os.makedirs(_dir)
                    _dir = f'{_dir}/{_category}'
                    if not os.path.exists(_dir):
                        os.makedirs(_dir)
                    _dir = f'{_dir}/{_name}.png'
                    plt.imsave(_dir, _image)

        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_npy_dir', type = str, default = 'images.npy', help = 'the directory of the image npy file')
    parser.add_argument('--types_npy_dir', type = str, default = 'types.npy', help = 'the directory of the types npy file')
    parser.add_argument('--masks_npy_dir', type = str, default = 'masks.npy', help = 'the directory of the masks npy file')
    parser.add_argument('--image_folder', type = str, default = 'images', help = 'the directory of the image folder')
    args = parser.parse_args()

    extract_annotation = extract_annotation(args.image_npy_dir, args.types_npy_dir, args.masks_npy_dir)
    extract_annotation.extract()
    extract_annotation.save(save_dir = args.image_folder)   

    # save the data

    

    # how to ran this file
    # example 
    # nohup python utils/extract_cells.py --image_npy_dir ../Dataset/pannuke/Fold_1/images/fold1/images.npy --types_npy_dir ../Dataset/pannuke/Fold_1/images/fold1/types.npy --masks_npy_dir ../Dataset/pannuke/Fold_1/masks/fold1/masks.npy --image_folder ./Data/folder_1 &
    # python utils/extract_cells.py --image_npy_dir ../Dataset/pannuke/Fold_2/images/fold2/images.npy --types_npy_dir ../Dataset/pannuke/Fold_2/images/fold2/types.npy --masks_npy_dir ../Dataset/pannuke/Fold_2/masks/fold2/masks.npy --image_folder ./Data/folder_2
    # nohup python utils/extract_cells.py --image_npy_dir ../Dataset/pannuke/Fold_3/images/fold3/images.npy --types_npy_dir ../Dataset/pannuke/Fold_3/images/fold3/types.npy --masks_npy_dir ../Dataset/pannuke/Fold_3/masks/fold3/masks.npy --image_folder ./Data/folder_3 &

    
            
            
    
            




        


