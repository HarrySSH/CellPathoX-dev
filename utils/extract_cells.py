import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colors
import argparse
from skimage import 
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
                one_segmented_cells = np.where(segmented_cells_by_instance == _instance)

                # catogory of the cell
                category = np.unique(_mask_orig[one_segmented_cells])
                
                # remove 5 because 5 is background
                category = category[category != 5]
                # use the category with the most pixels
                # besides 0 the category should have only one value
                if len(category) > 1:
                    # choose the most frequent category
                    category = np.argmax(np.bincount(category))
                else:
                    if len(category) == 0:
                        # if the category is empty, then skip this instance
                        # show the image for this instance
                        # show the image and the segmentation side by side
                        # create a figure
                        fig, ax = plt.subplots(1,2, figsize = (20,10))
                        # show the image
                        ax[0].imshow(_image)
                        # show the binary map using one_segmented_cells
                        cell_mask = np.zeros(cell_mask.shape)
                        cell_mask[one_segmented_cells] = 1

                        ax[1].imshow(cell_mask, cmap = 'gray')
                        plt.show()
                        # raise error
                        raise ValueError("the category is empty")
                    else:
                        category = category[0]
                
                

                _center = np.mean(one_segmented_cells, axis = 1)
                _center = np.round(_center)
                _center = tuple(_center)

                ## make them int
                _center = (int(_center[0]), int(_center[1]))
                # check if we are able to extract the cell with the max radius
    
                max_radius = max(radius)

                
                if _center[0] - max_radius >= 0 and _center[0] + max_radius < _image.shape[0] and _center[1] - max_radius >= 0 and _center[1] + max_radius < _image.shape[0]:
                    

                    cropped_images_list = self.extract_cells(_image, _center, radius)
                    self.data['name'].append(str(_index) + '_' + str(_instance) +'_'+ str(_center)[0] + '_' + str(_center)[1])
                    # add the center

                    for _num in range(len(radius)):
                        self.data[f'size_{radius[_num]*2}'].append(cropped_images_list[_num])

                        _name = str(_index) + '_' + str(_instance) +'_'+ str(_center[0])+ '_' + str(_center[1])
                        _category = category
                        _size = radius[_num]*2
                        _image = cropped_images_list[_num]

                        # create the directory if not exist


                        self.save(_size, _image, _name, _category, save_dir = save_dir)

                    self.data['category'].append(category)
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
    

    # save the data

    

    # how to ran this file
    # example 
    # python utils/extract_cells.py --image_npy_dir ../Dataset/pannuke/Fold_1/images/fold1/images.npy --types_npy_dir ../Dataset/pannuke/Fold_1/images/fold1/types.npy --masks_npy_dir ../Dataset/pannuke/Fold_1/masks/fold1/masks.npy --image_folder ./Data/folder_1 
    # python utils/extract_cells.py --image_npy_dir ../Dataset/pannuke/Fold_2/images/fold2/images.npy --types_npy_dir ../Dataset/pannuke/Fold_2/images/fold2/types.npy --masks_npy_dir ../Dataset/pannuke/Fold_2/masks/fold2/masks.npy --image_folder ./Data/folder_2
    # python utils/extract_cells.py --image_npy_dir ../Dataset/pannuke/Fold_3/images/fold3/images.npy --types_npy_dir ../Dataset/pannuke/Fold_3/images/fold3/types.npy --masks_npy_dir ../Dataset/pannuke/Fold_3/masks/fold3/masks.npy --image_folder ./Data/folder_3 

    
            
            
    
            




        


