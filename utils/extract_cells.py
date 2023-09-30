import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colors
import argparse

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

        # print image shape
        print(f"the shape of the image is {self.image.shape}")

        for _index in tqdm(range(self.image.shape[0])):
            _image = self.image[_index]
            _mask = self.mask[_index]

                    # create a color map for the mask
        

        

            # convert the npy image to uint8
            _image = _image.astype(np.uint8)
            
            # convert (256,256,3) to (256,256)
            _mask = np.squeeze(_mask)
            

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
                else:
                    pass
                    #print("skip this cell")
        return self.data
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_npy_dir', type = str, default = 'images.npy', help = 'the directory of the image npy file')
    parser.add_argument('--types_npy_dir', type = str, default = 'types.npy', help = 'the directory of the types npy file')
    parser.add_argument('--masks_npy_dir', type = str, default = 'masks.npy', help = 'the directory of the masks npy file')
    args = parser.parse_args()

    extract_annotation = extract_annotation(args.image_npy_dir, args.types_npy_dir, args.masks_npy_dir)
    data = extract_annotation.extract()
    

    print('How many instances are there in total?')
    print(len(data['name']))

    print('How many instances are there in each size?')
    for key in data.keys():
        if 'size' in key:
            print(key, len(data[key]))

    
            
            
    
            




        

