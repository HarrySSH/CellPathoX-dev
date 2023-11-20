# >>> import numpy as np
#>>> images = np.load('images.npy')
# >>> images.shape
# (2656, 256, 256, 3)
# >>> labels = np.load('types.npy')
# >>> labels.shape
# (2656,)

# >>> mask = np.load('masks.npy')
# >>> mask.shape
# (2656, 256, 256, 6)

# `masks.npy` an array of 6 channel instance-wise masks
# (0: Neoplastic cells, 1: Inflammatory, 2: Connective/Soft tissue cells, 3: Dead Cells, 4: Epithelial, 6: Background)


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colors
import argparse

from scipy import ndimage as ndi
from skimage import (
    color, feature, filters, measure, morphology, segmentation, util
)

class show_annotation:
    def __init__(self, image_npy_dir, types_npy_dir, masks_npy_dir, image_ID):
        self.image_npy_dir = image_npy_dir
        self.types_npy_dir = types_npy_dir
        self.masks_npy_dir = masks_npy_dir
        self.index = image_ID

        # load data in the most efficient way, 
        # we only need one image and its corresponding label and mask
        self.image = np.load(self.image_npy_dir)[self.index]
        
        self.tissue_type = np.load(self.types_npy_dir)[self.index]
        self.mask = np.load(self.masks_npy_dir)[self.index]
        
        self.mask = np.argmax(self.mask, axis = -1)
        self.mask = np.expand_dims(self.mask, axis = -1)
        
    def show(self, legand = True):

        # create a color map for the mask
        self.cmap = colors.ListedColormap(['black', 'red', 'blue', 'green', 'yellow', 'white'])
        self.bounds = [0,1,2,3,4,5]
        self.norm = colors.BoundaryNorm(self.bounds, self.cmap.N)
        if legand:
            # create a figure
            self.fig, self.ax = plt.subplots(1,3, figsize = (20,10))

            # convert the npy image to uint8
            self.image = self.image.astype(np.uint8)
            # show the image 
            self.ax[0].imshow(self.image)
            # show the mask
            self.ax[1].imshow(self.mask, cmap = self.cmap, norm = self.norm)
            # show the label
            self.ax[1].set_title(self.tissue_type)
            # aadd five circle with the corresponding color and labels as the legand on the lat panel
            self.ax[2].add_patch(patches.Circle((0.05, 0.05), 0.05, color = 'black', label = 'Neoplastic cells'))
            self.ax[2].add_patch(patches.Circle((0.05, 0.15), 0.05, color = 'red', label = 'Inflammatory'))
            self.ax[2].add_patch(patches.Circle((0.05, 0.25), 0.05, color = 'blue', label = 'Connective/Soft tissue cells'))
            self.ax[2].add_patch(patches.Circle((0.05, 0.35), 0.05, color = 'green', label = 'Dead Cells'))

            # make it square as well, no need to show the background
            self.ax[2].add_patch(patches.Circle((0.05, 0.45), 0.05, color = 'yellow', label = 'Epithelial'))
            self.ax[2].legend(loc = 'center left', bbox_to_anchor = (0, 0.5))

            # remove the axis
            self.ax[2].axis('off')
            # add a rectangle to block the circles 
            self.ax[2].add_patch(patches.Rectangle((0,0), 0.1, 0.5, color = 'white'))
            # remove the circle as well only keep the legand
        else:
            self.fig, self.ax = plt.subplots(1,2, figsize = (20,10))
            self.image = self.image.astype(np.uint8)
            self.ax[0].imshow(self.image)
            self.ax[1].imshow(self.mask, cmap = self.cmap, norm = self.norm)
            self.ax[1].set_title(self.tissue_type)


        plt.show()
    
    def show_by_instance(self):
        ### show the image and mask by instance
        ### need to seperate the mask into different instance especially when they overlap with each other

        # create a color map for the mask
        self.cmap = colors.ListedColormap(['black', 'red', 'blue', 'green', 'yellow', 'white'])
        self.bounds = [0,1,2,3,4,5]
        self.norm = colors.BoundaryNorm(self.bounds, self.cmap.N)

        # create a figure
        self.fig, self.ax = plt.subplots(1,3, figsize = (20,10))

        # convert the npy image to uint8
        self.image = self.image.astype(np.uint8)
        mask = self.mask.copy()
        # convert (256,256,3) to (256,256)
        mask = np.squeeze(mask)
        

        # make a segmentation as binary, background is 0, foreground is 1
        mask = np.where(mask == 5, 0, 1)

        
        cells = mask.copy()
        distance = ndi.distance_transform_edt(cells)
        local_max_coords = feature.peak_local_max(distance, min_distance=7)
        local_max_mask = np.zeros(distance.shape, dtype=bool)
        local_max_mask[tuple(local_max_coords.T)] = True
        markers = measure.label(local_max_mask)

        segmented_cells = segmentation.watershed(-distance, markers, mask=cells)

    
        
        
        
        # extract all the instances out
        


        
        # show the image
        self.ax[0].imshow(self.image)

        # show the binary map
        self.ax[1].imshow(cells, cmap = 'gray')


        # show the segmented cells
        self.ax[2].imshow(color.label2rgb(segmented_cells, bg_label=0))

        plt.show()

        # show the mask





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_npy_dir', type = str, default = 'images.npy', help = 'the directory of the image npy file')
    parser.add_argument('--types_npy_dir', type = str, default = 'types.npy', help = 'the directory of the types npy file')
    parser.add_argument('--masks_npy_dir', type = str, default = 'masks.npy', help = 'the directory of the masks npy file')
    parser.add_argument('--image_ID', type = int, default = 0, help = 'the ID of the image you want to show')
    args = parser.parse_args()

    show_annotation = show_annotation(args.image_npy_dir, args.types_npy_dir, args.masks_npy_dir, args.image_ID)
    #show_annotation.show(legand=False)
    #show_annotation.show()
    show_annotation.show_by_instance()

    
