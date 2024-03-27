import itk
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import gray2rgb


def show_itkimages(images, titles=None, figsize=6, imfuse=None):
    """Plot a 2-D ITK image object using matplotlib.
    
        images: list with ITK images
        figsize: define size of figure
        imfuse: list containing indexes of images you want to fuse, e.g. [0, 1]
    """
    # Create figure with subplots
    n_images = len(images)
    if imfuse is not None:
        figsize = (figsize*(n_images + 1), figsize)
    else:
        figsize = (figsize*n_images, figsize)
                   
    fig = plt.figure(figsize=figsize)
    
    for nim, image in enumerate(images):
        # Create a subplot
        if imfuse is not None:
            ax = fig.add_subplot(1, n_images + 1, nim + 1)
        else:
            ax = fig.add_subplot(1, n_images, nim + 1)
        ax.axis('off')
        
        # Convert ITK image to array
        image_array = itk.GetArrayFromImage(image)
    
        # If you plot a binary image, we will set the max of the plot to 1 and the min to 0
        if int(np.max(image)) == 1:
            vmax = np.max(image)
        else:
            vmax = None
 
        if int(np.min(image)) == 0:
            vmin = np.min(image)
        else:
            vmin = None
            
        # Plot the actual image
        ax.imshow(image_array, cmap=plt.cm.gray, interpolation='nearest', vmin=vmin, vmax=vmax)
        
        if titles is not None:
            ax.set_title(titles[nim])
                   
    if imfuse is not None:
        # Display fusion of images in last plot
        ax = fig.add_subplot(1, n_images + 1, n_images + 1)
                   
        # Create a color image: specific channel distribution creates magenta - green images
        im1 = images[imfuse[0]]
        im2 = images[imfuse[1]]
        color_img = np.zeros(gray2rgb(im1).shape)
        color_img[:,:,0] =  abs(np.asarray(im1)/255)
        color_img[:,:,1] =  abs(np.asarray(im2)/255)
        color_img[:,:,2] =  abs(np.asarray(im1)/255)
        
        # Display fusion image
        ax.imshow(color_img)    
        ax.set_title(f'Fused {imfuse}.')
        ax.axis('off')
       
        
def image_generator(x1, x2, y1, y2, mask=False, artefact=False):
    """Generate an image with a square and an edge artefact."""
    # Determine type
    if mask:
        image = np.zeros([100, 100], np.uint8)
    else:
        image = np.zeros([100, 100], np.float32)
        
    # Create square
    image[y1:y2, x1:x2] = 1
    
    # Create edge artefact
    if artefact:
        image[-10:, :] = 1
    
    # convert array to itk image
    image = itk.image_view_from_array(image)
    return image
