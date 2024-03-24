# -*- coding: utf-8 -*-

"""
Advanced Image Processing Exercises 2021
Week 3: Machine learning and Pattern Recognition
Additional functions

@author: Martijn Starmans

"""

import os
import numpy
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
from natsort import natsorted
from sklearn import datasets as ds
from sklearn.decomposition import PCA
import pandas as pd
import seaborn


class IndexTracker(object):
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = self.slices//2

        self.im = ax.imshow(self.X[:, :, self.ind])

        self.lstSeeds = []
        self.info = []
        self.update()

    def onscroll(self, event):
        # print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = numpy.clip(self.ind + 1, 0, self.slices - 1)
        else:
            self.ind = numpy.clip(self.ind - 1, 0, self.slices - 1)

        self.update()

    def onclick(self, event):
        print('bla bla')
        ix, iy = event.xdata, event.ydata
        self.lstSeeds.append((int(ix), int(iy), self.ind))

        self.X[int(iy), int(ix), self.ind] = 10000

        print(self.lstSeeds)
        self.update()

    def infclick(self, event):
        # Get intensity of the pixel you are looking at, plus location
        ix, iy = event.xdata, event.ydata
        ix = int(ix)
        iy = int(iy)
        intensity = self.X[ix, iy, self.ind]
        self.info = [ix, iy, intensity]
        print('X,Y, Int: ' + str(self.info))
        self.update()


    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()

    def get_seeds(self):
        return np.asarray(self.lstSeeds)

    def get_info(self):
        return np.asarray(self.info)

def sitk_show(img, title=None, margin=0.0, dpi=40):
    nda = sitk.GetArrayFromImage(img)
    nda = np.transpose(nda, [2, 1, 0])
    #spacing = img.GetSpacing()
    figsize = (1 + margin) * nda.shape[0] / dpi, (1 + margin) * nda.shape[1] / dpi
    #extent = (0, nda.shape[1]*spacing[1], nda.shape[0]*spacing[0], 0)
    extent = (0, nda.shape[1], nda.shape[0], 0)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])

    plt.set_cmap("gray")
    # myobj = ax.imshow(nda,extent=extent,interpolation=None)

    if title:
        plt.title(title)

    tracker = IndexTracker(ax, nda)
    # fig.canvas.mpl_connect('button_press_event', lambda event: onclick(event, myobj, fig))
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    raise IOError('Error raised in order to maintain scrolling event.')


def sitk_show_select_points(img, title=None, margin=0.0, dpi=40):
    nda = sitk.GetArrayFromImage(img)
    nda = np.transpose(nda, [2, 1, 0])
    figsize = (1 + margin) * nda.shape[0] / dpi, (1 + margin) * nda.shape[1] / dpi
    extent = (0, nda.shape[1], nda.shape[0], 0)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])

    plt.set_cmap("gray")

    if title:
        plt.title(title)

    tracker = IndexTracker(ax, nda)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    fig.canvas.mpl_connect('button_press_event', tracker.onclick)

    plt.show()

    seeds = tracker.get_seeds()
    seeds[:, [0, 1, 2]] = seeds[:, [1, 0, 2]]
    return seeds


def sitk_tile_vec(lstImgs):
    lstImgToCompose = []
    for idxComp in range(lstImgs[0].GetNumberOfComponentsPerPixel()):
        lstImgToTile = []
        for img in lstImgs:
            lstImgToTile.append(sitk.VectorIndexSelectionCast(img, idxComp))
        lstImgToCompose.append(sitk.Tile(lstImgToTile, (len(lstImgs), 1, 0)))
    sitk_show(sitk.Compose(lstImgToCompose))


def slicer(image, slices=None, mask=None):
    '''
    Make mosaic of nine slices of a 3d image.
    '''
    shape = image.shape
    fig = plt.figure(figsize=(9, 3))
    n_axial = shape[2] - 1  # -1 as indexing starts at 0

    if slices is None:
        if mask is None:
            # Take the 1/4, 2/4 and 3.4 slices
            slices = [int(n_axial/4), int(n_axial/4*2), int(n_axial/4*4)]
        else:
            # Take the 1/4, 2/4 and 3.4 slices of the mask
            mask_slices = np.argwhere(np.any(mask, axis=(0, 1)))
            n_axial = len(mask_slices) - 1
            slices = [mask_slices[int(n_axial/4)],
                      mask_slices[int(n_axial/4*2)],
                      mask_slices[int(n_axial/4*3)]]

    # Get min and max of images for plotting
    vmin = np.min(image)
    vmax = np.max(image)

    # Plot the axial slices
    nslices = len(slices)
    for i_slice in range(0, nslices):
        ax = fig.add_subplot(1, nslices, i_slice + 1)
        ax.imshow(np.squeeze(image[:, :, slices[i_slice]]), cmap=plt.cm.gray,
                  interpolation='nearest', vmin=vmin, vmax=vmax)
        ax.set_title(f'Slice {slices[i_slice]}.')

    plt.show()


def get_masked_slices(image_array, mask_array):
    '''
    Get only those slices on which there is a segmentation.
    '''
    mask_array = mask_array.astype(np.bool)

    mask_slices = np.any(mask_array, axis=(0, 1))
    try:
        image_array = image_array[:, :, mask_slices]
        mask_array = mask_array[:, :, mask_slices]
    except IndexError:
        print("Note: Mask indexing does not match image!")
        mask_slices = mask_slices[0:image_array.shape[2]]
        image_array = image_array[:, :, mask_slices]
        mask_array = mask_array[:, :, mask_slices]

    return image_array, mask_array


def get_masked_voxels(image_array, mask_array):
    '''
    Flattens the image and returns a vector containing only the voxel values
    present in the mask.
    '''
    mask_array = mask_array.astype(np.bool)

    mask_array = mask_array.flatten()
    image_array = image_array.flatten()

    masked_voxels = image_array[mask_array]

    return masked_voxels


def GetArrayFromImageC(image):
    image = sitk.GetArrayFromImage(image)
    image = np.transpose(image, [2, 1, 0])
    return image


def boxplot(image_features, patient_labels, patient_IDs=None, figsize=(15,15)):
    if patient_IDs is None:
        # Use numbers as ids
        patient_IDs = [str(num) for num in range(0, len(patient_labels))]

    # Make a feature value and label vector for each class per feature label
    labels = image_features[0].keys()
    featvect = dict()
    flab = dict()
    for l in labels:
        featvect[l] = {"all": [], "1": [], "0": []}
        flab[l] = {"all": [], "1": [], "0": []}

    # Stack per feature type and class
    print("Stacking features.")
    for imfeat, label, pid in zip(image_features, patient_labels, patient_IDs):
        for fl in labels:
            featvect[fl]['all'].append(imfeat[fl])
            flab[fl]['all'].append(pid)
            if label == 0:
                featvect[fl]['0'].append(imfeat[fl])
                flab[fl]['0'].append(pid)
            else:
                featvect[fl]['1'].append(imfeat[fl])
                flab[fl]['1'].append(pid)

    # Create the boxplots
    print("Generating boxplots.")
    # Split in 5x5 figures.
    nfig = np.ceil(len(labels) / 25.0)

    labels = sorted(labels)
    for fi in range(0, int(nfig)):
        plt.figure(figsize=figsize)
        fignum = 1
        for i in range(fi*25, min((fi+1)*25, len(labels))):
            ax = plt.subplot(5, 5, fignum)
            lab = labels[i]
            plt.subplots_adjust(hspace=0.3, wspace=0.2)
            ax.scatter(np.ones(len(featvect[lab]['all'])),
                       featvect[lab]['all'],
                       color='blue')
            ax.scatter(np.ones(len(featvect[lab]['1']))*2.0,
                       featvect[lab]['1'],
                       color='red')
            ax.scatter(np.ones(len(featvect[lab]['0']))*3.0,
                       featvect[lab]['0'],
                       color='green')

            plt.boxplot([featvect[lab]['all'], featvect[lab]['1'], featvect[lab]['0']])

            fz = 5  # Works best after saving
            ax.set_title(lab, fontsize=fz)
            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(fz)

            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(fz)

            fignum += 1
    plt.show()


def create_images(size=128):
    '''
    Creates three iamges: a circle, a square and a triangle. The images
    will have the shape size x size.
    '''
    ra = range(-int(size/2), int(size/2)+1)
    x, y = np.meshgrid(ra, ra)

    # Create a circle
    radius = np.sqrt(x**2 + y**2)
    circle = radius.astype(int) == int(size/4)

    # Create a square
    square = np.zeros(circle.shape)
    square[int(size/4):int(size/4*3), int(size/4)] = 1
    square[int(size/4):int(size/4*3), int(size/4*3)] = 1
    square[int(size/4), int(size/4):int(size/4*3)] = 1
    square[int(size/4*3), int(size/4):int(size/4*3)] = 1

    # Create a triangle
    triangle = np.zeros(circle.shape)
    triangle[int(size/4*3), int(size/4):int(size/4*3)] = 1
    yr = int(size/2)
    yl = int(size/2)
    slope = 0.5
    for x in range(int(size/4), int(size/4*3)):
        triangle[x, int(yr)] = 1
        yr += slope

        triangle[x, int(yl)] = 1
        yl -= slope

    # Convert the images to ITK objects
    circle = sitk.GetImageFromArray(circle.astype(int))
    square = sitk.GetImageFromArray(square.astype(int))
    triangle = sitk.GetImageFromArray(triangle.astype(int))

    return circle, square, triangle


def colorplot(clf, ax, x, y, h=100):
    '''
    Overlay the decision areas as colors in an axes.

    Input:
        clf: trained classifier
        ax: axis to overlay color mesh on
        x: feature on x-axis
        y: feature on y-axis
        h(optional): steps in the mesh
    '''
    # Create a meshgrid the size of the axis
    xstep = (x.max() - x.min()) / 20.0
    ystep = (y.max() - y.min()) / 20.0
    print(xstep)
    x_min, x_max = x.min() - xstep, x.max() + xstep
    y_min, y_max = y.min() - ystep, y.max() + ystep
    h = max((x_max - x_min, y_max - y_min))/h
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    elif hasattr(clf, "predict_proba"):
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    if len(Z.shape) > 1:
        Z = Z[:, 1]

    # Put the result into a color plot
    cm = plt.cm.RdBu_r
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
    del xx, yy, x_min, x_max, y_min, y_max, Z, cm


def load_breast_cancer(n_features=2):
    '''
    Load the sklearn breast data set, but reduce the number of features with PCA.
    '''
    data = ds.load_breast_cancer()
    x = data['data']
    y = data['target']

    p = PCA(n_components=n_features)
    p = p.fit(x)
    x = p.transform(x)
    return x, y


def make_friedman1(n_samples=100, n_features=1, noise=40.0):
    '''
    Use the Friedman1 generator from sklearn to create a regression example.
    Perform a PCA to reduce the number of features to one.
    '''
    x, y = ds.make_friedman1(n_samples=n_samples, n_features=5, noise=noise)

    p = PCA(n_components=n_features)
    p = p.fit(x)
    x = p.transform(x)
    return x, y


def load_boston(n_features=1):
    '''
    Load the sklearn boston data set, but reduce the number of features with PCA.
    '''
    data = ds.load_boston()
    x = data['data']
    y = data['target']

    p = PCA(n_components=n_features)
    p = p.fit(x)
    x = p.transform(x)
    return x, y


def load_diabetes(n_features=1):
    '''
    Load the sklearn bdiabetes data set, but reduce the number of features with PCA.
    '''
    data = ds.load_diabetes()
    x = data['data']
    y = data['target']

    p = PCA(n_components=n_features)
    p = p.fit(x)
    x = p.transform(x)
    return x, y


def barplot(values, labels, bar_width=0.9):
    '''
    Make a barplot of features. You need to provide both the feature
    values and the labels, each as a separate list.
    '''
    y = values
    x = np.arange(len(y))

    fig, ax = plt.subplots()
    ax.bar(x, y, width=bar_width)
    ax.set_xticks(x + (bar_width/2.0))
    ax.set_xticklabels(labels, rotation=90)
    plt.show()


def make_violin_plot(data):
    '''
    Make a violin plot of metadata
    '''
    # Convert the dict to a list
    n_scans = len(data[list(data.keys())[0]])
    f, axs = plt.subplots(1, len(data.keys()))
    pallettes = ['Blues', 'Reds', 'Purples', 'Greens']

    for k, ax, pal in zip(data.keys(), axs.flat, pallettes):
        result = list()
        for n in range(0, n_scans):
            result.append({
                'n_scan': n,
                'variable': k,
                'values': data[k][n]
            })

        result = pd.DataFrame(result)
        seaborn.violinplot(x='variable', y='values', data=result, ax=ax, palette=pal, inner='points')
        ax.set_ylabel('')
        ax.set_xlabel('')

    ax.set_xlabel('')
    plt.show()


def resample_images(image, resample_spacing=None, resample_size=None,
                    interpolator=sitk.sitkBSpline):
    """Resample an image to another spacing or size.

    Note, you need to either
    provide the resample_spacing OR resample_size argument.

    Parameters
    ----------
    image : ITK Image
        Input image.
    resample_spacing : list, optional
        Spacing to resample image to
    resample_size : list, optional
        Size to resample image to.
    interpolator: ITK object, default sitk.sitkBSpline
        Interpolator to use when resampling image. Other options are
        for example sitk.sitkNearestNeighbor or sitk.sitkLinear

    Returns
    -------
    resampled_image : ITK Image
        Output image.

    """
    if resample_spacing is None and resample_size is None:
        raise ValueError('Either provide resample_spacing OR resample_size as input!')

    if resample_spacing is not None and resample_size is not None:
        raise ValueError('Either provide resample_spacing OR resample_size as input!')

    if resample_spacing is not None:
        # Compute resample size
        size = np.asarray(image.GetSize())
        spacing = np.asarray(image.GetSpacing())

        resample_size = size * spacing / resample_spacing
        resample_size = resample_size.astype(int)
        resample_size = resample_size.tolist()

    if resample_size is not None:
        # Compute resample spacing
        original_size = image.GetSize()
        original_spacing = image.GetSpacing()

        if len(original_size) == 2:
            original_size = original_size + (1, )

        if len(original_spacing) == 2:
            original_spacing = original_spacing + (1.0, )

        resample_spacing = [original_size[0]*original_spacing[0]/resample_size[0],
                            original_size[1]*original_spacing[1]/resample_size[1],
                            original_size[2]*original_spacing[2]/resample_size[2]]

    # Actual resampling
    ResampleFilter = sitk.ResampleImageFilter()
    ResampleFilter.SetInterpolator(interpolator)
    ResampleFilter.SetOutputSpacing(resample_spacing)
    ResampleFilter.SetSize(resample_size)
    ResampleFilter.SetOutputDirection(image.GetDirection())
    ResampleFilter.SetOutputOrigin(image.GetOrigin())
    ResampleFilter.SetOutputPixelType(image.GetPixelID())
    ResampleFilter.SetTransform(sitk.Transform())
    resampled_image = ResampleFilter.Execute(image)
    return resampled_image


def find_patient_labels(objects, labelfile, label_name='1p19qDel'):
    '''
    Given a list of objects, match labels from label files to the objects.

    Parameters
    ----------
    objects : list of strings
        Contains elements of which the Patient ID from the labelfile is a
        substring.
    labelfile : string
        Filename of .csv in which labels are stored
    label_name : string, optional
        Name of label to extract. The default is '1p19qDel'.

    Returns
    -------
    labels : list
        Labels of the objects.

    '''
    # Chech if label file exists
    if not os.path.exists(labelfile):
        raise KeyError(f'File {labelfile} does not exist!')

    # Load the labels first
    data = pd.read_csv(labelfile, header=0)

    # Load and check the header
    header = data.keys()
    if header[0] != 'Patient':
        raise AssertionError('First column should be patient ID!')

    # Patient IDs are stored in the first column
    patient_ID = data['Patient'].values

    # label status is stored in all remaining columns
    labels = data.values[:, 1:]
    labels = labels.astype(np.float)

    # Initialize output objects
    objects_out = list()
    label_data = dict()
    patient_IDs = list()
    labels_out = list()

    # Now match to the objects
    for i_object, o in enumerate(objects):
        ifound = 0
        matches = list()
        for i_num, i_patient in enumerate(patient_ID):
            if i_patient in str(o):

                # Match: add the patient ID to the ID's and to the matches
                patient_IDs.append(i_patient)
                matches.append(i_patient)

                # If there are feature files given, add it to the list
                objects_out.append(objects[i_object])

                # Add the label value to the label list
                labels_out.append(labels[i_num])

                # Calculate how many matches we found for this (feature) file: should be one
                ifound += 1

        if ifound > 1:
            message = ('Multiple matches ({}) found in labeling for feature file {}.').format(str(matches), str(o))
            raise ValueError(message)

        elif ifound == 0:
            message = ('No entry found in labeling for feature file {}.').format(str(o))
            raise KeyError(message)

    label_data['patient_IDs'] = np.asarray(patient_IDs)
    label_data['label'] = np.asarray(labels_out)[:, 0]
    label_data['label_name'] = label_name

    return objects_out, label_data
