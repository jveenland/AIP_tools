import pandas as pd
import os
import glob


def list_data():
    """List data of the Low Grade Glioma dataset.

    Usage: data, labels = list_data()

    Output
        labels: dict
            Dictionary with as keys the patient ID, as values the ground truth
            of patients: 1 = 1p19q co-deletion, 0 = no co-deletion
        data: dict
            Dictionary with as keys the patient ID, as values a dict with
            all available files.

    """
    # Determine this directory
    this_directory = os.path.dirname(os.path.abspath(__file__))

    # Read the labels and convert to dictionary
    labels = pd.read_csv(os.path.join(this_directory, 'LGG_mutation_status.csv'))
    labels = {pid: label for pid, label in zip(labels['Patient'], labels['1p19qDel'])}

    # Gather all available files
    pfolders = glob.glob(os.path.join(this_directory, 'LGG-*'))
    pfolders.sort()
    data = dict()
    for pfolder in pfolders:
        PID = os.path.basename(pfolder)
        data[PID] = dict()
        data[PID]['T1_image'] = os.path.join(pfolder, 'T1.nii.gz')
        data[PID]['T2_image'] = os.path.join(pfolder, 'T2.nii.gz')
        data[PID]['Segmentation_full'] = os.path.join(pfolder, 'Full_segmentation.nii.gz')
        data[PID]['Segmentation_slices'] = os.path.join(pfolder, 'Segmentation.nii.gz')
        data[PID]['T1_metadata'] = os.path.join(pfolder, '00000_T1.dcm')
        data[PID]['T2_metadata'] = os.path.join(pfolder, '00000_T2.dcm')

    return data, labels


def list_data_registration():
    """List data for the registration exercises

    Usage: data, labels = list_data()

    Output
        labels: dict
            Dictionary with as keys the patient ID, as values the ground truth
            of patients: 1 = 1p19q co-deletion, 0 = no co-deletion
        data: dict
            Dictionary with as keys the patient ID, as values a dict with
            all available files.

    """
    # Determine this directory
    this_directory = os.path.dirname(os.path.abspath(__file__))

    # Gather all example files
    example_directory = os.path.join(this_directory, 'registration', 'elastix_example_v4.8', 'exampleinput')
    example_data = {os.path.basename(filename): filename for filename in glob.glob(os.path.join(example_directory, '*'))}

    # Gather the fundus dataset
    fundus_directory =  os.path.join(this_directory, 'registration', 'Fundus')
    fundus_data = glob.glob(os.path.join(fundus_directory, '*'))
    fundus_data = {os.path.basename(filename): filename for filename in fundus_data}

    # Gather the multimodal brain dataset
    brain_directory =  os.path.join(this_directory, 'registration', 'Brain')
    brain_data = dict()
    brain_data['CT'] = os.path.join(brain_directory, 'CT.nii.gz')
    brain_data['MR_T1'] = os.path.join(brain_directory, 'MR_T1.nii.gz')

    # Combine all data
    data = dict()
    data['example_data'] = example_data
    data['fundus_data'] = fundus_data
    data['brain_data'] = brain_data

    return data

def test():
    """Test the other functions"""
    # data, labels = list_data()
    # print(data, labels)

    data = list_data_registration()
    print(data)

if __name__ == '__main__':
    test()
