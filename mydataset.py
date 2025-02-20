import os

import numpy as np
import pandas as pd
import torch
import nibabel as nib
from monai.transforms import Compose, RandRotate, RandZoom, RandAffine, NormalizeIntensity, ToTensor
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset

from utils.utils import get_subjects_labels


Transforms1 = Compose([
    RandRotate(range_x=8.0, range_y=8.0, range_z=8.0),
    RandZoom(min_zoom=0.8, max_zoom=1.2),
    RandAffine(translate_range=(10, 10, 10), prob=1.0),

    NormalizeIntensity(nonzero=True),
    ToTensor()
])

Transforms2 = Compose([
    NormalizeIntensity(nonzero=True),
    ToTensor()
])


def load_mri(mri_image_path):
    """
    load mri image NIfTI:(h, w, d) <class 'numpy.ndarray'>
    """
    try:
        mri_img = nib.load(mri_image_path).get_fdata()
    except Exception as e:
        raise RuntimeError(f"Error loading NIfTI file: {mri_image_path}, {e}")

    # add Channel: (1, 181, 217, 181) <class 'numpy.ndarray'>
    mri_img = np.expand_dims(mri_img, 0)

    return mri_img


def filter_images_by_subjects(img_name_list, subject_list):
    """
    Filter image file names by subject list.
    """
    filtered_list = []
    for file_name in img_name_list:
        subject_name = '_'.join(file_name.split('_')[:3])  # file_name: 002_S_2043_uMCI.pt
        if subject_name in subject_list:
            filtered_list.append(file_name)
    return filtered_list


def find_label_2(img_name, class_names):
    """
    CN_AD or sMCI_pMCI
    """
    group = img_name.split('_')[3]
    group = group.split('.')[0]

    if group == class_names[0]:
        label = 0
    elif group == class_names[1]:
        label = 1
    else:
        raise ValueError("Unknown group value: {}".format(group))

    label = torch.tensor(label, dtype=torch.long)
    return label


class MyDataSetMri(Dataset):
    def __init__(self, mri_dir_path, img_name_list, subject_list, transform=None, class_names=None):
        """
        Args:
            mri_dir_path (string): MRI_data or PET_data: .pt
            img_name_list (list):  MRI PET clinical_data  are same
            subject_list: subject list
            transform: transform
            class_names (list): e.g. [pMCI,sMCI] or [CN,AD]
        """
        self.mri_dir_path = mri_dir_path
        self.transform = transform
        self.class_names = class_names

        self.img_name_list = filter_images_by_subjects(img_name_list, subject_list)

        self.mri_paths = [os.path.join(mri_dir_path, name) for name in self.img_name_list]

        self.labels = []
        for img_name in self.img_name_list:
            self.labels.append(find_label_2(img_name, self.class_names))

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        mri_img = torch.load(self.mri_paths[idx], weights_only=True)

        label = self.labels[idx]

        if self.transform is not None:
            mri_img = self.transform(mri_img)

        return mri_img, label


class MyDataSetMriPet(Dataset):
    def __init__(self, mri_dir_path, pet_dir_path, img_name_list, subject_list, transform=None,
                 class_names=None):
        """
        Args:
            mri_dir_path (string): MRI_data: .pt
            pet_dir_path (string): PET_data: .pt
            img_name_list (list):  MRI PET clinical_data  are same
            subject_list: subject list
            transform: transform
            class_names (list): e.g. [pMCI,sMCI] or [CN,AD]
        """
        self.mri_dir_path = mri_dir_path
        self.pet_dir_path = pet_dir_path
        self.transform = transform
        self.class_names = class_names

        self.img_name_list = filter_images_by_subjects(img_name_list, subject_list)

        self.mri_paths = [os.path.join(mri_dir_path, name) for name in self.img_name_list]
        self.pet_paths = [os.path.join(pet_dir_path, name) for name in self.img_name_list]

        self.labels = []
        for img_name in self.img_name_list:
            self.labels.append(find_label_2(img_name, self.class_names))

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        mri_img = torch.load(self.mri_paths[idx], weights_only=True)
        pet_img = torch.load(self.pet_paths[idx], weights_only=True)

        label = self.labels[idx]

        if self.transform is not None:
            mri_img = self.transform(mri_img)
            pet_img = self.transform(pet_img)

        return mri_img, pet_img, label


class MyDataSetMriPetClinical(Dataset):
    def __init__(self, mri_dir_path, pet_dir_path, clinical_dir_path, img_name_list, subject_list, transform=None,
                 class_names=None):
        """
        Args:
            mri_dir_path (string): MRI_data: .pt
            pet_dir_path (string): PET_data: .pt
            clinical_dir_path (string): clinical_data: .pt
            img_name_list (list):  MRI PET clinical_data  are same
            subject_list: subject list
            transform: transform
            class_names (list): e.g. [CN AD] or [pMCI,sMCI]
        """
        self.mri_dir_path = mri_dir_path
        self.pet_dir_path = pet_dir_path
        self.clinical_dir_path = clinical_dir_path
        self.transform = transform
        self.class_names = class_names

        self.img_name_list = filter_images_by_subjects(img_name_list, subject_list)

        self.mri_paths = [os.path.join(mri_dir_path, name) for name in self.img_name_list]
        self.pet_paths = [os.path.join(pet_dir_path, name) for name in self.img_name_list]
        self.clinical_paths = [os.path.join(clinical_dir_path, name) for name in self.img_name_list]

        self.labels = []
        for img_name in self.img_name_list:
            self.labels.append(find_label_2(img_name, self.class_names))

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        mri_img = torch.load(self.mri_paths[idx], weights_only=True)
        pet_img = torch.load(self.pet_paths[idx], weights_only=True)
        clinical_features = torch.load(self.clinical_paths[idx], weights_only=True)

        label = self.labels[idx]

        if self.transform is not None:
            mri_img = self.transform(mri_img)
            pet_img = self.transform(pet_img)

        return mri_img, pet_img, clinical_features, label


if __name__ == '__main__':
    torch.set_printoptions(profile="full")
    class_name = ['CN','AD']
    mri_path = r"F:\ADNI_PROCESSED_version1\ADNI_MRI_T1_LINEAR"
    pet_path = r"F:\ADNI_PROCESSED_version1\ADNI_PET_T1_LINEAR_SMOOTH"
    clinical_path = r"E:\中间处理过程\clinical_pt"

    mri_img_name_list = os.listdir(pet_path)

    subject_list_file = 'Data/Group_Subject_MRI_PET.csv'
    df = pd.read_csv(subject_list_file)

    # k-fold
    selected_columns = [col for col in df.columns if col in class_name]
    subjects, labels = get_subjects_labels(df, selected_columns)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_index, test_index) in enumerate(skf.split(subjects, labels)):
        dataset = MyDataSetMriPetClinical(
            mri_dir_path=mri_path,
            pet_dir_path=pet_path,
            clinical_dir_path=clinical_path,
            img_name_list=mri_img_name_list,
            subject_list=subjects,
            transform=Transforms1,
            class_names=class_name
        )
        for i in range(669):
            mris, pets, clinicals, labels = dataset[i]
            # print(f"\nSample {i + 1}:")
            # print(f"Image name: {dataset.img_name_list[i]}")
            # print(f"MRI shape: {mris.shape}")
            # print(f"PET shape: {pets.shape}")
            # print(f"Clinical features: {clinicals}")
            # print(f"Label: {labels}")
            # print("-" * 50)
        print(dataset.__len__())
        break
