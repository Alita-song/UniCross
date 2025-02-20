import os
import nibabel as nib
import numpy as np
import torch
from monai.transforms import (
    Compose,
    Resize,
    NormalizeIntensity,
    ToTensor,
    CropForeground,
    SpatialPad,
)
from tqdm import tqdm


def get_transforms(img):
    cropper = CropForeground(source_key=None)
    cropped_img = cropper(img)
    max_length = max(cropped_img.shape[1:])

    transforms = Compose([
        CropForeground(source_key=None),
        SpatialPad(
            spatial_size=(max_length, max_length, max_length),
            mode="constant",
            value=0
        ),
        Resize(
            spatial_size=(128, 128, 128),
            mode="trilinear",
            align_corners=True
        ),
        # NormalizeIntensity(nonzero=True)
    ])

    return cropped_img, transforms


def convert_nii_to_pt(input_dir, output_dir_pt, output_dir_nii):

    os.makedirs(output_dir_pt, exist_ok=True)
    os.makedirs(output_dir_nii, exist_ok=True)

    nii_files = [f for f in os.listdir(input_dir) if f.endswith('.nii.gz')]
    t = ToTensor()

    for file in tqdm(nii_files, desc="Converting files"):
        try:
            input_path = os.path.join(input_dir, file)
            output_path_pt = os.path.join(output_dir_pt, file.replace('.nii.gz', '.pt'))
            output_path_nii = os.path.join(output_dir_nii, file)

            img = nib.load(input_path).get_fdata()
            img = np.expand_dims(img, 0)  # ndarray: (1, 181, 217, 181)

            img, transforms = get_transforms(img)
            transformed_img = transforms(img)

            # save .pt files
            img_tensor = t(transformed_img).as_tensor()
            torch.save(img_tensor.float(), output_path_pt)

            # save .nii.gz files
            transformed_img = np.squeeze(transformed_img)
            nii_img = nib.Nifti1Image(transformed_img, np.eye(4))
            nib.save(nii_img, output_path_nii)

        except Exception as e:
            print(f"Error processing {file}: {str(e)}")


if __name__ == '__main__':
    # Mri
    # mri_input_dir = r"E:\中间处理过程\mri_original"
    # mri_output_dir_pt = r"E:\中间处理过程\transform_final\mri_crop_pt"
    # mri_output_dir_nii = r"E:\中间处理过程\transform_final\mri_crop"
    #
    # print("Converting MRI files...")
    # convert_nii_to_pt(mri_input_dir, mri_output_dir_pt, mri_output_dir_nii)

    # Pet
    pet_input_dir = r"E:\中间处理过程\pet_original"
    pet_output_dir_pt = r"E:\中间处理过程\transform_final\pet_crop_pt"
    pet_output_dir_nii = r"E:\中间处理过程\transform_final\pet_crop"

    print("Converting PET files...")
    convert_nii_to_pt(pet_input_dir, pet_output_dir_pt, pet_output_dir_nii)

    print("Conversion completed!")