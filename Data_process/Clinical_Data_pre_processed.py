import os

import pandas as pd
import torch
from tqdm import tqdm


def process_clinical_data(clinical_path, out_dir, file_name_list):
    os.makedirs(out_dir, exist_ok=True)

    clinical_data = pd.read_csv(clinical_path)

    success_count = 0

    for img_name in tqdm(file_name_list, desc="Converting clinical data"):
        subject = '_'.join(img_name.split('_')[:3])  # 002_S_4171_pMCI.nii.gz

        clinical_row = clinical_data[clinical_data['PTID'] == subject]
        if clinical_row.empty:
            raise ValueError(f"No clinical data found for subject {subject}")

        # 'AGE', 'PTGENDER', 'PTEDUCAT'
        features = clinical_row[
            ['AGE', 'PTEDUCAT']].values.flatten()
        gender = clinical_row['PTGENDER'].values[0]
        gender_one_hot = [1, 0] if gender == 0 else [0, 1]

        combined_features = list(features) + gender_one_hot

        feature_tensor = torch.tensor(combined_features, dtype=torch.float)

        output_path = os.path.join(out_dir, str(img_name).replace('.nii.gz', '.pt'))
        torch.save(feature_tensor.float(), output_path)

        success_count += 1

    print(f"Conversion completed: {success_count} successful")


if __name__ == "__main__":
    clinical_csv_path = r"E:\PythonProjects\Alz_2025\Data\MRI_PET_clinical_original.csv"
    output_dir = r"E:\中间处理过程\clinical_original_pt"

    mri_dir = r"E:\中间处理过程\transform_final\mri_crop"
    img_name_list = [str(f) for f in os.listdir(mri_dir) if f.endswith('.nii.gz')]
    process_clinical_data(clinical_csv_path, output_dir, img_name_list)
