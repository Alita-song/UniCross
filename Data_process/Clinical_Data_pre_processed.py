import os

import pandas as pd
import torch
from tqdm import tqdm


def load_subject_list(group_csv_path):
    """
    从 Group_Subject_MRI_PET.csv 读取 subject 列表及其对应标签。
    返回: [(subject_id, label), ...]
    """
    df = pd.read_csv(group_csv_path)
    subject_list = []

    for label in df.columns:  # AD, CN, pMCI, sMCI
        subjects = df[label].dropna().tolist()
        for subj in subjects:
            subject_list.append((subj, label))

    return subject_list


def process_clinical_data(clinical_path, out_dir, subject_list):
    """
    处理临床数据，为每个 subject 生成 .pt 文件。

    Args:
        clinical_path: MRI_PET_clinical_original.csv 路径
        out_dir: 输出目录
        subject_list: [(subject_id, label), ...] 格式的列表
    """
    os.makedirs(out_dir, exist_ok=True)

    clinical_data = pd.read_csv(clinical_path)

    success_count = 0
    missing_subjects = []

    for subject, label in tqdm(subject_list, desc="Converting clinical data"):
        clinical_row = clinical_data[clinical_data['PTID'] == subject]

        if clinical_row.empty:
            missing_subjects.append(subject)
            continue

        # 提取特征：'AGE', 'PTEDUCAT'
        features = clinical_row[['AGE', 'PTEDUCAT']].values.flatten()

        # 性别 one-hot 编码
        gender = clinical_row['PTGENDER'].values[0]
        gender_one_hot = [1, 0] if gender == 0 else [0, 1]

        combined_features = list(features) + gender_one_hot
        feature_tensor = torch.tensor(combined_features, dtype=torch.float)

        # 输出文件名格式：{subject}_{label}.pt
        output_path = os.path.join(out_dir, f"{subject}_{label}.pt")
        torch.save(feature_tensor.float(), output_path)

        success_count += 1

    print(f"Conversion completed: {success_count} successful")
    if missing_subjects:
        print(f"Warning: {len(missing_subjects)} subjects not found in clinical data:")
        print(missing_subjects[:10])  # 只显示前 10 个


if __name__ == "__main__":
    # 获取项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 输入文件
    group_csv_path = os.path.join(project_root, "Data", "Group_Subject_MRI_PET.csv")
    clinical_csv_path = os.path.join(project_root, "Data", "MRI_PET_clinical_original.csv")

    # 输出目录
    output_dir = os.path.join(project_root, "Data", "clinical_pt")

    # 从 Group_Subject_MRI_PET.csv 加载 subject 列表
    subject_list = load_subject_list(group_csv_path)
    print(f"从 Group CSV 加载 {len(subject_list)} 个 subject")
    print(f"临床数据文件: {clinical_csv_path}")
    print(f"输出目录: {output_dir}")

    process_clinical_data(clinical_csv_path, output_dir, subject_list)
