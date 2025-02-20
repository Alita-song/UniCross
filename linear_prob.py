# -*-coding:utf-8-*-
import argparse
import os
import sys
from typing import Any

import wandb
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from model.mynet import create_vit_backbone, MriClassifier, PetClassifier
from mydataset import Transforms1, Transforms2
from mydataset import MyDataSetMri
from utils.utils import calculate_metrics, get_subjects_labels, calculate_all_folds_avg_metrics, set_seed

global model_filepath


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--experiment_name', type=str, default='Vit_Joint_sum')
    parser.add_argument('--class_names', type=str, default='CN,AD',
                        choices=['CN,AD', 'sMCI,pMCI'], help='names of the two classes.')

    parser.add_argument('--encoder_checkpoints_save_root', type=str,
                        default=r"E:\PythonProjects\mlawc\checkpoints\CN_AD\Vit_Joint_sum",
                        help='Root directory for encoder_checkpoints_save from train_stage1')

    parser.add_argument('--use_wandb', type=bool, default=False,
                        help='use wandb to log')
    parser.add_argument('--backbone', type=str, default='vit',
                        choices=['vit'], help='names of the backbone.')
    parser.add_argument('--use_pet', type=bool, default=False,
                        help='if use pet')

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--optim_type', type=str, default='Adam')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--epochs', type=int, default=15)

    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use for computation, e.g., "cpu", "cuda:0"')

    return parser.parse_args()


def train_and_test(args, train_dataloader, test_dataloader, model, device, optimizer, fold, metrics_dict):
    global model_filepath

    class_names = args.class_names.split(',')

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_metrics_fold = {
        'acc': 0.0,
        'spec': 0.0,
        'recall': 0.0,
        'f1': 0.0,
        'auc': 0.0
    }

    best_acc = 0.0
    best_auc = 0.0
    previous_model_filepath = None
    for epoch in range(args.epochs):
        ####################################################################################################################
        # train
        model.train()
        model.net.eval()

        train_loss = 0.0
        train_bar = tqdm(train_dataloader, file=sys.stdout)
        train_data: tuple[Any, Any]
        for step, train_data in enumerate(train_bar):
            mri_data, labels = train_data
            mri_data = mri_data.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            _, outputs = model(mri_data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_bar.desc = "fold {}: train epoch[{}/{}] loss:{:.3f}".format(fold, epoch + 1, args.epochs, loss)

        ################################################################################################################
        # test
        model.eval()

        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            test_data: tuple[Any, Any]
            for test_data in test_dataloader:
                mri_data, labels = test_data
                mri_data = mri_data.to(device)
                labels = labels.to(device)

                _, outputs = model(mri_data)

                _, preds = torch.max(outputs, 1)
                probabilities = torch.softmax(outputs, dim=1)
                positive_probs = probabilities[:, 1]

                all_probs.extend(positive_probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        ################################################################################################################
        train_loss /= len(train_dataloader)

        metrics = calculate_metrics(all_labels, all_preds, all_probs=all_probs)
        print(
            f"fold {fold}: Epoch {epoch + 1}/{args.epochs}: "
            f"Train Loss: {train_loss:.4f}, "
            f"Acc: {metrics['acc']:.4f}, "
            f"SPEC: {metrics['spec']:.4f}, "
            f"Recall: {metrics['recall']:.4f}, "
            f"F1: {metrics['f1']:.4f}, "
            f"AUC: {metrics['auc']:.4f}, ")

        acc_is_best = metrics['acc'] > best_acc
        auc_is_best = (metrics['acc'] == best_acc and metrics['auc'] > best_auc)
        # save best model
        if acc_is_best or auc_is_best:
            best_acc = metrics['acc']
            best_metrics_fold = metrics.copy()

            # create save dir
            save_dir = os.path.join('checkpoints', f'{class_names[0]}_{class_names[1]}', f'{args.experiment_name}')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # create save model file path
            if args.use_pet:
                model_filename = f'fold-{fold}_model_pet.pth'
            else:
                model_filename = f'fold-{fold}_model_mri.pth'
            model_filepath = os.path.join(save_dir, model_filename)

            if previous_model_filepath is not None and os.path.exists(previous_model_filepath):
                os.remove(previous_model_filepath)
                print(f"Deleted previous model file: {previous_model_filepath}")

            # save model
            torch.save(model.state_dict(), model_filepath)
            print(f"Saved new best model to: {model_filepath}")

            previous_model_filepath = model_filepath

        print(
            '--------------------------------------------------------------------------------------------------------')
    print('Finished Training and validating')

    if args.use_wandb:
        wandb.log({
            'fold': fold,
            'best_acc': best_acc,
            'best_auc': best_auc,
        })

    metrics_dict[fold] = best_metrics_fold
    return metrics_dict


def main():
    args = get_arguments()
    set_seed(args.seed)
    class_names = args.class_names.split(',')
    class_num = len(class_names)

    nw = args.num_workers  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    if 'cuda' in args.device and not torch.cuda.is_available():
        print("CUDA is not available on this machine. Switching to CPU.")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    ####################################################################################################################
    # Initialize wandb
    if args.use_wandb:
        os.environ["WANDB_API_KEY"] = "your_key"
        config_dict = vars(args)
        project_name = f'final_{class_names[0]}-{class_names[1]}_linear_evaluation'
        wandb.init(project=project_name, config=config_dict, mode="offline", save_code=True, name=args.experiment_name)

    # Folder path
    if os.name == 'nt':  # Windows
        if args.use_pet:
            mri_path = r"E:\中间处理过程\transform_final\pet_crop_pt"
        else:
            mri_path = r"E:\中间处理过程\transform_final\mri_crop_pt"

    elif os.name == 'posix':  # Linux
        if args.use_pet:
            mri_path = '/root/autodl-tmp/pet_crop_pt'
        else:
            mri_path = '/root/autodl-tmp/mri_crop_pt'
    else:
        raise ValueError("Unsupported operating system!")

    mri_img_name_list = os.listdir(mri_path)
    subject_list_file = 'Data/Group_Subject_MRI_PET.csv'
    df = pd.read_csv(subject_list_file)

    model_dir_paths = args.encoder_checkpoints_save_root

    ####################################################################################################################
    # k-fold
    all_folds_metrics = {}

    selected_columns = [col for col in df.columns if col in class_names]
    subjects, labels = get_subjects_labels(df, selected_columns)

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    for fold, (train_index, test_index) in enumerate(skf.split(subjects, labels)):

        ################################################################################################################
        # model
        if args.backbone == 'vit':
            if args.use_pet:
                pet_backbone = create_vit_backbone(pretrained=False)
                model = PetClassifier(pet_backbone, out_feature_dim=768, class_num=class_num).to(device)
                model_path = os.path.join(model_dir_paths, f'fold-{fold}_model_pet.pth')
                model.load_state_dict(torch.load(model_path, weights_only=True))
            else:
                mri_backbone = create_vit_backbone(pretrained=False)
                model = MriClassifier(mri_backbone, out_feature_dim=768, class_num=class_num).to(device)
                model_path = os.path.join(model_dir_paths, f'fold-{fold}_model_mri.pth')
                model.load_state_dict(torch.load(model_path, weights_only=True))

        else:
            raise ValueError("Unsupported backbone!")

        optimizer = getattr(optim, args.optim_type)(model.classifier.parameters(), lr=args.lr, betas=(0.9, 0.999),
                                                    eps=1e-08, weight_decay=0)
        ################################################################################################################
        train_subjects = subjects[train_index]
        test_subjects = subjects[test_index]

        train_dataset = MyDataSetMri(
            mri_dir_path=mri_path,
            img_name_list=mri_img_name_list,
            subject_list=train_subjects,
            transform=Transforms1,
            class_names=class_names,
        )
        test_dataset = MyDataSetMri(
            mri_dir_path=mri_path,
            img_name_list=mri_img_name_list,
            subject_list=test_subjects,
            transform=Transforms2,
            class_names=class_names,
        )
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                                      shuffle=True, num_workers=nw, drop_last=False)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size,
                                     shuffle=False, num_workers=nw)
        print("fold {}: {} subjects for training, {} subjects for test.".format(fold, len(train_subjects),
                                                                                len(test_subjects)))
        # train and test
        all_folds_metrics = train_and_test(args, train_dataloader, test_dataloader, model, device, optimizer,
                                           fold, all_folds_metrics)

    # calculate all folds avg metrics
    calculate_all_folds_avg_metrics(args, all_folds_metrics)

    if args.use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
