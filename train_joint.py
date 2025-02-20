# -*-coding:utf-8-*-
import argparse
import os
import sys
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold

import wandb
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

from model.fusion_modules import ConcatFusion, SumFusion, FiLM, GatedFusion, CrossAttention
from mydataset import MyDataSetMriPet, Transforms1, Transforms2
from model.mynet import create_vit_backbone, MriClassifier, PetClassifier

from utils.utils import get_subjects_labels, calculate_metrics, calculate_all_folds_avg_metrics, set_seed


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--experiment_name', type=str, default='Vit_Joint_concat')
    parser.add_argument('--class_names', type=str, default='CN,AD',
                        choices=['CN,AD', 'sMCI,pMCI'], help='names of the classes.')

    parser.add_argument('--use_wandb', type=bool, default=False,
                        help='if use wandb to log')
    parser.add_argument('--backbone', type=str, default='vit',
                        choices=['vit'], help='names of the backbone.')
    parser.add_argument('--test_only', type=bool, default=False)

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use for computation, e.g., "cpu", "cuda:0"')

    parser.add_argument('--optim_type', type=str, default='SGD')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--epochs', type=int, default=40)

    parser.add_argument('--fusion_method', type=str, default='concat',
                        choices=['concat', 'sum', 'film', 'gated', 'CrossAttention'],
                        help='the type of fusion_method')

    return parser.parse_args()


def test_only(args, test_dataloader, model_mri, model_pet, classifier, device, fold, metrics_dict):

    # load model
    dir_path = r"E:\PythonProjects\mlawc\checkpoints\CN_AD\Vit_Joint_concat-v2"

    model_mri_path = os.path.join(dir_path, f'fold-{fold}_model_mri.pth')
    model_pet_path = os.path.join(dir_path, f'fold-{fold}_model_pet.pth')
    model_head_path = os.path.join(dir_path, f'fold-{fold}_model_head_{args.fusion_method}.pth')

    model_mri.load_state_dict(torch.load(model_mri_path, weights_only=True))
    model_pet.load_state_dict(torch.load(model_pet_path, weights_only=True))
    classifier.load_state_dict(torch.load(model_head_path, weights_only=True))

    model_mri.to(device).eval()
    model_pet.to(device).eval()
    classifier.to(device).eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for data in tqdm(test_dataloader, desc="Evaluating"):
            mri_data, pet_data, labels = data
            mri_data = mri_data.to(device)
            pet_data = pet_data.to(device)
            labels = labels.to(device)

            mri_feature, _ = model_mri(mri_data)
            pet_feature, _ = model_pet(pet_data)
            _, _, out = classifier(mri_feature, pet_feature)

            _, predicted = torch.max(out, 1)
            probabilities = torch.softmax(out, dim=1)
            positive_probs = probabilities[:, 1]

            all_probs.extend(positive_probs.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    metrics = calculate_metrics(all_labels, all_preds, all_probs=all_probs)

    if args.use_wandb:
        wandb.log({
            'fold': fold
        })

    best_metrics_fold = metrics.copy()
    metrics_dict[fold] = best_metrics_fold
    return metrics_dict


def train_and_test(args, train_dataloader, test_dataloader, model_mri, model_pet, classifier, device, optimizer,
                   lr_scheduler, fold, metrics_dict):
    class_names = args.class_names.split(',')

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)


    best_metrics_fold = {
        'acc': 0.0,
        'spec': 0.0,
        'recall': 0.0,
        'f1': 0.0,
        'auc': 0.0,
        'cm': np.zeros((2, 2), dtype=int)
    }

    best_acc = 0.0
    best_auc = 0.0
    previous_model_mri_filepath = None
    previous_model_pet_filepath = None
    previous_model_head_filepath = None
    for epoch in range(args.epochs):
        model_mri.train()
        model_pet.train()
        classifier.train()

        train_loss = 0.0
        train_bar = tqdm(train_dataloader, file=sys.stdout)
        for step, train_data in enumerate(train_bar):
            train_data: tuple[Any, Any, Any]
            mri_data, pet_data, labels = train_data
            mri_data = mri_data.to(device)
            pet_data = pet_data.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            mri_feature, _ = model_mri(mri_data)
            pet_feature, _ = model_pet(pet_data)
            _, _, out = classifier(mri_feature, pet_feature)

            loss = criterion(out, labels)
            loss.backward()

            optimizer.step()

            train_loss += loss.item()
            train_bar.desc = "fold {}: train epoch[{}/{}] loss:{:.3f}".format(fold, epoch + 1, args.epochs, loss)

        lr_scheduler.step()
        print('fold {}: Current Learning Rate: {}'.format(fold, lr_scheduler.get_last_lr()))
        ####################################################################################################################
        # test
        model_mri.eval()
        model_pet.eval()
        classifier.eval()

        all_preds = []
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for test_data in test_dataloader:
                test_data: tuple[Any, Any, Any]
                mri_data, pet_data, labels = test_data

                mri_data = mri_data.to(device)
                pet_data = pet_data.to(device)
                labels = labels.to(device)

                mri_feature, _ = model_mri(mri_data)
                pet_feature, _ = model_pet(pet_data)
                _, _, out = classifier(mri_feature, pet_feature)

                _, preds = torch.max(out, dim=1)
                probabilities = torch.softmax(out, dim=1)
                positive_probs = probabilities[:, 1]

                all_probs.extend(positive_probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

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
            if auc_is_best:
                best_auc = metrics['auc']
            best_metrics_fold = metrics.copy()

            save_dir = os.path.join('checkpoints', f'{class_names[0]}_{class_names[1]}', f'{args.experiment_name}')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            model_mri_filename = f'fold-{fold}_model_mri.pth'
            model_pet_filename = f'fold-{fold}_model_pet.pth'
            model_classifier_filename = f'fold-{fold}_model_head_{args.fusion_method}.pth'

            model_mri_filepath = os.path.join(save_dir, model_mri_filename)
            model_pet_filepath = os.path.join(save_dir, model_pet_filename)
            model_classifier_filepath = os.path.join(save_dir, model_classifier_filename)

            if previous_model_mri_filepath is not None and os.path.exists(previous_model_mri_filepath):
                os.remove(previous_model_mri_filepath)
                print(f"Deleted previous mri model file: {previous_model_mri_filepath}")

            if previous_model_pet_filepath is not None and os.path.exists(previous_model_pet_filepath):
                os.remove(previous_model_pet_filepath)
                print(f"Deleted previous pet model file: {previous_model_pet_filepath}")

            if previous_model_head_filepath is not None and os.path.exists(previous_model_head_filepath):
                os.remove(previous_model_head_filepath)
                print(f"Deleted previous head model file: {previous_model_head_filepath}")

            torch.save(model_mri.state_dict(), model_mri_filepath)
            print(f"Saved new best model to: {model_mri_filepath}")

            torch.save(model_pet.state_dict(), model_pet_filepath)
            print(f"Saved new best model to: {model_pet_filepath}")

            torch.save(classifier.state_dict(), model_classifier_filepath)
            print(f"Saved new best model to: {model_classifier_filepath}")

            previous_model_mri_filepath = model_mri_filepath
            previous_model_pet_filepath = model_pet_filepath
            previous_model_head_filepath = model_classifier_filepath
        print('-------------------------------------------------------------------------------------------------------')

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
        name = f'{args.experiment_name}_{class_names[0]}-{class_names[1]}'
        wandb.init(project='UniCross', config=config_dict, mode="online", save_code=True, name=name)

    # Folder path
    if os.name == 'nt':  # Windows
        mri_path = r"E:\中间处理过程\transform_final\mri_crop_pt"
        pet_path = r"E:\中间处理过程\transform_final\pet_crop_pt"
    elif os.name == 'posix':  # Linux
        mri_path = '/root/autodl-tmp/pet_crop_pt'
        pet_path = '/root/autodl-tmp/mri_crop_pt'
    else:
        raise ValueError("Unsupported operating system!")

    mri_img_name_list = os.listdir(mri_path)
    subject_list_file = 'Data/Group_Subject_MRI_PET.csv'
    df = pd.read_csv(subject_list_file)

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
            mri_backbone = create_vit_backbone(pretrained=not args.test_only)
            pet_backbone = create_vit_backbone(pretrained=not args.test_only)

            model_mri = MriClassifier(mri_backbone, out_feature_dim=768, class_num=class_num).to(device)
            model_pet = PetClassifier(pet_backbone, out_feature_dim=768, class_num=class_num).to(device)
            out_feature_dim = 768

        else:
            raise ValueError("Unsupported backbone!")

        # fusion method
        if args.fusion_method == 'sum':
            classifier = SumFusion(input_dim=out_feature_dim, output_dim=class_num)
        elif args.fusion_method == 'concat':
            classifier = ConcatFusion(input_dim=out_feature_dim, output_dim=class_num)
        elif args.fusion_method == 'film':
            classifier = FiLM(input_dim=out_feature_dim, output_dim=class_num)
        elif args.fusion_method == 'gated':
            classifier = GatedFusion(input_dim=out_feature_dim, output_dim=class_num)
        elif args.fusion_method == 'CrossAttention':
            classifier = CrossAttention(input_dim=out_feature_dim, output_dim=class_num)
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(args.fusion_method))

        classifier.to(device)

        parameters = list(model_mri.net.parameters()) + list(model_pet.net.parameters()) + list(classifier.parameters())
        optimizer = getattr(optim, args.optim_type)(parameters, lr=args.lr, weight_decay=args.weight_decay,
                                                    momentum=args.momentum)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=3, eta_min=0.00001)

        ################################################################################################################
        train_subjects = subjects[train_index]
        test_subjects = subjects[test_index]

        train_dataset = MyDataSetMriPet(
            mri_dir_path=mri_path,
            pet_dir_path=pet_path,
            img_name_list=mri_img_name_list,
            subject_list=train_subjects,
            transform=Transforms1,
            class_names=class_names
        )

        test_dataset = MyDataSetMriPet(
            mri_dir_path=mri_path,
            pet_dir_path=pet_path,
            img_name_list=mri_img_name_list,
            subject_list=test_subjects,
            transform=Transforms2,
            class_names=class_names
        )

        train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                                      shuffle=True, num_workers=nw, pin_memory=True, drop_last=True)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size,
                                     shuffle=False, num_workers=nw, pin_memory=True, drop_last=False)

        print("fold {}: {} subjects for training, {} subjects for test.".format(fold, len(train_subjects),
                                                                                len(test_subjects)))

        # train and test
        if args.test_only:
            all_folds_metrics = test_only(args, test_dataloader, model_mri, model_pet, classifier, device, fold,
                                          all_folds_metrics)
        else:
            all_folds_metrics = train_and_test(args, train_dataloader, test_dataloader, model_mri, model_pet,
                                               classifier, device, optimizer, lr_scheduler, fold, all_folds_metrics)

    # calculate all folds avg metrics
    calculate_all_folds_avg_metrics(args, all_folds_metrics)

    # save code
    if args.use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
