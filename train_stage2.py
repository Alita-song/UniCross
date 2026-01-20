# -*-coding:utf-8-*-
import argparse
import os
import sys
from typing import Any

from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

import wandb
import pandas as pd
from tqdm import tqdm
import numpy as np

import torch
from torch import optim, nn
from torch.utils.data import DataLoader

from model.fusion_modules import ConcatFusion, CrossAttention, SumFusion, FiLM, GatedFusion
from model.mynet import create_vit_backbone, MriClassifier, PetClassifier
from mydataset import Transforms1, Transforms2
from mydataset import MyDataSetMriPet

from utils.utils import calculate_metrics, get_subjects_labels, calculate_all_folds_avg_metrics, set_seed

global model_filepath


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--experiment_name', type=str, default='NO_Shared')
    parser.add_argument('--encoder_checkpoints_save_root', type=str,
                        default=r"E:\PythonProjects\UniCross\checkpoints\CN_AD\NO_shared",
                        help='Root directory for encoder_checkpoints_save from train_stage1')
    parser.add_argument('--class_names', type=str, default='CN,AD',
                        choices=['CN,AD', 'sMCI,pMCI'], help='names of the two classes.')

    parser.add_argument('--use_wandb', type=bool, default=False,
                        help='if use wandb to log')
    parser.add_argument('--backbone', type=str, default='vit',
                        choices=['vit'], help='names of the backbone.')
    parser.add_argument('--only_test', type=bool, default=False,
                        help='if only_test to get metrics')

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use for computation, e.g., "cpu", "cuda:0"')

    parser.add_argument('--optim_type', type=str, default='Adam')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=10)

    parser.add_argument('--fusion_method', type=str, default='concat',
                        choices=['concat', 'sum', 'film', 'gated', 'CrossAttention'],
                        help='the type of fusion_method')

    return parser.parse_args()


def only_test(args, test_dataloader, mri_encoder, pet_encoder, classifier, classifier_mri, classifier_pet, device, fold,
              metrics_dict):

    mri_encoder.eval()
    pet_encoder.eval()
    classifier.eval()
    classifier_mri.eval()
    classifier_pet.eval()

    all_preds = []
    all_probs = []

    all_preds_mri = []
    all_preds_pet = []
    all_labels = []

    with torch.no_grad():
        test_bar = tqdm(test_dataloader, file=sys.stdout)
        test_data: tuple[Any, Any, Any]
        for step, test_data in enumerate(test_bar):
            mri_data, pet_data, labels = test_data
            mri_data = mri_data.to(device)
            pet_data = pet_data.to(device)
            labels = labels.to(device)

            mri_features = mri_encoder(mri_data)
            pet_features = pet_encoder(pet_data)

            out_mri = classifier_mri(mri_features)
            out_pet = classifier_pet(pet_features)
            _, _, out = classifier(mri_features, pet_features)

            _, preds_mri = torch.max(out_mri, 1)
            _, preds_pet = torch.max(out_pet, 1)

            _, preds = torch.max(out, dim=1)
            probabilities = torch.softmax(out, dim=1)
            positive_probs = probabilities[:, 1]

            all_probs.extend(positive_probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

            all_preds_mri.extend(preds_mri.cpu().numpy())
            all_preds_pet.extend(preds_pet.cpu().numpy())

            all_labels.extend(labels.cpu().numpy())

    ################################################################################################################

    acc_mri = accuracy_score(all_labels, all_preds_mri)
    acc_pet = accuracy_score(all_labels, all_preds_pet)

    metrics = calculate_metrics(all_labels, all_preds, all_probs=all_probs)
    print(f"fold {fold}: Acc: {metrics['acc']:.4f}, "
          f"SPEC: {metrics['spec']:.4f}, "
          f"Recall: {metrics['recall']:.4f}, "
          f"AUC: {metrics['auc']:.4f}, "
          f"F1: {metrics['f1']:.4f}, "
          f"mri_only_accuracy: {acc_mri:.4f}, "
          f"pet_only_accuracy: {acc_pet:.4f}")

    print('Finished test!!!')
    print(
        '############################################################################################################')
    if args.use_wandb:
        wandb.log({
            'fold': fold
        })

    metrics_dict[fold] = metrics.copy()
    return metrics_dict


def train_and_test(args, train_dataloader, test_dataloader, mri_encoder, pet_encoder, classifier, classifier_mri,
                   classifier_pet, device, optimizer, fold, metrics_dict):
    global model_filepath

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
    acc_mri = 0.0
    acc_pet = 0.0
    previous_model_filepath = None
    for epoch in range(args.epochs):
        ################################################################################################################
        # train
        classifier.train()
        mri_encoder.eval()
        pet_encoder.eval()

        train_loss = 0.0
        train_bar = tqdm(train_dataloader, file=sys.stdout)
        train_data: tuple[Any, Any, Any]
        for step, train_data in enumerate(train_bar):
            mri_data, pet_data, labels = train_data
            mri_data = mri_data.to(device)
            pet_data = pet_data.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            with torch.no_grad():
                mri_features = mri_encoder(mri_data)
                pet_features = pet_encoder(pet_data)
            _, _, out = classifier(mri_features, pet_features)

            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_bar.desc = "fold {}: train epoch[{}/{}] loss:{:.3f}".format(fold, epoch + 1, args.epochs, loss)

        ################################################################################################################
        # val
        mri_encoder.eval()
        pet_encoder.eval()
        classifier.eval()
        classifier_mri.eval()
        classifier_pet.eval()

        all_preds = []
        all_probs = []

        all_preds_mri = []
        all_preds_pet = []
        all_labels = []

        with torch.no_grad():
            test_bar = tqdm(test_dataloader, file=sys.stdout)
            test_data: tuple[Any, Any, Any]
            for step, test_data in enumerate(test_bar):
                mri_data, pet_data, labels = test_data
                mri_data = mri_data.to(device)
                pet_data = pet_data.to(device)
                labels = labels.to(device)

                mri_features = mri_encoder(mri_data)
                pet_features = pet_encoder(pet_data)

                out_mri = classifier_mri(mri_features)
                out_pet = classifier_pet(pet_features)
                _, _, out = classifier(mri_features, pet_features)

                _, preds_mri = torch.max(out_mri, 1)
                _, preds_pet = torch.max(out_pet, 1)

                _, preds = torch.max(out, dim=1)
                probabilities = torch.softmax(out, dim=1)
                positive_probs = probabilities[:, 1]

                all_probs.extend(positive_probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

                all_preds_mri.extend(preds_mri.cpu().numpy())
                all_preds_pet.extend(preds_pet.cpu().numpy())

                all_labels.extend(labels.cpu().numpy())

        ################################################################################################################
        train_loss /= len(train_dataloader)
        print(
            f'fold {fold}: Epoch {epoch + 1}/{args.epochs}, Train Loss: {train_loss:.4f}')

        acc_mri = accuracy_score(all_labels, all_preds_mri)
        acc_pet = accuracy_score(all_labels, all_preds_pet)

        metrics = calculate_metrics(all_labels, all_preds, all_probs=all_probs)
        print(f"fold {fold}: Acc: {metrics['acc']:.4f}, "
              f"SPEC: {metrics['spec']:.4f}, "
              f"Recall: {metrics['recall']:.4f}, "
              f"F1: {metrics['f1']:.4f}, "
              f"AUC: {metrics['auc']:.4f}, "
              f"mri_only_accuracy: {acc_mri:.4f}, "
              f"pet_only_accuracy: {acc_pet:.4f}")

        acc_is_best = metrics['acc'] > best_acc
        auc_is_best = (metrics['acc'] == best_acc and metrics['auc'] > best_auc)
        # save best model
        if acc_is_best or auc_is_best:
            best_acc = metrics['acc']
            best_auc = metrics["auc"]
            best_metrics_fold = metrics.copy()

            save_dir = os.path.join('checkpoints', f'{class_names[0]}_{class_names[1]}', f'{args.experiment_name}')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            model_classifier_name = f'fold-{fold}_model_head_{args.fusion_method}.pth'
            model_filepath = os.path.join(save_dir, model_classifier_name)

            if previous_model_filepath is not None and os.path.exists(previous_model_filepath):
                os.remove(previous_model_filepath)
                print(f"Deleted previous model file: {previous_model_filepath}")

            torch.save(classifier.state_dict(), model_filepath)
            print(f"Saved new best model to: {model_filepath}")

            previous_model_filepath = model_filepath
        print('-------------------------------------------------------------------------------------------------------')

    print('Finished Training and validating')

    if args.use_wandb:
        wandb.log({
            'fold': fold,
            'acc_mri': acc_mri,
            'acc_pet': acc_pet,
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
        project_name = f'final_{class_names[0]}-{class_names[1]}_Stage2'

        wandb.init(project=project_name, config=config_dict, mode="online", save_code=True, name=args.experiment_name)

    # Folder path
    if os.name == 'nt':  # Windows
        mri_path = r"E:\中间处理过程\transform_final\mri_crop_pt"
        pet_path = r"E:\中间处理过程\transform_final\pet_crop_pt"
    elif os.name == 'posix':  # Linux
        mri_path = '/root/autodl-tmp/ADNI_PROCESSED_version1/ADNI_MRI_T1_LINEAR'
        pet_path = '/root/autodl-tmp/ADNI_PROCESSED_version1/ADNI_PET_T1_LINEAR_SMOOTH'
    else:
        raise ValueError("Unsupported operating system!")

    mri_img_name_list = os.listdir(mri_path)
    subject_list_file = 'Data/Group_Subject_MRI_PET.csv'
    df = pd.read_csv(subject_list_file)

    model_dir_path = args.encoder_checkpoints_save_root

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
            mri_backbone = create_vit_backbone(pretrained=False)
            pet_backbone = create_vit_backbone(pretrained=False)

            model_mri = MriClassifier(mri_backbone, out_feature_dim=768, class_num=class_num).to(device)
            model_pet = PetClassifier(pet_backbone, out_feature_dim=768, class_num=class_num).to(device)
            out_feature_dim = 768

        else:
            raise ValueError("Unsupported backbone!")

        # load mri and pet encoder
        model_mri_path = os.path.join(model_dir_path, f'fold-{fold}_model_mri.pth')
        model_pet_path = os.path.join(model_dir_path, f'fold-{fold}_model_pet.pth')

        model_mri.load_state_dict(torch.load(model_mri_path, map_location=device, weights_only=True))
        print(f"load model_mri from {model_mri_path}!")
        model_pet.load_state_dict(torch.load(model_pet_path, map_location=device, weights_only=True))
        print(f"load model_pet from {model_pet_path}!")

        mri_encoder = model_mri.net
        pet_encoder = model_pet.net

        # fusion method (classifier)
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

        if args.only_test:
            model_head_path = os.path.join(model_dir_path, f'fold-{fold}_model_head_{args.fusion_method}.pth')
            classifier.load_state_dict(torch.load(model_head_path, map_location=device, weights_only=True))
            print(f"load model_pet from {model_head_path}!")

        classifier.to(device)
        classifier_mri = model_mri.classifier
        classifier_pet = model_pet.classifier

        # only train classifier
        optimizer = getattr(optim, args.optim_type)(classifier.parameters(), lr=args.lr, betas=(0.9, 0.999),
                                                    eps=1e-08, weight_decay=0)
        ####################################################################################################################
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
                                      shuffle=True, num_workers=nw, pin_memory=True, drop_last=False)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size,
                                     shuffle=False, num_workers=nw, pin_memory=True, drop_last=False)
        print("fold {}: {} subjects for training, {} subjects for test.".format(fold, len(train_subjects),
                                                                                len(test_subjects)))
        ################################################################################################################
        # train and test
        if args.only_test:
            all_folds_metrics = only_test(args, test_dataloader, mri_encoder, pet_encoder, classifier, classifier_mri,
                                          classifier_pet, device, fold, all_folds_metrics)
        else:
            all_folds_metrics = train_and_test(args, train_dataloader, test_dataloader, mri_encoder, pet_encoder,
                                               classifier, classifier_mri, classifier_pet, device, optimizer,
                                               fold, all_folds_metrics)
    # calculate all folds avg metrics
    calculate_all_folds_avg_metrics(args, all_folds_metrics)

    if args.use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
