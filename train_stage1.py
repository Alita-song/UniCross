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

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from mydataset import Transforms1, MyDataSetMriPetClinical, Transforms2
from model.mynet import create_vit_backbone, MriClassifier, PetClassifier

from utils.utils import get_subjects_labels, set_seed
from utils.utils import cosine_similarity
from loss.MetaWeightContrastiveLoss import WeightSupConLoss
from loss.loss import calculate_my_loss


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--experiment_name', type=str, default='KJKJK')
    parser.add_argument('--class_names', type=str, default='CN,AD',
                        choices=['CN,AD', 'sMCI,pMCI'], help='names of the classes.')

    parser.add_argument('--use_wandb', type=bool, default=False,
                        help='if use wandb to log')
    parser.add_argument('--backbone', type=str, default='vit',
                        choices=['vit'], help='names of the backbone.')

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use for computation, e.g., "cpu", "cuda:0"')

    parser.add_argument('--optim_type', type=str, default='SGD', choices=['SGD'])
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--epochs', type=int, default=40)


    parser.add_argument('--use_Shared_Classifier', type=bool, default=True,
                        help='Shared_Classifier')
    parser.add_argument('--use_WeightSupConLoss', type=bool, default=True,
                        help='Weight supervised contrastive (SupCon) loss')

    parser.add_argument('--temperature', type=float, default=0.07)

    return parser.parse_args()


def save_models(fold, class_names, experiment_name, model_mri, model_pet, classifier,
                acc_mri, acc_pet, best_acc, best_accs, previous_filepaths):
    """
    Save model function
    Args:
        fold: Current fold number
        class_names: List of class names
        experiment_name: Name of experiment
        model_mri: MRI model
        model_pet: PET model
        classifier: Shared classifier
        acc_mri: MRI accuracy
        acc_pet: PET accuracy
        best_acc: Previous best accuracy
        best_accs: best_acc fold's mri acc and pet acc
        previous_filepaths: Dictionary containing paths of previous models
    Returns:
        current_acc: current best accuracy
        new_filepaths: Dictionary containing paths of new models
        best_accs: Dictionary containing best accuracies
    """
    current_acc = acc_mri + acc_pet
    if current_acc < best_acc:
        return best_acc, previous_filepaths, best_accs

    # make save dir
    save_dir = os.path.join('checkpoints', f'{class_names[0]}_{class_names[1]}', f'{experiment_name}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # make save path
    new_filepaths = {
        'mri': os.path.join(save_dir, f'fold-{fold}_model_mri.pth'),
        'pet': os.path.join(save_dir, f'fold-{fold}_model_pet.pth'),
    }
    if classifier is not None:
        new_filepaths['shared'] = os.path.join(save_dir, f'fold-{fold}_model_shared_head.pth')

    # deleted old path
    for model_type, old_path in previous_filepaths.items():
        if old_path and os.path.exists(old_path):
            os.remove(old_path)
            print(f"Deleted previous {model_type} model file: {old_path}")

    # save model
    torch.save(model_mri.state_dict(), new_filepaths['mri'])
    print(f"Saved new best mri model to: {new_filepaths['mri']}")

    torch.save(model_pet.state_dict(), new_filepaths['pet'])
    print(f"Saved new best pet model to: {new_filepaths['pet']}")

    if classifier is not None:
        torch.save(classifier.state_dict(), new_filepaths['shared'])
        print(f"Saved new best shared classifier to: {new_filepaths['shared']}")

    return current_acc, new_filepaths, {'mri': acc_mri, 'pet': acc_pet}


def forward(criterion, weight_supcon_criterion, train_data, model_mri, model_pet, classifier,
            clinical_encoder, device):
    # get data
    train_data: tuple[Any, Any, Any, Any]
    mri_data, pet_data, clinical_data, labels = train_data
    mri_data = mri_data.to(device)
    pet_data = pet_data.to(device)
    clinical_data = clinical_data.to(device)
    labels = labels.to(device)


    # forward
    mri_feature, out_mri = model_mri(mri_data)
    pet_feature, out_pet = model_pet(pet_data)

    clinical_feature = clinical_encoder(clinical_data)

    shared_out_mri = classifier(mri_feature)
    shared_out_pet = classifier(pet_feature)

    # calculate loss
    loss_align_mri = criterion(shared_out_mri, labels)
    loss_align_pet = criterion(shared_out_pet, labels)

    loss_mri = criterion(out_mri, labels)
    loss_pet = criterion(out_pet, labels)

    # multimodal contrastive loss
    similarity = cosine_similarity(clinical_feature)
    loss_contrastive = calculate_my_loss(mri_feature, pet_feature, labels, weight_supcon_criterion,
                                                 similarity)

    return loss_mri, loss_pet, loss_align_mri, loss_align_pet, loss_contrastive


def train_and_test(args, train_dataloader, test_dataloader, model_mri, model_pet, classifier, device, optimizer,
                   lr_scheduler, fold, clinical_encoder):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    weight_supcon_criterion = WeightSupConLoss(temperature=args.temperature)

    best_acc = 0.0
    best_accs = {'mri': 0.0, 'pet': 0.0}
    previous_filepaths = {'mri': None, 'pet': None, 'shared': None}

    for epoch in range(args.epochs):
        ################################################################################################################
        # train
        model_mri.train()
        model_pet.train()
        classifier.train()
        clinical_encoder.train()

        train_loss = 0.0
        _loss_mri = 0.0
        _loss_pet = 0.0
        _loss_align_mri = 0.0
        _loss_align_pet = 0.0
        _loss_contrastive = 0.0
        train_bar = tqdm(train_dataloader, file=sys.stdout)
        for step, train_data in enumerate(train_bar):
            optimizer.zero_grad()
            loss_mri, loss_pet, loss_align_mri, loss_align_pet, loss_contrastive = forward(criterion,
                                                                                           weight_supcon_criterion,
                                                                                           train_data, model_mri,
                                                                                           model_pet, classifier,
                                                                                           clinical_encoder, device)
            loss = loss_mri + loss_pet + loss_align_mri + loss_align_pet + loss_contrastive

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _loss_align_mri += loss_align_mri.item()
            _loss_align_pet += loss_align_pet.item()
            _loss_mri += loss_mri.item()
            _loss_pet += loss_pet.item()
            _loss_contrastive += loss_contrastive.item()

            train_bar.desc = ("fold {}: train epoch[{}/{}] loss:{:.3f} loss_mri:{:.3f} loss_pet:{:.3f} "
                              "loss_align_mri:{:.3f} loss_align_pet:{:.3f} loss_contrastive:{:.3f}").format(
                fold, epoch + 1, args.epochs, loss, loss_mri, loss_pet, loss_align_mri, loss_align_pet,
                loss_contrastive)

        lr_scheduler.step()
        print('fold {}: Current Learning Rate: {}'.format(fold, lr_scheduler.get_last_lr()))
        ################################################################################################################
        # test
        model_mri.eval()
        model_pet.eval()

        all_preds_mri = []
        all_preds_pet = []
        all_labels = []

        with torch.no_grad():
            for test_data in test_dataloader:
                test_data: tuple[Any, Any, Any, Any]
                mri_data, pet_data, _, labels = test_data

                mri_data = mri_data.to(device)
                pet_data = pet_data.to(device)
                labels = labels.to(device)

                _, out_mri = model_mri(mri_data)
                _, out_pet = model_pet(pet_data)

                _, preds_mri = torch.max(out_mri, 1)
                _, preds_pet = torch.max(out_pet, 1)
                all_preds_mri.extend(preds_mri.cpu().numpy())
                all_preds_pet.extend(preds_pet.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        train_loss /= len(train_dataloader)
        _loss_align_mri /= len(train_dataloader)
        _loss_align_pet /= len(train_dataloader)
        _loss_mri /= len(train_dataloader)
        _loss_pet /= len(train_dataloader)
        _loss_contrastive /= len(train_dataloader)

        acc_mri = accuracy_score(all_labels, all_preds_mri)
        acc_pet = accuracy_score(all_labels, all_preds_pet)

        print(
            f'fold {fold}: Epoch {epoch + 1}/{args.epochs}, '
            f'Train Loss: {train_loss:.4f}, '
            f'loss_mri: {_loss_mri:.4f}, '
            f'loss_pet: {_loss_pet:.4f}, '
            f'loss_align_mri: {_loss_align_mri:.4f}, '
            f'loss_align_pet: {_loss_align_pet:.4f}, '
            f'loss_contrastive: {_loss_contrastive:.4f}, '
            f'mri Val acc: {acc_mri:.4f}, '
            f'pet Val acc: {acc_pet:.4f}'
        )

        # save best model
        best_acc, previous_filepaths, best_accs = save_models(
            fold=fold,
            class_names=args.class_names.split(','),
            experiment_name=args.experiment_name,
            model_mri=model_mri,
            model_pet=model_pet,
            classifier=classifier,
            acc_mri=acc_mri,
            acc_pet=acc_pet,
            best_acc=best_acc,
            best_accs=best_accs,
            previous_filepaths=previous_filepaths
        )
        print('-------------------------------------------------------------------------------------------------------')
    print('Finished Training and validating')

    if args.use_wandb:
        wandb.log({
            'fold': fold,
            'mri_acc': best_accs['mri'],
            'pet_acc': best_accs['pet'],
        })


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
        project_name = f'final_{class_names[0]}-{class_names[1]}_Stage1'
        wandb.init(project=project_name, config=config_dict, mode="online", save_code=True, name=args.experiment_name)

    # Folder path
    if os.name == 'nt':  # Windows
        mri_path = r"E:\中间处理过程\transform_final\mri_crop_pt"
        pet_path = r"E:\中间处理过程\transform_final\pet_crop_pt"
        clinical_path = r"E:\中间处理过程\clinical_pt"
    elif os.name == 'posix':  # Linux
        mri_path = '/root/autodl-tmp/v2/mri_crop_pt'
        pet_path = '/root/autodl-tmp/v2/pet_crop_pt'
        clinical_path = '/root/autodl-tmp/v2/clinical_pt'
    else:
        raise ValueError("Unsupported operating system!")

    mri_img_name_list = os.listdir(pet_path)
    subject_list_file = 'Data/Group_Subject_MRI_PET.csv'
    df = pd.read_csv(subject_list_file)

    ####################################################################################################################
    # k-fold
    selected_columns = [col for col in df.columns if col in class_names]
    subjects, labels = get_subjects_labels(df, selected_columns)

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    for fold, (train_index, test_index) in enumerate(skf.split(subjects, labels)):
        ################################################################################################################
        # model
        if args.backbone == 'vit':
            mri_backbone = create_vit_backbone(pretrained=True)
            pet_backbone = create_vit_backbone(pretrained=True)

            model_mri = MriClassifier(mri_backbone, out_feature_dim=768, class_num=class_num).to(device)
            model_pet = PetClassifier(pet_backbone, out_feature_dim=768, class_num=class_num).to(device)
            classifier = nn.Linear(768, class_num).to(device)  # shared classifier
        else:
            raise ValueError("Unsupported backbone!")

        clinical_encoder = nn.Linear(4, 32).to(device)

        models = [model_mri, model_pet, clinical_encoder, classifier]
        parameters = [p for model in models for p in model.parameters()]

        # optimizer
        if args.optim_type == 'SGD':
            optimizer = optim.SGD(parameters,
                                  lr=args.lr,
                                  weight_decay=args.weight_decay,
                                  momentum=args.momentum)
        else:
            raise ValueError("Unsupported optimizer!")
        lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=3, eta_min=0.00001)

        ################################################################################################################
        train_subjects = subjects[train_index]
        test_subjects = subjects[test_index]

        train_dataset = MyDataSetMriPetClinical(
            mri_dir_path=mri_path,
            pet_dir_path=pet_path,
            clinical_dir_path=clinical_path,
            img_name_list=mri_img_name_list,
            subject_list=train_subjects,
            transform=Transforms1,
            class_names=class_names
        )
        test_dataset = MyDataSetMriPetClinical(
            mri_dir_path=mri_path,
            pet_dir_path=pet_path,
            clinical_dir_path=clinical_path,
            img_name_list=mri_img_name_list,
            subject_list=train_subjects,
            transform=Transforms2,
            class_names=class_names
        )

        train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                                      shuffle=True, num_workers=nw, pin_memory=True, drop_last=False)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size,
                                     shuffle=False, num_workers=nw, pin_memory=True, drop_last=False)

        print("fold {}: {} subjects for training, {} subjects for test.".format(fold, len(train_subjects),
                                                                                len(test_subjects)))

        # train and test
        train_and_test(args, train_dataloader, test_dataloader, model_mri, model_pet, classifier,
                       device, optimizer, lr_scheduler, fold, clinical_encoder)

    # save code
    if args.use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
