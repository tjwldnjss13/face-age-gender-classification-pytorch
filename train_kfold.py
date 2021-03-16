import time
import argparse
import torch
import numpy as np
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision.models import mobilenet_v2
from sklearn.model_selection import KFold

from models.facenet_ds import FaceNetDS
from models.mobilenetv2 import MobilenetV2
from models.custom_mobilenet_v2 import CustomMobilenetV2
from models.mobilenetv2_mini import MobilenetV2Mini
from datasets.afad_dataset import AFADDataset
from datasets.augment import GaussianNoise
from loss import custom_weighted_focal_loss as loss_func
from metric import accuracy_argmax as acc_func
from utils.pytorch_util import make_batch
from utils.util import time_calculator
from early_stopping import EarlyStopping


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Define hyper parameters, parsers
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', required=False, type=int, default=64)
    parser.add_argument('--learning_rate', required=False, type=float, default=.0005)
    parser.add_argument('--weight_decay', required=False, type=float, default=.0005)
    parser.add_argument('--momentum', required=False, type=float, default=.9)
    parser.add_argument('--num_epochs', required=False, type=int, default=50)

    args = parser.parse_args()

    batch_size = args.batch_size
    learning_rate = args.learning_rate
    momentum = args.momentum
    weight_decay = args.weight_decay
    num_epochs = args.num_epochs

    model_save_term = 1

    # Load AFAD dataset
    dset_name = 'afadfull'
    root = 'C://DeepLearningData/AFAD-Full/Train'
    transform_og = transforms.Compose([transforms.Resize((224, 224)),
                                       transforms.ToTensor()])
    # transform_norm = transforms.Compose([transforms.Resize((224, 224)),
    #                                      transforms.ToTensor(),
    #                                      transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    # transform_noise = transforms.Compose([transforms.Resize((224, 224)),
    #                                         transforms.ToTensor(),
    #                                         GaussianNoise(mean=0, std=.1)])
    # transform_flip = transforms.Compose([transforms.Resize((224, 224)),
    #                                      transforms.RandomVerticalFlip(1),
    #                                        transforms.ToTensor()])
    # transform_rotate = transforms.Compose([transforms.Resize((224, 224)),
    #                                        transforms.RandomRotation(60),
    #                                        transforms.ToTensor()])
    # transform_flip = transforms.Compose([transforms.Resize((224, 224)),
    #                                      transforms.RandomHorizontalFlip(1),
    #                                      transforms.ToTensor(),
    #                                      transforms.Normalize([.485, .456, .406], [.229, .224, .225])])

    dset_og = AFADDataset(root=root, transforms=transform_og, categorical=True)
    # dset_norm = AFADDataset(root=root, transforms=transform_norm, categorical=True)
    # dset_noise = AFADDataset(root=root, transforms=transform_noise, categorical=True)
    # dset_flip = AFADDataset(root=root, transforms=transform_flip, categorical=True)
    # dset_rotate = AFADDataset(root, transforms=transform_rotate, categorical=True)
    # dset_flip = AFADDataset(root, transforms=transform_flip, categorical=True)

    # train_dset = ConcatDataset([train_dset_og, train_dset_noise, train_dset_flip, train_dset_rotate])
    dset = dset_og

    weight_factor_age = dset_og.age_weight_factor.to(device)
    weight_factor_gender = dset_og.gender_weight_factor.to(device)

    # Define K-fold validation validator
    num_split = 5
    kfold = KFold(n_splits=num_split, shuffle=True)

    # Define model
    model_name = 'mobilenetv2mini'
    # model = CustomMobilenetV2(dset_og.num_age_classes, freeze_convs=True).to(device)
    model = MobilenetV2Mini(dset_og.num_age_classes).to(device)
    state_dict_pth = None
    state_dict_pth = 'pretrained models/mobilenetv2mini_afadfull_1+1epoch_4fold_0.0001lr_4.92482loss_3.73339lossage_1.19143lossgender_0.07789accage_0.46133accgender.pth'
    if state_dict_pth is not None:
        model.load_state_dict(torch.load(state_dict_pth), strict=False)

    # Define optimizer, metrics
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Define early-stopping class
    early_stopping = EarlyStopping(patience=7)

    train_loss_list = []
    train_loss_age_list = []
    train_loss_gender_list = []
    train_acc_age_list = []
    train_acc_gender_list = []
    val_loss_list = []
    val_loss_age_list = []
    val_loss_gender_list = []
    val_acc_age_list = []
    val_acc_gender_list = []

    t_start = time.time()
    for e in range(num_epochs):
        num_data = 0
        num_train_batches = 0
        num_val_batches = 0

        train_loss = 0
        train_loss_age = 0
        train_loss_gender = 0
        train_acc_age = 0
        train_acc_gender = 0

        val_loss = 0
        val_loss_age = 0
        val_loss_gender = 0
        val_acc_age = 0
        val_acc_gender = 0

        t_train_start = time.time()
        model.train()
        for fold, (train_ids, val_ids) in enumerate(kfold.split(dset)):
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

            train_loader = DataLoader(dset, batch_size=batch_size, sampler=train_subsampler, collate_fn=dset_og.custom_collate_fn)
            val_loader = DataLoader(dset, batch_size=batch_size, sampler=val_subsampler, collate_fn=dset_og.custom_collate_fn)

            num_data = 0
            for i, (imgs, anns) in enumerate(train_loader):
                num_data += len(imgs)
                num_train_batches += 1

                print(f'[{e + 1}/{num_epochs}] ', end='')
                print(f'({fold + 1}/{num_split} FOLD) ', end='')
                print(f'{num_data}/{int(len(dset) * (1 - 1 / num_split))}  ', end='')

                x = make_batch(imgs).to(device)
                y_age = make_batch(anns, 'age_categorical').to(device)
                y_gender = make_batch(anns, 'gender_categorical').to(device)

                pred_age, pred_gender = model(x)
                loss_age = loss_func(pred_age, y_age, weight_factor_age, 2)
                loss_gender = loss_func(pred_gender, y_gender, weight_factor_gender, 2)

                loss = loss_age + loss_gender

                optimizer.zero_grad()
                loss_age.backward()
                optimizer.step()

                loss = loss.detach().cpu().item()
                loss_age = loss_age.detach().cpu().item()
                loss_gender = loss_gender.detach().cpu().item()
                acc_age = acc_func(pred_age, y_age).item()
                acc_gender = acc_func(pred_gender, y_gender).item()

                acc_gender = acc_func(pred_gender, y_gender).item()
                acc_age = acc_func(pred_age, y_age).item()

                train_loss += loss
                train_loss_age += loss_age
                train_loss_gender += loss_gender
                train_acc_age += acc_age
                train_acc_gender += acc_gender

                t_batch_end = time.time()
                H, M, S = time_calculator(t_batch_end - t_start)

                print(f'<loss> {loss:<9f}  <loss_age> {loss_age:<9f}  <loss_gender> {loss_gender:<9f}  '
                      f'<acc_age> {acc_age:<9f}  <acc_gender> {acc_gender:<9f}  ', end='')
                print(f'<loss_avg> {train_loss / num_train_batches:>9f}  <loss_age_avg> {train_loss_age / num_train_batches:<9f}  <loss_gender_avg> {train_loss_gender / num_train_batches:<9f}  '
                      f'<acc_age_avg> {train_acc_age / num_train_batches:<9f}  <acc_gender_avg> {train_acc_gender / num_train_batches:<9f}  ', end='')
                print(f'<time> {int(H):02d}:{int(M):02d}:{int(S):02d}')

                del x, y_age, y_gender, loss, loss_age, loss_gender, acc_age, acc_gender

            t_train_end = time.time()
            H, M, S = time_calculator(t_train_end - t_train_start)

            t_val_start = time.time()
            model.eval()
            with torch.no_grad():
                for j, (imgs, anns) in enumerate(val_loader):
                    num_val_batches += 1

                    x = make_batch(imgs).to(device)
                    y_age = make_batch(anns, 'age_categorical').to(device)
                    y_gender = make_batch(anns, 'gender_categorical').to(device)

                    pred_age, pred_gender = model(x)
                    loss_age = loss_func(pred_age, y_age, weight_factor_age, 2)
                    loss_gender = loss_func(pred_gender, y_gender, weight_factor_gender, 2)

                    loss = loss_age + loss_gender

                    loss = loss.cpu().item()
                    loss_age = loss_age.cpu().item()
                    loss_gender = loss_gender.cpu().item()
                    acc_age = acc_func(pred_age, y_age).item()
                    acc_gender = acc_func(pred_gender, y_gender).item()
                    acc_age = acc_func(pred_age, y_age).item()
                    acc_gender = acc_func(pred_gender, y_gender).item()

                    val_loss += loss
                    val_loss_age += loss_age
                    val_loss_gender += loss_gender
                    val_acc_age += acc_age
                    val_acc_gender += acc_gender

                    del x, y_age, loss_age, acc_age

                print(f'\t\t<val_loss> {val_loss / num_val_batches:<9f}  <val_loss_age> {val_loss_age / num_val_batches:<9f}  <val_loss_gender> {val_loss_gender / num_val_batches:<9f}  '
                      f'<val_acc_age> {val_acc_age / num_val_batches:<9f}  <val_acc_gender> {val_acc_gender / num_val_batches:<9f}')

            save_pth = f'saved models/{model_name}_{dset_name}_{e + 1}epoch_{fold + 1}fold_{learning_rate}lr_{val_loss / num_val_batches:.5f}loss_' \
                       f'{val_loss_age / num_val_batches:.5f}lossage_{val_loss_gender / num_val_batches:.5f}lossgender_' \
                       f'{val_acc_age / num_val_batches:.5f}accage_{val_acc_gender / num_val_batches:.5f}accgender.pth'

            if (e + 1) % model_save_term == 0:
                torch.save(model.state_dict(), save_pth)

        t_val_end = time.time()
        H, M, S = time_calculator(t_val_end - t_val_start)
        print(f'<time> {int(H):02d}:{int(M):02d}:{int(S):02d}')

        train_loss_list.append(train_loss / num_train_batches)
        train_loss_age_list.append(train_loss_age / num_train_batches)
        train_loss_gender_list.append(train_loss_gender / num_train_batches)
        train_acc_age_list.append(train_acc_age / num_train_batches)
        train_acc_gender_list.append(train_acc_gender / num_train_batches)

        val_loss_list.append(val_loss / num_val_batches)
        val_loss_age_list.append(val_loss_age / num_val_batches)
        val_loss_gender_list.append(val_loss_gender / num_val_batches)
        val_acc_age_list.append(val_acc_age / num_val_batches)
        val_acc_gender_list.append(val_acc_gender / num_val_batches)

        # early_stopping(val_loss_age_list[-1], model)
        # if early_stopping.early_stop:
        #     print('Early stopping')
        #     break

    x_axis = [i for i in range(len(train_loss_age_list))]

    plt.figure(0)
    plt.plot(x_axis, train_loss_age_list, 'r-', label='Train')
    plt.plot(x_axis, val_loss_age_list, 'b-', label='Val')
    plt.title('Loss')
    plt.legend()

    plt.figure(1)
    plt.plot(x_axis, train_acc_age_list, 'r-', label='Train')
    plt.plot(x_axis, val_acc_age_list, 'b-', label='Val')
    plt.title('Age accuracy')
    plt.legend()

    plt.figure(2)
    plt.plot(x_axis, train_acc_gender_list, 'r-', label='Train')
    plt.plot(x_axis, val_acc_gender_list, 'b-', label='Val')
    plt.title('Gender accuracy')
    plt.legend()

    plt.show()



















