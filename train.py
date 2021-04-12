import time
import argparse
import torch
import numpy as np
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision.models import mobilenet_v2

from models.facenet_ds import FaceNetDS
from models.mobilenetv2 import MobilenetV2
from models.custom_mobilenet_v2 import CustomMobilenetV2
# from models.mobilenetv2_mini import MobilenetV2Mini
from models.mobilenetv2_mini2 import MobilenetV2Mini
from models.agnet import AGNet
from datasets.afad_dataset import AFADDataset
from datasets.augment import GaussianNoise
from loss import *
from metric import accuracy_argmax as acc_func
from utils.pytorch_util import make_batch
from utils.util import time_calculator
from early_stopping import EarlyStopping


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Define hyper parameters, parsers
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', required=False, type=int, default=32)
    parser.add_argument('--learning_rate', required=False, type=float, default=.0001)
    parser.add_argument('--weight_decay', required=False, type=float, default=.00005)
    parser.add_argument('--momentum', required=False, type=float, default=.9)
    parser.add_argument('--num_epochs', required=False, type=int, default=20)

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
    transform_og = transforms.Compose([transforms.Resize((112, 112)),
                                       transforms.ToTensor(),
                                       transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    # transform_noise = transforms.Compose([transforms.Resize((224, 224)),
    #                                       transforms.ToTensor(),
    #                                       GaussianNoise(mean=0, std=.1),
    #                                       transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    transform_flip = transforms.Compose([transforms.Resize((112, 112)),
                                         transforms.RandomHorizontalFlip(1),
                                         transforms.ToTensor(),
                                         transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    transform_rotate = transforms.Compose([transforms.Resize((112, 112)),
                                           transforms.RandomRotation(60),
                                           transforms.ToTensor(),
                                           transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    transform_shear = transforms.Compose([transforms.Resize((112, 112)),
                                          transforms.RandomAffine(degrees=0, shear=20),
                                          transforms.ToTensor(),
                                          transforms.Normalize([.485, .456, .406], [.229, .224, .225])])

    dset_og = AFADDataset(root=root, transform=transform_og, categorical=True)
    # dset_noise = AFADDataset(root=root, transform=transform_noise, categorical=True)
    dset_rotate = AFADDataset(root, transform=transform_rotate, categorical=True)
    dset_flip = AFADDataset(root, transform=transform_flip, categorical=True)
    dset_shear = AFADDataset(root, transform=transform_shear, categorical=True)

    n_data = len(dset_og)
    indices = list(range(n_data))
    n_train_data = int(n_data * .7)
    np.random.shuffle(indices)
    train_idx, val_idx = indices[:n_train_data], indices[n_train_data:]

    train_dset = Subset(dset_og, indices=train_idx)
    # train_dset_noise = Subset(dset_noise, indices=train_idx)
    train_dset_rotate = Subset(dset_rotate, indices=train_idx)
    train_dset_flip = Subset(dset_flip, indices=train_idx)
    train_dset_shear = Subset(dset_shear, indices=train_idx)

    train_dset = ConcatDataset([train_dset, train_dset_flip, train_dset_rotate, train_dset_shear])
    val_dset = Subset(dset_og, indices=val_idx)

    weight_factor_age = dset_og.age_weight_factor.to(device)
    weight_factor_gender = dset_og.gender_weight_factor.to(device)

    # Generate data loader
    train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=dset_og.custom_collate_fn)
    val_loader = DataLoader(val_dset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=dset_og.custom_collate_fn)

    # Define model
    model_name = 'AGNet'
    # model = CustomMobilenetV2(dset_og.num_age_classes, freeze_convs=True).to(device)
    model = AGNet(dset_og.num_age_classes).to(device)
    state_dict_pth = None
    # state_dict_pth = 'pretrained models/AGNet_afadfull_7epoch_0.0001lr_5.68728loss_5.63766lossage_0.04962lossgender_0.10419accage_0.93183accgender.pth'
    if state_dict_pth is not None:
        model.load_state_dict(torch.load(state_dict_pth), strict=False)

    # Define optimizer, metrics
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    # Define early-stopping class
    early_stopping = EarlyStopping(patience=7)

    num_iter = 0
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
        num_batches = 0
        train_loss = 0
        train_loss_age = 0
        train_loss_gender = 0
        train_acc_age = 0
        train_acc_gender = 0

        t_train_start = time.time()
        model.train()
        for i, (imgs, anns) in enumerate(train_loader):
            num_data += len(imgs)
            num_batches += 1
            num_iter += 1

            print('[{}/{}] '.format(e + 1, num_epochs), end='')
            print(f'({num_iter}) ', end='')
            print('{}/{}  '.format(num_data, len(train_dset)), end='')

            x = make_batch(imgs).to(device)
            y_age = make_batch(anns, 'age_categorical').to(device)
            y_gender = make_batch(anns, 'gender_categorical').to(device)

            pred_age, pred_gender = model(x)
            # loss_age = custom_weighted_focal_loss(pred_age, y_age, weight_factor_age, 2)
            loss_age = custom_focal_loss(pred_age, y_age)
            loss_gender = custom_focal_loss(pred_gender, y_gender) * .1

            loss = loss_age + loss_gender

            if torch.isnan(loss).sum() > 0:
                exit(0)

            optimizer.zero_grad()
            loss.backward()
            # loss_age.backward()
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
            print(f'<loss_avg> {train_loss / num_batches:>9f}  <loss_age_avg> {train_loss_age / num_batches:<9f}  <loss_gender_avg> {train_loss_gender / num_batches:<9f}  '
                  f'<acc_age_avg> {train_acc_age / num_batches:<9f}  <acc_gender_avg> {train_acc_gender / num_batches:<9f}  ', end='')
            # print(f'<loss_age> {loss_age:<9f}  <acc_age> {acc_age:<9f}  ', end='')
            # print(f'<loss_age_avg> {train_loss_age / num_batches:<9f}  <acc_age_avg> {train_acc_age / num_batches:<9f}  ',
            #     end='')
            print(f'<time> {int(H):02d}:{int(M):02d}:{int(S):02d}')

            if num_iter > 0 and num_iter % 5000 == 0:
                save_pth = f'saved models/ckp_{model_name}_{dset_name}_{e + 1}epoch_{num_iter}iter_{learning_rate}lr.pth'
                torch.save(model.state_dict(), save_pth)

            del x, y_age, y_gender, loss, loss_age, loss_gender, acc_age, acc_gender

        train_loss_list.append(train_loss / num_batches)
        train_loss_age_list.append(train_loss_age / num_batches)
        train_loss_gender_list.append(train_loss_gender / num_batches)
        train_acc_age_list.append(train_acc_age / num_batches)
        train_acc_gender_list.append(train_acc_gender / num_batches)

        t_train_end = time.time()
        H, M, S = time_calculator(t_train_end - t_train_start)

        print(f'\t\t<train_loss> {train_loss_list[-1]:<9f}  <train_loss_age> {train_loss_age_list[-1]:<9f}  <trian loss_gender> {train_loss_gender_list[-1]:<9f}  ', end='')
        print(f'<train_acc_age> {train_acc_age_list[-1]:<9f}  <train_acc_gender> {train_acc_gender_list[-1]:<9f}  ', end='')
        # print(f'\t\t<train_loss_age> {train_loss_age_list[-1]:<9f}  ', end='')
        # print(f'<train_acc_age> {train_acc_age_list[-1]:<9f}  ', end='')
        print(f'<time> {int(H):02d}:{int(M):02d}:{int(S):02d}')

        num_batches = 0
        val_loss = 0
        val_loss_age = 0
        val_loss_gender = 0
        val_acc_age = 0
        val_acc_gender = 0

        t_val_start = time.time()
        model.eval()
        with torch.no_grad():
            for j, (imgs, anns) in enumerate(val_loader):
                num_batches += 1

                x = make_batch(imgs).to(device)
                y_age = make_batch(anns, 'age_categorical').to(device)
                y_gender = make_batch(anns, 'gender_categorical').to(device)

                pred_age, pred_gender = model(x)
                loss_age = custom_focal_loss(pred_age, y_age)
                loss_gender = custom_focal_loss(pred_gender, y_gender) * .1

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

            val_loss_list.append(val_loss / num_batches)
            val_loss_age_list.append(val_loss_age / num_batches)
            val_loss_gender_list.append(val_loss_gender / num_batches)
            val_acc_age_list.append(val_acc_age / num_batches)
            val_acc_gender_list.append(val_acc_gender / num_batches)

            print(f'\t\t<val_loss> {val_loss_list[-1]:<9f}  <val_loss_age> {val_loss_age_list[-1]:<9f}  <val_loss_gender> {val_loss_gender_list[-1]:<9f}  '
                  f'<val_acc_age> {val_acc_age_list[-1]:<9f}  <val_acc_gender> {val_acc_gender_list[-1]:<9f}  ', end='')

        t_val_end = time.time()
        H, M, S = time_calculator(t_val_end - t_val_start)
        print(f'<time> {int(H):02d}:{int(M):02d}:{int(S):02d}')

        save_pth = f'saved models/{model_name}_{dset_name}_{e + 1}epoch_{learning_rate}lr_{val_loss_list[-1]:.5f}loss_' \
                   f'{val_loss_age_list[-1]:.5f}lossage_{val_loss_gender_list[-1]:.5f}lossgender_' \
                   f'{val_acc_age_list[-1]:.5f}accage_{val_acc_gender_list[-1]:.5f}accgender.pth'
        if (e + 1) % model_save_term == 0:
            torch.save(model.state_dict(), save_pth)

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



















