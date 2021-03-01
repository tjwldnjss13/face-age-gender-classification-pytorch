import time
import argparse
import torch
import numpy as np
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Subset, ConcatDataset

from models.facenet_ds import FaceNetDS
from models.mobilenetv2 import MobilenetV2
from datasets.afad_dataset import AFADDataset
from loss import custom_weighted_softmax_focal_loss as loss_func
from metric import accuracy_argmax
from utils.pytorch_util import make_batch
from utils.util import time_calculator


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Define hyper parameters, parsers
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', required=False, type=int, default=32)
    parser.add_argument('--learning_rate', required=False, type=float, default=.001)
    parser.add_argument('--weight_decay', required=False, type=float, default=.00005)
    parser.add_argument('--momentum', required=False, type=float, default=.9)
    parser.add_argument('--num_epochs', required=False, type=int, default=100)

    args = parser.parse_args()

    batch_size = args.batch_size
    learning_rate = args.learning_rate
    momentum = args.momentum
    weight_decay = args.weight_decay
    num_epochs = args.num_epochs

    model_save_term = 1

    # Load AFAD dataset
    dset_name = 'afadfull'
    root = 'D://DeepLearningData/AFAD-Full/'
    transform_og = transforms.Compose([transforms.Resize((224, 224)),
                                       transforms.ToTensor()])
    transform_norm = transforms.Compose([transforms.Resize((224, 224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    # transform_rotate = transforms.Compose([transforms.Resize((224, 224)),
    #                                        transforms.RandomRotation((-60, 60)),
    #                                        transforms.ToTensor(),
    #                                        transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    # transform_flip = transforms.Compose([transforms.Resize((224, 224)),
    #                                      transforms.RandomHorizontalFlip(1),
    #                                      transforms.ToTensor(),
    #                                      transforms.Normalize([.485, .456, .406], [.229, .224, .225])])

    dset_og = AFADDataset(root=root, transforms=transform_og, categorical=True)
    dset_norm = AFADDataset(root=root, transforms=transform_og, categorical=True)
    # dset_rotate = AFADDataset(root, transforms=transform_rotate, categorical=True)
    # dset_flip = AFADDataset(root, transforms=transform_flip, categorical=True)

    n_data = len(dset_og)
    indices = list(range(n_data))
    n_train_data = int(n_data * .7)
    np.random.shuffle(indices)
    train_idx, val_idx = indices[:n_train_data], indices[n_train_data:]

    train_dset_og = Subset(dset_og, indices=train_idx)
    train_dset_norm = Subset(dset_norm, indices=train_idx)
    # train_dset_rotate = Subset(dset_rotate, indices=train_idx)
    # train_dset_flip = Subset(dset_flip, indices=train_idx)
    # train_dset = ConcatDataset([train_dset_og, train_dset_rotate, train_dset_flip])

    train_dset = ConcatDataset([train_dset_og, train_dset_norm])
    val_dset = Subset(dset_og, indices=val_idx)

    weight_factor_age = dset_og.weight_factor.to(device)

    # Generate data loader
    train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=dset_og.custom_collate_fn)
    val_loader = DataLoader(val_dset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=dset_og.custom_collate_fn)

    # Define model
    # model_name = 'facenet'
    model_name = 'mobilenetv2'
    # model = FaceNetDS(num_age_classes=len(dset.age_class_list)).to(device)
    model = MobilenetV2(dset_og.num_age_classes).to(device)
    state_dict_pth = None
    if state_dict_pth is not None:
        model.load_state_dict(torch.load(state_dict_pth))

    # Define optimizer, metrics
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay, nesterov=True)
    acc_func = accuracy_argmax

    # train_loss_list = []
    train_loss_age_list = []
    # train_loss_gender_list = []
    train_acc_age_list = []
    # train_acc_gender_list = []
    # val_loss_list = []
    val_loss_age_list = []
    # val_loss_gender_list = []
    val_acc_age_list = []
    # val_acc_gender_list = []

    t_start = time.time()
    for e in range(num_epochs):
        num_data = 0
        num_batches = 0
        # train_loss = 0
        train_loss_age = 0
        # train_loss_gender = 0
        train_acc_age = 0
        # train_acc_gender = 0

        t_train_start = time.time()
        model.train()
        for i, (imgs, anns) in enumerate(train_loader):
            num_data += len(imgs)
            num_batches += 1

            print('[{}/{}] '.format(e + 1, num_epochs), end='')
            print('{}/{} '.format(num_data, len(train_dset)), end='')

            x = make_batch(imgs).to(device)
            # y_gender = make_batch(anns, 'gender').to(device)
            y_age = make_batch(anns, 'age_categorical').to(device)

            # pred_age, pred_gender = model(x)
            # loss_age = custom_softmax_cross_entropy_loss(pred_age, y_age)
            # loss_gender = custom_softmax_cross_entropy_loss(pred_gender, y_gender)
            pred_age = model(x)
            loss_age = loss_func(pred_age, y_age, weight_factor_age, 3)

            # loss = loss_age + loss_gender

            optimizer.zero_grad()
            loss_age.backward()
            optimizer.step()

            # loss = loss.detach().cpu().item()
            loss_age = loss_age.detach().cpu().item()
            # loss_gender = loss_gender.detach().cpu().item()
            acc_age = acc_func(pred_age, y_age).item()
            # acc_gender = acc_func(pred_gender, y_gender).item()

            # acc_gender = acc_func(predict[:, :2], y[:, :2]).item()
            # acc_age = acc_func(pred_age[:, 2:], y[:, 2:]).item()

            # train_loss += loss
            train_loss_age += loss_age
            # train_loss_gender += loss_gender
            train_acc_age += acc_age
            # train_acc_gender += acc_gender
            #
            t_batch_end = time.time()
            H, M, S = time_calculator(t_batch_end - t_start)

            print('<loss_age> {:<9f}  <acc_age> {:<9f}  '.format(loss_age, acc_age), end='')
            print('<loss_age_avg> {:<9f}  <acc_age_avg> {:<9f}  '.format(
                train_loss_age / num_batches, train_acc_age / num_batches
            ), end='')
            print('<time> {:02d}:{:02d}:{:02d}'.format(int(H), int(M), int(S)))

            del x, y_age, loss_age, acc_age

        # train_loss_list.append(train_loss / num_batches)
        train_loss_age_list.append(train_loss_age / num_batches)
        # train_loss_gender_list.append(train_loss_gender / num_batches)
        train_acc_age_list.append(train_acc_age / num_batches)
        # train_acc_gender_list.append(train_acc_gender / num_batches)

        t_train_end = time.time()
        H, M, S = time_calculator(t_train_end - t_train_start)

        print('        <train_loss_age> {:<9f}  <train_acc_acc> {:<9f}  '.format(
            train_loss_age_list[-1], train_acc_age_list[-1]
        ), end='')
        print('<time> {:02d}:{:02d}:{:02d}'.format(int(H), int(M), int(S)))

        num_batches = 0
        # val_loss = 0
        val_loss_age = 0
        # val_loss_gender = 0
        val_acc_age = 0
        # val_acc_gender = 0

        t_val_start = time.time()
        model.eval()
        with torch.no_grad():
            for j, (imgs, anns) in enumerate(val_loader):
                num_batches += 1

                x = make_batch(imgs).to(device)
                y_age = make_batch(anns, 'age_categorical').to(device)
                # y_gender = make_batch(anns, 'gender').to(device)

                # pred_age, pred_gender = model(x)
                pred_age = model(x)
                loss_age = loss_func(pred_age, y_age, weight_factor_age, 3)

                # loss_age = loss_func(pred_age, y_age)
                # loss_gender = loss_func(pred_gender, y_gender)
                # loss = loss_age + loss_gender

                # loss = loss.cpu().item()
                loss_age = loss_age.cpu().item()
                # loss_gender = loss_gender.cpu().item()
                # acc_age = acc_func(pred_age, y_age).item()
                # acc_gender = acc_func(pred_gender, y_gender).item()
                # acc_gender = acc_func(pred[:, :2], y[:, :2]).item()
                # acc_age = acc_func(predict[:, 2:], y[:, 2:]).item()

                # val_loss += loss
                val_loss_age += loss_age
                # val_loss_gender += loss_gender
                val_acc_age += acc_age
                # val_acc_gender += acc_gender

                del x, y_age, loss_age, acc_age

            # val_loss_list.append(val_loss / num_batches)
            val_loss_age_list.append(val_loss_age / num_batches)
            # val_loss_gender_list.append(val_loss_gender / num_batches)
            val_acc_age_list.append(val_acc_age / num_batches)
            # val_acc_gender_list.append(val_acc_gender / num_batches)

            print('        <val_loss_age> {:.9f} <val_acc_age> {:<9f}  '.format(
                val_loss_age_list[-1], val_acc_age_list[-1]
            ), end='')

        t_val_end = time.time()
        H, M, S = time_calculator(t_val_end - t_val_start)
        print('<time> {:02d}:{:02d}:{:02d}'.format(int(H), int(M), int(S)))

        if (e + 1) % model_save_term == 0:
            save_pth = 'saved models/{}_{}_{}epoch_{}lr_{:.5f}lossage_{:.5f}accage.pth'.format(
                model_name, dset_name, e + 1, learning_rate, val_loss_age_list[-1], val_acc_age_list[-1]
            )
            torch.save(model.state_dict(), save_pth)

    x_axis = [i for i in range(num_epochs)]

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

    # plt.figure(2)
    # plt.plot(x_axis, train_acc_gender_list, 'r-', label='Train')
    # plt.plot(x_axis, val_acc_gender_list, 'b-', label='Val')
    # plt.title('Gender accuracy')
    # plt.legend()

    plt.show()



















