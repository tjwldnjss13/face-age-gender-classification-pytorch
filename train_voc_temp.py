import torch
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split, ConcatDataset

import time
import argparse
from datasets.voc_dataset import VOCDataset, custom_collate_fn
from loss import custom_softmax_cross_entropy_loss
from metric import exact_match_ratio
from utils.pytorch_util import make_batch
from utils.util import time_calculator


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device: {}'.format(torch.cuda.get_device_name()))

    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', required=False, type=int, default=32)
    parser.add_argument('--epoch', required=False, type=int, default=100)
    parser.add_argument('--lr', required=False, type=float, default=.0001)

    args = parser.parse_args()

    batch_size = args.batch_size
    learning_rate = args.lr
    num_epochs = args.epoch

    # Generate VOC dataset
    dset_name = 'voc'
    root = 'C://DeepLearningData/VOC2012/'
    original_transforms = transforms.Compose([transforms.RandomCrop((224, 224), pad_if_needed=True),
                                              transforms.ToTensor(),
                                              transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    rotate_transforms = transforms.Compose([transforms.RandomCrop((224, 224), pad_if_needed=True),
                                            transforms.RandomRotation((-30, 30)),
                                            transforms.ToTensor(),
                                            transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    horizontal_flip_transforms = transforms.Compose([transforms.RandomCrop((224, 224), pad_if_needed=True),
                                                     transforms.RandomHorizontalFlip(1),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    vertical_flip_transforms = transforms.Compose([transforms.RandomCrop((224, 224), pad_if_needed=True),
                                                   transforms.RandomVerticalFlip(1),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize([.485, .456, .406], [.229, .224, .225])])

    original_dset = VOCDataset(root, img_size=(224, 224), transforms=original_transforms, is_categorical=True)
    rotate_dset = VOCDataset(root, img_size=(224, 224), transforms=rotate_transforms, is_categorical=True)
    horizontal_flip_dset = VOCDataset(root, img_size=(224, 224), transforms=horizontal_flip_transforms, is_categorical=True)
    vertical_flip_dset = VOCDataset(root, img_size=(224, 224), transforms=vertical_flip_transforms, is_categorical=True)

    n_class = original_dset.num_classes

    n_data = len(original_dset)
    n_train_data = int(n_data * .7)
    n_val_data = n_data - n_train_data
    tarin_val_ratio = [n_train_data, n_val_data]

    original_dset, dset_val = random_split(original_dset, tarin_val_ratio)
    rotate_dset, _ = random_split(rotate_dset, tarin_val_ratio)
    horizontal_flip_dset, _ = random_split(horizontal_flip_dset, tarin_val_ratio)
    vertical_flip_dset, _ = random_split(vertical_flip_dset, tarin_val_ratio)

    dset_train = ConcatDataset([original_dset, rotate_dset, horizontal_flip_dset, vertical_flip_dset])

    train_loader = DataLoader(dset_train, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(dset_val, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

    # Load model
    from models.mobilenetv2 import MobilenetV2
    model_name = 'mobilenetv2'
    print('Building model...')
    model = MobilenetV2(20, 0).to(device)

    # Load loss function, optimizer
    loss_func = custom_softmax_cross_entropy_loss
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=.001)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=.001)
    # summary(model, (3, 224, 224))

    print('Training...')
    train_loss_list = []
    train_emr_list = []
    val_loss_list = []
    val_emr_list = []

    n_train_data = len(dset_train)

    print('Train: {} Validation: {}'.format(len(dset_train), len(dset_val)))

    t_start = time.time()
    for e in range(num_epochs):
        loss_sum = 0
        emr_sum = 0
        num_batches = 0

        model.train()
        t_train_start = time.time()
        for i, (images, labels) in enumerate(train_loader):
            print('[{}/{}] '.format(e + 1, num_epochs), end='')
            print('{}/{} '.format((i + 1) * len(images), len(dset_train)), end='')

            num_batches += 1

            x = make_batch(images).to(device)
            y = make_batch(labels, 'label').to(device)

            predict = model(x)
            predict = torch.nn.Sigmoid()(predict)

            optimizer.zero_grad()
            loss = loss_func(predict, y)
            loss.backward()
            optimizer.step()

            predict = (predict > .5)
            emr = exact_match_ratio(predict, y)

            loss_sum += loss.item()
            emr_sum += emr

            print('<loss> {:<20} <emr> {:<20} '.format(loss.item(), emr), end='')
            print('<loss_avg> {:<20} <emr_avg> {:<21} '.format(loss_sum / num_batches, emr_sum / num_batches), end='')

            del x, y, predict, loss, emr

            t_mid = time.time()
            H, M, S = time_calculator(t_mid - t_start)

            print('<time> {:02d}:{:02d}:{:02d}'.format(H, M, int(S)))

        t_train_end = time.time()
        H, M, S = time_calculator(t_train_end - t_train_start)

        train_loss_list.append(loss_sum / num_batches)
        train_emr_list.append(emr_sum / num_batches)

        print('        <train_loss> {:<20} <train_emr> {:<20} '.format(train_loss_list[-1], train_emr_list[-1]), end='')
        print('<time> {:02d}:{:02d}:{:02d} '.format(H, M, int(S)), end='')

        with torch.no_grad():
            loss_sum = 0
            emr_sum = 0
            num_batches = 0

            model.eval()
            for i, (images, labels) in enumerate(val_loader):
                num_batches += 1

                x = make_batch(images).to(device)
                y = make_batch(labels, 'label').to(device)

                predict = model(x)
                predict = torch.nn.Sigmoid()(predict)

                loss = loss_func(predict, y)

                predict = (predict > .5)
                emr = exact_match_ratio(predict, y)

                loss_sum += loss.item()
                emr_sum += emr

                del x, y, predict, loss, emr

            val_loss_list.append(loss_sum / num_batches)
            val_emr_list.append(emr_sum / num_batches)

            print('<val_loss> {:<20} <val_emr> {:<20} '.format(val_loss_list[-1], val_emr_list[-1]))

        if (e + 1) % 5 == 0:
            PATH = 'saved models/{}_{}_{}epoch_{}lr_{:.5f}loss_{:.5f}emr.pth'.format(model_name, dset_name, e + 1,
                                                                                     learning_rate, val_loss_list[-1],
                                                                                     val_emr_list[-1])
            torch.save(model.state_dict(), PATH)

    x_axis = [i for i in range(num_epochs)]

    plt.figure(0)
    plt.title('Loss')
    plt.plot(x_axis, train_loss_list, 'r-', label='Train')
    plt.plot(x_axis, val_loss_list, 'b:', label='Validation')
    plt.legend()

    plt.figure(1)
    plt.title('Exact Match Ratio')
    plt.plot(x_axis, train_emr_list, 'r-', label='Train')
    plt.plot(x_axis, val_emr_list, 'b:', label='Validation')
    plt.legend()

    plt.show()

