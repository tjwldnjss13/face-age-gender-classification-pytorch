import time
import argparse
import torch
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, random_split

from models.facenet_ds import FaceNetDS
from datasets.afad_dataset import AFADDataset
from loss import custom_softmax_cross_entropy_loss
from metric import accuracy_argmax
from utils.pytorch_util import make_batch
from utils.util import time_calculator


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Define hyper parameters, parsers
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', required=False, type=int, default=32)
    parser.add_argument('--learning_rate', required=False, type=float, default=.001)
    parser.add_argument('--weight_decay', required=False, type=float, default=.001)
    parser.add_argument('--num_epochs', required=False, type=int, default=50)

    args = parser.parse_args()

    batch_size = args.batch_size
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    num_epochs = args.num_epochs

    model_save_term = 2

    # Load AFAD dataset
    dset_name = 'afadfull'
    root = 'D://DeepLearningData/AFAD-Full/'
    transform = transforms.Compose([transforms.Resize((112, 112)), transforms.ToTensor()])
    dset = AFADDataset(root=root, transforms=transform, categorical=True)

    n_data = len(dset)
    n_train_data = int(n_data * .7)
    n_val_data = n_data - n_train_data
    train_val_ratio = [n_train_data, n_val_data]

    train_dset, val_dset = random_split(dset, train_val_ratio)

    # Generate data loader
    train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, collate_fn=dset.custom_collate_fn)
    val_loader = DataLoader(val_dset, batch_size=batch_size, shuffle=True, collate_fn=dset.custom_collate_fn)

    # Define model
    model_name = 'facenet'
    model = FaceNetDS(num_age_classes=len(dset.age_class_list)).to(device)
    state_dict_pth = None
    if state_dict_pth is not None:
        model.load_state_dict(torch.load(state_dict_pth))

    # Define optimizer, metrics
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_func = custom_softmax_cross_entropy_loss
    acc_func = accuracy_argmax

    train_loss_list = []
    train_acc_age_list = []
    train_acc_gender_list = []
    val_loss_list = []
    val_acc_age_list = []
    val_acc_gender_list = []

    t_start = time.time()
    for e in range(num_epochs):
        num_data = 0
        num_batches = 0
        train_loss = 0
        train_acc_age = 0
        train_acc_gender = 0

        t_train_start = time.time()
        model.train()
        for i, (imgs, anns) in enumerate(train_loader):
            num_data += len(imgs)
            num_batches += 1

            print('[{}/{}] '.format(e + 1, num_epochs), end='')
            print('{}/{} '.format(num_data, len(dset)), end='')

            x = make_batch(imgs).to(device)
            y_gender = make_batch(anns, 'gender').to(device)
            y_age = make_batch(anns, 'age').to(device)

            pred_age, pred_gender = model(x)
            loss_age = custom_softmax_cross_entropy_loss(pred_age, y_age)
            loss_gender = custom_softmax_cross_entropy_loss(pred_gender, y_gender)

            loss = loss_age + loss_gender

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss = loss.detach().cpu().item()
            acc_age = acc_func(pred_age, y_age).item()
            acc_gender = acc_func(pred_gender, y_gender).item()

            train_loss += loss
            train_acc_age += acc_age
            train_acc_gender += acc_gender

            t_batch_end = time.time()
            H, M, S = time_calculator(t_batch_end - t_start)

            print('<loss> {:<10f} <acc_age> {:<10f} <acc_gender> {:<10f} '.format(loss, acc_age, acc_gender), end='')
            print('<loss_avg> {:<10f} <acc_age_avg> {:<10f} <acc_gender_avg> {:<10f} '.format(
                train_loss / num_batches, train_acc_age / num_batches, train_acc_gender / num_batches
            ), end='')
            print('<time> {:02d}:{:02d}:{:02d}'.format(int(H), int(M), int(S)))

            del x, y_age, y_gender, pred_age, pred_gender, loss, acc_age, acc_gender

        train_loss_list.append(train_loss / num_batches)
        train_acc_age_list.append(train_acc_age / num_batches)
        train_acc_gender_list.append(train_acc_gender / num_batches)

        t_train_end = time.time()
        H, M, S = time_calculator(t_train_end - t_train_start)

        print('        <train_loss> {:<10f} <train_acc_acc> {:<10f} <train_acc_gender> {:<10f} '.format(
            train_loss_list[-1], train_acc_age_list[-1], train_acc_gender_list[-1]
        ), end='')
        print('<time> {:02d}:{:02d}:{:02d}'.format(int(H), int(M), int(S)))

        num_batches = 0
        val_loss = 0
        val_acc_age = 0
        val_acc_gender = 0

        model.eval()
        with torch.no_grad():
            for j, (imgs, anns) in enumerate(val_loader):
                num_batches += 1

                x = make_batch(imgs).to(device)
                y_age = make_batch(anns, 'age').to(device)
                y_gender = make_batch(anns, 'gender').to(device)

                pred_age, pred_gender = model(x)

                loss_age = loss_func(pred_age, y_age)
                loss_gender = loss_func(pred_gender, y_gender)
                loss = loss_age + loss_gender

                loss = loss.cpu().item()
                acc_age = acc_func(pred_age, y_age).item()
                acc_gender = acc_func(pred_gender, y_gender).item()

                val_loss += loss
                val_acc_age += acc_age
                val_acc_gender += acc_gender

                del x, y_age, y_gender, pred_age, pred_gender, loss, acc_age, acc_gender

            val_loss_list.append(val_loss / num_batches)
            val_acc_age_list.append(val_acc_age / num_batches)
            val_acc_gender_list.append(val_acc_gender / num_batches)

            print('        <val_loss> {:<20} <val_acc_age> {:<20} <val_acc_gender> {:<20}'.format(
                val_loss_list[-1], val_acc_age_list[-1], val_acc_gender_list[-1]
            ))

        if (e + 1) % model_save_term == 0:
            save_pth = 'saved models/{}_{}_{}lr_{:.5f}loss_{:.3f}accage_{:.3f}accgender'.format(
                model_name, dset_name, learning_rate, val_loss_list[-1], val_acc_age_list[-1], val_acc_gender_list[-1]
            )

    x_axis = [i for i in range(num_epochs)]

    plt.figure(0)
    plt.plot(x_axis, train_loss_list, 'r-', label='Train')
    plt.plot(x_axis, val_loss_list, 'b-', label='Val')
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



















