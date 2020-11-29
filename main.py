import argparse
import time
from tensorboardX import SummaryWriter
from dataset import dataset_loader
from model import LeNet5, ResNet34
import torch
from torch import nn
'''
--model ResNet34 --dataset sign_mnist --log_interval 100
--model ResNet34 --dataset sign_mnist --log_interval 100 --gray_scale
--model ResNet34 --dataset kinect_leap --log_interval 8
--model ResNet34 --dataset kinect_leap --log_interval 8 --gray_scale
--model LeNet5 --dataset sign_mnist --log_interval 100
--model LeNet5 --dataset sign_mnist --log_interval 100 --gray_scale
--model LeNet5 --dataset kinect_leap --log_interval 8
--model LeNet5 --dataset kinect_leap --log_interval 8 --gray_scale
'''

parser = argparse.ArgumentParser(description='PyTorch Gesture Recognition Model')
parser.add_argument('--model', choices=['LeNet5', 'ResNet34'], default='LeNet5', type=str, help='LeNet5(default), ResNet34')
parser.add_argument('--dataset',
                    choices=['sign_mnist', 'kinect_leap'],
                    default='sign_mnist',
                    type=str,
                    help='sign_mnist(default), kinect_leap')
parser.add_argument('--gray_scale', action='store_true', help='train with grayscale image or rgb image(defalt: rgb)')
parser.add_argument('--epoch', default=100, type=int, help='amount of epoches been processed')
parser.add_argument('--save_model', action='store_true', help='saving model or not(default: false)')
parser.add_argument('--log_interval', default=100, type=int, help='how many batches to wait before logging status(defalut:100)')
parser.set_defaults(augment=True)

writer = SummaryWriter('./log')


def main():
    global args

    args = parser.parse_args()

    # load  dataset
    if args.dataset == 'sign_mnist':
        loader = dataset_loader(True)
        if args.model == 'LeNet5':
            train_loader, test_loader = loader.load_sign_mnist(28, isGrayScale=args.gray_scale)
        elif args.model == 'ResNet34':
            train_loader, test_loader = loader.load_sign_mnist(224, isGrayScale=args.gray_scale)
        else:
            raise RuntimeError('unrecognized model name ' + repr(args.model))
    elif args.dataset == 'kinect_leap':
        loader = dataset_loader(False)
        if args.model == 'LeNet5':
            train_loader, test_loader = loader.load_kinect_leap(img_size=28, isGrayScale=args.gray_scale)
        elif args.model == 'ResNet34':
            train_loader, test_loader = loader.load_kinect_leap(img_size=224, isGrayScale=args.gray_scale)
        else:
            raise RuntimeError('unrecognized model name ' + repr(args.model))
    else:
        raise RuntimeError('unrecogniazed dataset name' + repr(args.dataset))

    # load model
    if args.model == 'LeNet5':
        model = LeNet5(class_num=loader.class_num, is_gray_scale=args.gray_scale).cuda()
    elif args.model == 'ResNet34':
        model = ResNet34(class_num=loader.class_num, is_gray_scale=args.gray_scale).cuda()
    else:
        raise RuntimeError('unrecognized model name ' + repr(args.model))

    print(model)

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # time counting
    start_time = time.time()

    for epoch in range(1, args.epoch + 1):
        train(model, train_loader, criterion, optimizer, epoch)
        test(model, test_loader, epoch)

    end_time = time.time()
    print('training process using ', end_time - start_time)

    # save model
    if args.save_model:
        gray = ''
        if args.gray_scale:
            gray = '_grayscale'
        saving_path = './trained_model/' + args.model + '_' + args.dataset + gray + 'state_dict' + '.pkl'
        torch.save(model.state_dict(), saving_path)
    return


def train(model, train_loader, criterion, optimizer, epoch):
    epoch_start_time = time.time()
    model.train()

    total = len(train_loader.dataset)

    # loss sum
    train_loss = 0

    correct = 0

    data_start_time = time.time()
    for i, (input, target) in enumerate(train_loader):
        # data_start_time = time.time()

        input = input.cuda()
        target = target.cuda()

        optimizer.zero_grad()
        output = model(input)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        predicted = output.argmax(dim=1)
        correct += torch.eq(predicted, target).float().sum().item()

        # data_time = time.time() - data_start_time

        trained_total = (i + 1) * len(target)
        loss_mean = train_loss / (i + 1)
        acc = correct / trained_total
        process = 100 * trained_total / total

        if (i + 1) % args.log_interval == 0:
            data_time = time.time() - data_start_time
            print('epoch: {} [{}/{} ({:.0f}%)] \tloss:{:.6f}({:.6f}) \taccuracy:{:.6f} \ttime:{:.6f}'.format(
                epoch, trained_total, total, process, loss, loss_mean, acc, data_time))
            data_start_time = time.time()

        writer.add_scalar('train/loss_mean', loss_mean, (epoch - 1) * len(train_loader) + i)
        writer.add_scalar('train/accuracy', acc, (epoch - 1) * len(train_loader) + i)
        writer.add_scalar('train/loss', loss, (epoch - 1) * len(train_loader) + i)

    epoch_time = time.time() - epoch_start_time
    print('epoch trained in ', epoch_time)

    return


def test(model, test_loader, epoch):
    model.eval()

    correct = 0

    for i, (input, target) in enumerate(test_loader):
        input = input.cuda()
        target = target.cuda()

        output = model(input)

        predicted = output.argmax(dim=1)
        correct += torch.eq(predicted, target).float().sum().item()

    acc = correct / len(test_loader.dataset)

    writer.add_scalar('test/accuracy', acc, (epoch - 1) * len(test_loader))

    print('epoch: {} accuracy:{}'.format(epoch, acc))


if __name__ == "__main__":
    main()
