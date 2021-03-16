import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader
import argparse
from torchvision import datasets,transforms
import os
import misc
import vggbnn
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import statistics as stats
"""
Created on 22:04:10 2021/3/5

@author: (ATRer)hwh

"""
def precision(confusion):
    correct = confusion * torch.eye(confusion.shape[0])
    incorrect = confusion - correct
    correct = correct.sum(0)
    incorrect = incorrect.sum(0)
    precision = correct / (correct + incorrect)
    total_correct = correct.sum().item()
    total_incorrect = incorrect.sum().item()
    percent_correct = total_correct / (total_correct + total_incorrect)
    return precision, percent_correct

parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Example')
parser.add_argument('--wd', type=float, default=0.001, help='weight decay')
parser.add_argument('--wd_mil', type=float, default=0.000005, help='weight decay')
parser.add_argument('--batch_size', type=int, default=256, help='input batch size for training (default: 64)')
parser.add_argument('--epoch_', type=int, default=0, help='epoch to start training from')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 1e-3)')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--sample_interval', type=int, default=100, help='interval between sampling of images from generators')
parser.add_argument('--test_interval', type=int, default=100,  help='how many epochs to wait before another test')
parser.add_argument('--logdir', default='log', help='folder to save to the log')
parser.add_argument('--decreasing_lr', default='3', help='decreasing strategy')
args = parser.parse_args()
args.logdir = os.path.join(os.path.dirname(__file__), args.logdir)
misc.logger.init(args.logdir, 'log_2021_3_8_vgg_bnn')
print = misc.logger.info
os.makedirs('vgg16_models_bnn3-8', exist_ok=True)

args.cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)
print(args.cuda)

if args.cuda:
    torch.cuda.manual_seed(args.seed)

print("=================FLAGS==================")
for k, v in args.__dict__.items():
    print('{}: {}'.format(k, v))
print("========================================")


transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(45),
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

transform_test = transforms.Compose(
    [transforms.ToTensor(),
     ]
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 10
train_dataset=datasets.CIFAR10(root=r'./data1',train=True,transform=transform_train,download=True)
test_dataset=datasets.CIFAR10(root=r'./data1',train=False,transform=transform_test)
# ds = CIFAR10(r'c:\data\tv', download=True, transform=transform_test)
# len_train = len(train_dataset) // 10 * 9
# len_test = len(test_dataset) - len_train
# train, test = random_split(ds, [len_train, len_test])
train_l = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
test_l = DataLoader(test_dataset, batch_size=128, shuffle=False, drop_last=True)

if args.epoch_ != 0:
    # Load pretrained models
    model = vggbnn.vgg_small([0.5, 2, 1, 1, 1, 0.5])
    # model = vgg16.vgg16(pretrained=False)
    # model.load_state_dict(torch.load('save_models_152/model_%d.pth'%(args.epoch-1)))
    print(">>>>loading model........<<<<")
    model.load_state_dict(torch.load('vgg16_models_bnn3-8/model_190.pth'))
else:
    # model = vgg16.vgg16(pretrained=True, model_root=args.logdir)
    model = vggbnn.vgg_small([0.5, 2, 1, 1, 1, 0.5])


print(model)

if torch.cuda.is_available(): #是否使用cuda加速
    model = model.cuda()
decreasing_lr = list(map(int, args.decreasing_lr.split(',')))

print('decreasing_lr: ' + str(decreasing_lr))
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd)
i = 0
for epoch in range(args.n_epochs):
    total = 0
    running_loss = 0.0
    running_correct = 0
    # if epoch in decreasing_lr :
    #     optimizer.param_groups[0]['lr'] *= 0.1
    if (epoch+1) % 60 == 0:
        optimizer.param_groups[0]['lr'] *= 0.1
        print("Learning rate decay to :{}".format(optimizer.param_groups[0]['lr']))
    ll = []
    batch = tqdm(train_l, total=len(train_dataset) // args.batch_size)
    for x, target in batch:
        # print(x.size())
        x = x.to(device)
        # x = x.view(256, 28*28*1)
        # print(x.size())
        target = target.to(device)
        optimizer.zero_grad()
        y = model(x)
        # print(f"result :{y}")
        # print(f"target : {target}")
        loss = criterion(y, target)
        ll.append(loss.detach().item())
        batch.set_description(f'{epoch} Train Loss: {stats.mean(ll)}')
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(y.data, 1)
        # print(f"predicted:{predicted}")
        total += target.size(0)
        running_correct += (predicted == target).sum()
    print('第%d个epoch的训练识别准确率为：%d%%' % (epoch , (100 * running_correct / len(train_dataset))))

    confusion = torch.zeros(num_classes, num_classes)
    batch = tqdm(test_l, total=len(test_dataset) // 128)
    ll = []
    for x, target in batch:
        x = x.to(device)
        target = target.to(device)
        # x = x.view(256, 28 * 28 * 1)
        y = model(x)
        loss = criterion(y, target)
        ll.append(loss.detach().item())
        batch.set_description(f'{epoch} Test Loss: {stats.mean(ll)}')

        _, predicted = y.detach().max(1)
        # print(f"predicted:{predicted}")

        for item in zip(predicted, target):
            confusion[item[0], item[1]] += 1

    precision_each, percent_correct = precision(confusion)
    precision_each = precision_each.numpy()
    percent_correct = np.array(percent_correct)
    precision_each = precision_each.reshape(1, -1)
    percent_correct = percent_correct.reshape(1, -1)
    precision_total = np.append(precision_each, percent_correct, axis=1)
    print(precision_total)
    if epoch % 10 == 0:
        torch.save(model.state_dict(), 'vgg16_models_bnn3-8/model_%d.pth' % epoch)


