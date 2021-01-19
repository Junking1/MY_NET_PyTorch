import torch.nn.functional as F
import torch
import torch.nn as nn
from MyVgg import VGG16
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import math
from tqdm import tqdm
import time
def one_epoch(net, loss, epoch_size, traindata, Freeze_Epoch, Cuda, epoch):
    total_loss = 0
    val_loss = 0
    total_accuracy = 0
    val_total_accuracy = 0

    start_time = time.time()

    #    训练集开始训练
    # with tqdm(total=epoch_size, desc=f'Epoch {epoch + 1}/{Freeze_Epoch}', postfix=dict, mininterval=0.3) as pbar:
    for i, data in enumerate(traindata):
        inputs, labels = data

        if Cuda:
            inputs = Variable(inputs).cuda()
            labels = Variable(labels).cuda()
        else:
            inputs = Variable(inputs.type(torch.FloatTensor))
            labels = Variable(labels.type(torch.FloatTensor))

        optimizer.zero_grad()
        outputs = nn.Sigmoid()(net(inputs))
        output = loss(outputs, labels)

        output.backward()
        optimizer.step()

        _, qq = torch.max(outputs, 1)
        print(f"result = {qq}")
        print(f"labels = {labels}")
        equal = torch.eq(qq, labels)
        accuracy = torch.mean(equal.float())

        total_loss += output.item()
        total_accuracy += accuracy.item()
        waste_time = time.time() - start_time
        print(f"Epoch = {epoch+1}/{Freeze_Epoch} lr = {optimizer.param_groups[0]['lr']} i = {i} output = {output} acc = {accuracy} total_acc = {total_accuracy / (i + 1)}")
            #
            # pbar.set_postfix(**{'total_loss': total_loss / (i + 1),
            #                     'acc': total_accuracy / (i + 1),
            #                     's/step': waste_time})
            # pbar.update(5)




    # 主函数 #
if __name__ == '__main__':

    # 超参数设定 #
    lr = 1e-3
    Batch_size = 16
    Init_Epoch = 0
    Freeze_Epoch = 256
    train_ratio = 0.9
    gamma = 0.95
    Cuda = True

    # 路径设定 #
    train_dir = "D:\pythonProject\Jun_VGG16_0111\PetImages"
    test_dir = "D:\pythonProject\Jun_VGG16_0111\PetImages"
    log_dir = "log/"

    # 加载网络并转移至GPU #
    net = VGG16()
    # cudnn.benchmark = True
    net = net.cuda()
    print(net)

    # 设置优化器和损失函数 #
    loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=net.parameters(), lr=lr)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=1, gamma=gamma)

    # 数据和标签读入以及标签处理 #
    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}
    transform = transforms.Compose(
        transforms=[transforms.RandomResizedCrop(size=224),
                    transforms.ToTensor()]

    )

    train_set = torchvision.datasets.ImageFolder(root=train_dir, transform=transform)
    train_data = DataLoader(dataset=train_set, batch_size=Batch_size, shuffle=True, num_workers=4)
    train_num = len(train_set.imgs)

    epoch_size = max(1, train_num // Batch_size)

    for epoch in range(Init_Epoch, Freeze_Epoch):
        one_epoch(net, loss, epoch_size, train_data, Freeze_Epoch, Cuda, epoch)
        lr_scheduler.step()