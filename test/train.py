import torchvision
from torch import nn
from model import *
from torch.utils.data import DataLoader

test_data = torchvision.datasets.CIFAR10(root='../data',train=False,transform=torchvision.transforms.ToTensor(),download=True)

print(len(test_data))

test_dataloader = DataLoader(test_data,batch_size=64)

mynet = Mynet()



epochs = 10

total_train_step = 0

loss_fn = nn.CrossEntropyLoss()

learning_rate = 1e-2

optimizer = torch.optim.Adam(mynet.parameters(),lr=learning_rate)

# for data in test_dataloader:
#     imgs, targets = data
#     print('imgs:',imgs)
#     print('imgs.shape:  ',imgs.shape)
#     print('targets:',targets)
#     print(targets.shape)
#     break

# for i in range(epochs):
#     print("------------第{}轮训练开始----------".format(i + 1))

    # 训练步骤开始
for data in test_dataloader:
    imgs, targets = data


    output = mynet(imgs)

    print(output)
    print(output.shape)

    loss = loss_fn(output, targets)

    # 反向传播 调优
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    break
        # total_train_step += 1
        # print("训练次数:{},Loss:{}".format(total_train_step, loss.item()))

