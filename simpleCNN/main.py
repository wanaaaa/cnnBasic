import torch
from torch.utils.data import DataLoader, TensorDataset
from classFun import *

x_train, y_train = genRanTensor(4,6)
y_train = y_train.long()

dataSet = TensorDataset(x_train, y_train)
loader = DataLoader(dataSet, batch_size=4, shuffle=True, num_workers=0)

model = SimpleCNN()

error = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

for epoch in range(1):
    print("epoch-->", epoch)
    running_loss = 0.0
    for i, data in enumerate(loader, 0):
        x_train, y_train = data
        # print("i->", i, "  ", x_train.shape)
        model.train()
        optimizer.zero_grad()
        y_predicted = model(x_train)
        loss = error(y_predicted, y_train)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

PATH = './simpleNet.pth'
torch.save(model.state_dict(), PATH)

predic_x = torch.rand(1, 1, 6, 6)
predic_y = model(predic_x)
print(predic_y)
