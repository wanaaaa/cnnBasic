import torch

def genRanTensor(samplePerClass, pixelNum):
    print("====================================")
    numOfSamples = samplePerClass
    pixellNum = pixelNum
    train_x0 = torch.rand(numOfSamples, 1, pixellNum, pixellNum) - 0.05
    train_x1 = torch.rand(numOfSamples, 1, pixellNum, pixellNum)
    train_x2 = torch.rand(numOfSamples, 1, pixellNum, pixellNum ) + 0.05

    train_y0 = torch.ones(numOfSamples)
    train_y1 = torch.ones(numOfSamples)
    train_y2 = torch.ones(numOfSamples)

    train_y0[:] = 0.0
    train_y1[:] = 1.0
    train_y2[:] = 2.0

    train_x = torch.cat((train_x0, train_x1, train_x2), 0)
    train_y = torch.cat((train_y0, train_y1, train_y2), 0)

    indexRandom = torch.randperm(train_x.shape[0])
    train_xRan = train_x[indexRandom]
    train_yRan = train_y[indexRandom]

    return train_xRan, train_yRan

class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.cov1 = torch.nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3,stride=1,padding=0)
        self.relu = torch.nn.ReLU()
        self.maxPool = torch.nn.MaxPool2d(kernel_size=2)
        self.fc = torch.nn.Linear(16*2*2, 3) #2*2 is a pool size

    def forward(self, x):
        # print("first x->", x.shape)
        out = self.cov1(x)
        # print('after cov1->', out.shape)
        out = self.relu(out)
        # print("after relu->", out.shape)
        out = self.maxPool(out)
        # print("after maxPool->", out.shape)

        # out = out.view(12, 16*2*2)
        out = out.view(-1, 16*2*2)
        # print("view->", out.shape)

        out = self.fc(out)
        # print("after fc->", out.shape)

        return out
