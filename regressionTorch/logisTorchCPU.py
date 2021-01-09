import torch
from torch.autograd import Variable
from torch.nn import functional as F

x_data = Variable(torch.Tensor([[10.0], [9.0], [3.0],  [2.0]]))
y_data = Variable(torch.Tensor([[1.0], [1.0], [0.0], [0.0]]))


class LogisticRegression(torch.nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = F.sigmoid(self.linear(x))
        return y_pred

model = LogisticRegression()

criterion  = torch.nn.BCELoss(size_average=True) # loss function
optimizer  = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(20):
    model.train()
    optimizer.zero_grad()

    #Forward pass
    y_pred = model(x_data)

    # Compute Loss
    loss = criterion(y_pred, y_data)

    #Backward pass
    loss.backward()
    optimizer.step()

    print('epoch--->', epoch)

new_x =  Variable(torch.Tensor([[4.0]]))
y_pred = model(new_x)
print("predicted Y --->", y_pred[0][0])
