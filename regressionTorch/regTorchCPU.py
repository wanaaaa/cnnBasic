import torch
from torch.autograd import Variable

x_data = Variable(torch.Tensor([[10.0], [9.0], [3.0],  [2.0] ]))
y_data = Variable(torch.Tensor([[90.0], [80.0], [50.0], [30.0]]))

class LinerRegression(torch.nn.Module):
    def __init__(self):
        super(LinerRegression, self).__init__()
        self.linear = torch.nn.Linear(1,1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

model = LinerRegression()

criterion  = torch.nn.MSELoss(size_average=False) # loss function
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
