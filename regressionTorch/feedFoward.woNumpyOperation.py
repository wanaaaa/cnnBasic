import torch
import numpy as np
from sklearn.datasets import make_blobs

class Feedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        output = self.sigmoid(output)
        return output

def blob_label(y, label, loc):
    target = np.copy(y)
    for l in loc:
        target[y == l] = label
    return target

x_train, y_train = make_blobs(n_samples=40, n_features=2, centers= 2, cluster_std=10, shuffle=True)
x_train = torch.FloatTensor(x_train)
y_train = torch.FloatTensor(y_train)


x_test, y_test = make_blobs(n_samples=10, n_features=2, centers= 2, cluster_std=1.5, shuffle=True)
x_test = torch.FloatTensor(x_test)
y_test = torch.FloatTensor(y_test)

model = Feedforward(2, 10)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

#To see how the model is improving, we can check the test loss before the model training and
#compare it with the test loss after the training.

model.eval()
y_pred = model(x_test)
before_train = criterion(y_pred.squeeze(), y_test)
print('Test loss before traing', before_train.item())

model.train()
epoch = 20
# epoch = 100
for epoch in range(epoch):
    optimizer.zero_grad()

    y_pred = model(x_train) #Forward pass
    loss = criterion(y_pred.squeeze(), y_train) # Compute Loss

    print('Epock -->', epoch, ' train loss--> ',  loss.item())

    #Bacward pass
    loss.backward()
    optimizer.step()

model.eval()
y_pred = model(x_test)
after_train = criterion(y_pred.squeeze(), y_test)
print('Test loss after Training', after_train.item())

print(y_test)
print(y_pred.squeeze())