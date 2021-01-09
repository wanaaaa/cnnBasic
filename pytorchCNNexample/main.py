from funClass import *
import torch

trainloader, testloader, classes = dataLoad()

# trainNet(trainloader)

dataIter = iter(testloader)
images, labels = dataIter.next()

# imShowNet(torchvision.utils.make_grid(images))
# print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

net = Net()
PATH = './cifar_net.pth'
net.load_state_dict(torch.load(PATH))
# ========================================================================
# =========================================================================
# images = torch.rand(1, 3, 32, 32)

outputs = net(images)
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))



















