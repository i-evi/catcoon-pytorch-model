import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F

NEPOCH = 2
BATCH_SIZE = 64

IMG_H = 28
IMG_W = 28
IMG_C = 1

CLASSES = 10

transform_train = transforms.Compose([
	transforms.RandomResizedCrop((IMG_H, IMG_W),
		scale=(0.7, 1.2), ratio=(0.75, 1.33), interpolation=2),
	transforms.Resize((IMG_H, IMG_W), interpolation=2),
	transforms.ToTensor(),
])

transform_test = transforms.Compose([
	transforms.Resize((IMG_H, IMG_W), interpolation=2),
	transforms.ToTensor(),
])

trainset = torchvision.datasets.MNIST(root='./data',
	train=True,download=True,transform=transform_train)
testset  = torchvision.datasets.MNIST(root='./data',
	train=False,download=True,transform=transform_test)

trainloader = torch.utils.data.DataLoader(trainset,
	batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
testloader  = torch.utils.data.DataLoader(testset,
	batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

class classifier(nn.Module):
	def __init__(self, classes):
		super(classifier, self).__init__()
		self.classes = classes
		self.conv = nn.Sequential(
			nn.Conv2d(1, 32, 5, 1, 2, bias=True),
			nn.ReLU(),
			nn.MaxPool2d(2, 2),
			nn.Conv2d(32, 64, 5, 1, 2, bias=True),
			nn.ReLU(),
			nn.MaxPool2d(2, 2),
		)
		self.fc = nn.Sequential(
			nn.Linear(7 * 7 * 64, 128),
			nn.ReLU(),
			nn.Linear(128, 10),
			nn.Softmax()
		)
	def forward(self, x):
		x = self.conv(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)
		return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

C = classifier(CLASSES)
C.to(device)

optimizer = optim.Adam(C.parameters(),lr=1e-3)

def torch_make_onehot(c, x):
	l = torch.zeros(c * x.shape[0]).reshape(x.shape[0], c)
	for i, x in enumerate(x):
		l[i][x] = 1.0
	return l

C.train()

for epoch in range(NEPOCH):
	for i, data in enumerate(trainloader):
		inputs, labels = data
		inputs, labels = inputs.to(device), labels.to(device)
		outputs = C(inputs)
		onehot_labels = torch_make_onehot(10, labels).to(device)
		loss = -torch.sum(onehot_labels * torch.log(outputs))
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		print('%04d'%epoch, i, loss)
	print('%04d'%epoch, loss)

torch.save(C, 'classifier.pkl')

with torch.no_grad():
	correct = 0
	total = 0
	for data in testloader:
		images, labels = data
		images, labels = images.to(device), labels.to(device)

		out = C(images)
		_, predicted = torch.max(out.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum().item()

print('Accuracy of the network on test images:{}%'.format(100 * correct / total))
