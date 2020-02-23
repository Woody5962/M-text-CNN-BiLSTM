import torch as t
from torch.utils import data
import pandas as pd
from sklearn.model_selection import train_test_split


class Torque_Dataset(data.Dataset):
	def __init__(self, sheet_name):
		self.X = []
		self.y = []

		df = pd.read_csv('./data/' + str(sheet_name) + '.csv', header=None)

		for i in range(len(df)):
			self.X.append(df.iloc[i][1:].tolist())
			self.y.append(df.iloc[i][0].tolist())

		self.X = t.unsqueeze(t.Tensor(self.X), dim=1)
		self.y = t.Tensor(self.y)

		X_train, X_test, y_train, y_test = train_test_split(self.X.numpy(), self.y.numpy(), test_size=0.3)
		self.X_train = t.from_numpy(X_train)
		self.X_test = t.from_numpy(X_test)
		self.y_train = t.from_numpy(y_train)
		self.y_test = t.from_numpy(y_test)

	def getX(self):
		return self.X

	def gety(self):
		return self.y

	def getX_train(self):
		return self.X_train

	def getX_test(self):
		return self.X_test

	def gety_train(self):
		return self.y_train

	def gety_test(self):
		return self.y_test

	def getPos_weight(self):
		return t.Tensor([(t.Tensor(self.y_train) == 0).float().sum() /
		                 (t.Tensor(self.y_train) == 1).float().sum()])


def dataloader(sheet_name, batch_ratio=1):
	"""
	bacth_ratio: batchsize / datasize
	"""
	torque = Torque_Dataset(sheet_name=sheet_name)

	X_train = torque.getX_train()
	X_test = torque.getX_test()
	y_train = torque.gety_train()
	y_test = torque.gety_test()

	batchsize_train = int(len(y_train) * batch_ratio)
	batchsize_test = int(len(y_test) * batch_ratio)

	train_dataset = data.TensorDataset(X_train, y_train)
	test_dataset = data.TensorDataset(X_test, y_test)
	train_loader = data.DataLoader(dataset=train_dataset, batch_size=batchsize_train, shuffle=True, pin_memory=True)
	test_loader = data.DataLoader(dataset=test_dataset, batch_size=batchsize_test, shuffle=True, pin_memory=True)

	return train_loader, test_loader, torque
