import os
import time
import torch as t
import pdb
import torch.nn as nn
import torch.optim as optim
from model import CNN_BiLSTM
from dataset import dataloader
from tensorboardX import SummaryWriter

sheet_name = 213
train_loader, test_loader, torque = dataloader(sheet_name)
device = t.device('cuda' if t.cuda.is_available() else 'cpu')

def train(model: CNN_BiLSTM, optimizer: optim.Adam, criterion: nn.modules.loss, model_path, epochs: int = 100):
	"""
	model: CNN_BiLSTM
	optimizer: usualy adam
	criterion: loss function
	epochs: training epochs
	model_path: path of model storage
	"""
	best_test_loss = float('inf')
	for epoch in range(1, epochs + 1):
		start_time = time.time()

		# train
		train_loss, train_acc, epoch_loss, epoch_acc = 0, 0, 0, 0
		steps = 0
		matrix = t.zeros([2, 2])
		for step, (X_train, y_train) in enumerate(train_loader):
			model.train()
			optimizer.zero_grad()  # 不同bacth的梯度累加没有意义, optimizer实例化之后，该操作等价于model.zero_grad()
			# Forward prop
			predictions = model(X_train).squeeze(1)
			loss = criterion(predictions, y_train)
			acc = cal_accuracy(predictions, y_train)
			print('Train loss in step {}: {}'.format(step+1, loss))

			# Backward prop
			loss.backward()
			optimizer.step()

			epoch_loss += loss.item()
			epoch_acc += acc.item()
			matrix += cal_matrix(predictions, y_train)
			steps += 1

		print('The training confusion matrix in epoch {}: \n{}'.format(epoch, matrix.numpy()))
		train_loss = epoch_loss / steps
		train_acc = epoch_acc / steps
		train_IO_P, train_IO_R, train_IO_F = cal_PR(matrix, 0)
		train_NIO_P, train_NIO_R, train_NIO_F = cal_PR(matrix, 1)

		# test
		test_loss, test_acc, epoch_loss, epoch_acc = 0, 0, 0, 0
		steps = 0
		matrix = t.zeros([2, 2])
		for step, (X_test, y_test) in enumerate(test_loader):
			model.eval()
			with t.no_grad():
				predictions = model(X_test).squeeze(1)
				loss = criterion(predictions, y_test)
				acc = cal_accuracy(predictions, y_test)
				epoch_loss += loss.item()
				epoch_acc += acc.item()
				matrix += cal_matrix(predictions, y_test)
				steps += 1

		print('The test confusion matrix in epoch {}: \n{}'.format(epoch, matrix.numpy()))
		test_loss = epoch_loss / steps
		test_acc = epoch_acc / steps
		test_IO_P, test_IO_R, test_IO_F = cal_PR(matrix, 0)
		test_NIO_P, test_NIO_R, test_NIO_F = cal_PR(matrix, 1)

		# save the best model
		if test_loss < best_test_loss:
			best_test_loss = test_loss
			t.save(model.state_dict(), model_path)

		# print the performance
		print('\nEpoch: {} | Epoch Time: {:.2f} s'.format(epoch, time.time() - start_time))
		print('\tTrain: loss = {:.2f}, accuracy = {:.2f}%'.format(train_loss, train_acc * 100))
		print('\tTrain: P0 = {:.2f}%, R0 = {:.2f}%, P1 = {:.2f}%, R1 = {:.2f}%'.format(train_IO_P * 100,
		                                                                               train_IO_R * 100,
		                                                                               train_NIO_P * 100,
		                                                                               train_NIO_R * 100))
		print('\tTest: loss = {:.2f}, accuracy = {:.2f}%'.format(test_loss, test_acc * 100))
		print('\tTest: P0 = {:.2f}%, R0 = {:.2f}%, P1 = {:.2f}%, R1 = {:.2f}%'.format(test_IO_P * 100,
		                                                                              test_IO_R * 100,
		                                                                              test_NIO_P * 100,
		                                                                              test_NIO_R * 100))

def cal_accuracy(pred: t.Tensor, y: t.Tensor):
	"""
	calculate the accuracy of each batch
	pred: output of model
	y: true labels
	"""
	rounded_pred = t.round(t.sigmoid(pred))
	correct = (rounded_pred == y).float()
	accuracy = correct.sum() / y.shape[0]
	return accuracy

def cal_matrix(pred: t.Tensor, y: t.Tensor):
	"""
	calculate the confusion matrix
	"""
	rounded_pred = t.round(t.sigmoid(pred))
	matrix = t.zeros([2, 2])
	# row: actual labels
	for i in range(2):
		matrix[i][i] = ((rounded_pred == y) & (y == i)).float().sum()
		matrix[i][1-i] = ((rounded_pred != y) & (y == i)).float().sum()
	return matrix

def cal_PR(matrix: t.Tensor, i: int):
	"""
	calculate the precision ,recall and F_score
	"""
	P = matrix[i][i] / (matrix[i][i] + matrix[1-i][i])
	R = matrix[i][i] / (matrix[i][i] + matrix[i][1-i])
	F = 2*P*R / (P+R)
	return P, R, F

def main():
	# writer = SummaryWriter()
	model_path = './saved_models/model_{}.pth'.format(0)
	model = CNN_BiLSTM(in_channel=1, out_channel=2, max_len=700, window_sizes=[3, 7, 10, 20, 100], pool_maintain=4, dropout=0.5)
	# writer.add_graph(model, (next(iter(train_loader))[0],))
	optimizer = optim.Adam(model.parameters())
	criterion = nn.BCEWithLogitsLoss(pos_weight=torque.getPos_weight())

	# cuda or cpu
	model = model.to(device)
	criterion = criterion.to(device)

	if os.path.exists(model_path):
		model.load_state_dict(t.load(model_path))

	train(model, optimizer, criterion, model_path, epochs=1000)
	# writer.close()

	# load the best model and check accuracy on testset
	model.load_state_dict(t.load(model_path))
	test_loss = 0
	test_acc = 0
	epoch_loss, epoch_acc = 0, 0
	steps = 0
	matrix = t.zeros([2, 2])
	for step, (X_test, y_test) in enumerate(test_loader):
		# Change mode to eval.
		model.eval()
		with t.no_grad():
			predictions = model(X_test).squeeze(1)
			loss = criterion(predictions, y_test)
			acc = cal_accuracy(predictions, y_test)
			epoch_loss += loss.item()
			epoch_acc += acc.item()
			matrix += cal_matrix(predictions, y_test)
			steps += 1

	test_loss = epoch_loss / steps
	test_acc = epoch_acc / steps
	test_IO_P, test_IO_R, test_IO_F = cal_PR(matrix, 0)
	test_NIO_P, test_NIO_R, test_NIO_F = cal_PR(matrix, 1)
	print('\nTest the best model ever trained:')
	print('The test confusion matrix on the best model ever trained: \n{}'.format(matrix.numpy()))
	print('Test: loss = {:.2f}, accuracy = {:.2f}%'.format(test_loss, test_acc * 100))
	print('Test: P0 = {:.2f}%, R0 = {:.2f}%, P1 = {:.2f}%, R1 = {:.2f}%'.format(test_IO_P * 100, test_IO_R * 100,
	                                                                            test_NIO_P * 100, test_NIO_R * 100))


if __name__ == '__main__':
	main()