import torch as t
import torch.nn as nn
import pdb
import math


class CNN_BiLSTM(nn.Module):
	def __init__(self, in_channel=1, out_channel=2, max_len=700, window_sizes=None, pool_maintain=1, dropout=0.4):
		"""
		in_channel: the number of channels of input, for our problem, is 1
		out_channel: the numebr of convolutional kernels
		max_len: the max_len of a torque sequence
		window_size: (list) different size of convolutional kernel
		pool_maintain: the lenth of representations after pooling layer
		dropout: the dropout rate
		"""
		super(CNN_BiLSTM, self).__init__()
		# 避免使用可变参数
		if window_sizes is None:
			window_sizes = [3, 5, 7]
		self.lstm_hidden_size = 2  # 实验表明，hidden_size为2效果要比1提升至少2%
		self.out_features = 1

		def padding_size(h: int):
			input_size = max_len - h + 1
			if input_size / pool_maintain == 0:
				p = 0
			else:
				p = math.ceil((math.ceil(input_size / pool_maintain) * pool_maintain - input_size) / 2)
			return p

		# input of conv1d: (batch, in_channel, seq_len)
		self.CNN = nn.ModuleList([
			nn.Sequential(nn.Conv1d(in_channels=in_channel, out_channels=out_channel, kernel_size=h, bias=True),
			              nn.BatchNorm1d(out_channel),
			              nn.ReLU(),
			              nn.MaxPool1d(kernel_size=math.ceil((max_len - h + 1)/pool_maintain), padding=padding_size(h)))
			for h in window_sizes
		])
		# output: (len(window_sizes, batch, out_channels, pool_maintain))

		# input of lstm: (seq_len, batch, input_size)
		self.BiLSTM = nn.LSTM(input_size=out_channel * len(window_sizes), hidden_size=self.lstm_hidden_size,
		                      num_layers=1, batch_first=True, dropout=0, bidirectional=True)
		# output of lstm: (seq_len, batch, hidden_size * num_directions)
		self.dropout = nn.Dropout(p=dropout)
		self.fc1 = nn.Linear(in_features=pool_maintain * self.lstm_hidden_size * 2, out_features=pool_maintain * self.lstm_hidden_size)
		self.fc2 = nn.Linear(in_features=pool_maintain * self.lstm_hidden_size, out_features=self.out_features)

	def forward(self, x):
		x = [conv(x) for conv in self.CNN]
		# 将各个尺寸kernel提取的特征拼接，然后将维度变换成lstm的input维度
		x = t.cat(x, dim=1).permute(2, 0, 1).contiguous()
		x = self.BiLSTM(x)[0].permute(1, 0, 2).contiguous()
		# 将seq_len * lstm_hidden_size拉伸成一维向量，作为fc输入
		x = x.view(x.size(0), -1)
		# lstm只有一层，无法加dropout。所以在这里加入dropout
		x = self.dropout(x)
		# 在这里使用sigmoid比leakey relu效果要好
		x = t.sigmoid(self.fc1(x))
		return self.fc2(x)
