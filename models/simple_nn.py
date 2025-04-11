import mlx.core as mx

class Dense:
	def __init__(self, in_features, out_features):
		self.W = mx.random.normal(shape=(in_features, out_features)) * 0.01
		self.b = mx.zeros(shape=(out_features,))

	def forward(self, x):
		return mx.matmul(x, self.W) + self.b

	def parameters(self):
		return [self.W, self.b]

class ReLU:
	def forward(self, x):
		return mx.maximum(x, 0)

class SimpleNN:
	def __init__(self):
		self.fc1 = Dense(784, 128)
		self.relu1 = ReLU()
		self.fc2 = Dense(128, 64)
		self.relu2 = ReLU()
		self.fc3 = Dense(64, 10)

	def forward(self, x):
		x = self.fc1.forward(x)
		x = self.relu1.forward(x)
		x = self.fc2.forward(x)
		x = self.relu2.forward(x)
		x = self.fc3.forward(x)
		return x

	def parameters(self):
		return self.fc1.parameters() + self.fc2.parameters() + self.fc3.parameters()
	
	def set_parameters(self, params):
		self.fc1.W, self.fc1.b, self.fc2.W, self.fc2.b, self.fc3.W, self.fc3.b = params