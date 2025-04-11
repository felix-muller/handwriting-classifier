import mlx.core as mx
import numpy as np

def softmax(x):
	exp_x = mx.exp(x - mx.max(x, axis=1, keepdims=True))
	return exp_x / mx.sum(exp_x, axis=1, keepdims=True)

def cross_entropy(predictions, targets):
	preds = softmax(predictions)
	targets_one_hot = one_hot(targets, num_classes=10)
	loss = -mx.sum(targets_one_hot * mx.log(preds + 1e-12)) / targets.shape[0]
	return loss

def sgd_step(params, grads, learning_rate):
	return [p - learning_rate * g for p, g in zip(params, grads)]

def load_mnist_images(filename):
	with open(filename, 'rb') as f:
		f.read(16)
		data = np.frombuffer(f.read(), dtype=np.uint8)
		data = data.reshape(-1, 28*28).astype(np.float32) / 255.0
	return data

def load_mnist_labels(filename):
	with open(filename, 'rb') as f:
		f.read(8)
		labels = np.frombuffer(f.read(), dtype=np.uint8)
	return labels

def one_hot(x, num_classes):
	eye = mx.eye(num_classes)
	return eye[x]