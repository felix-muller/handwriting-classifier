import mlx.core as mx
from models.simple_nn import SimpleNN
from utils import cross_entropy, sgd_step, load_mnist_images, load_mnist_labels

train_images = load_mnist_images('data/train-images.idx3-ubyte')
train_labels = load_mnist_labels('data/train-labels.idx1-ubyte')
test_images = load_mnist_images('data/t10k-images.idx3-ubyte')
test_labels = load_mnist_labels('data/t10k-labels.idx1-ubyte')

train_images = mx.array(train_images)
train_labels = mx.array(train_labels)
test_images = mx.array(test_images)
test_labels = mx.array(test_labels)

model = SimpleNN()
params = model.parameters()

learning_rate = 0.1
batch_size = 16
num_epochs = 10

num_batches = train_images.shape[0] // batch_size

for epoch in range(num_epochs):
	epoch_loss = 0.0

	for i in range(num_batches):
		batch_images = train_images[i * batch_size:(i + 1) * batch_size]
		batch_labels = train_labels[i * batch_size:(i + 1) * batch_size]

		def loss_fn(params):
			model.set_parameters(params)
			logits = model.forward(batch_images)
			loss = cross_entropy(logits, batch_labels)
			return loss

		loss = loss_fn(params)
		epoch_loss += loss.item()

		grads = mx.grad(loss_fn)(params)

		new_params = sgd_step(model.parameters(), grads, learning_rate)
		model.set_parameters(new_params)
		params = new_params

	test_logits = model.forward(test_images)
	test_preds = mx.argmax(test_logits, axis=1)
	test_acc = mx.sum(test_preds == test_labels) / test_labels.shape[0]

	print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / num_batches:.4f}, Test Accuracy: {test_acc:.4f}")