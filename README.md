# Handwriting Classifier with Neural Network from Scratch

This project is a **from-scratch** implementation of a simple neural network to classify handwritten digits from the MNIST dataset.

I manually built:

- a data loader for the MNIST `.idx` files,
- a simple feedforward neural network (fully connected layers + ReLU),
- my own softmax and cross-entropy loss functions,
- a custom training loop using functional-style gradient updates.

The model was implemented using **MLX**, since I am working on a MacBook with Apple Silicon.

✅ Final performance:  

- **Loss after 10 epochs:** ~0.024  
- **Test accuracy after 10 epochs:** ~97.4%

The project was inspired by [this video](https://youtu.be/w8yWXqWQYmU?si=AsviLXftIEwUjOiP).
