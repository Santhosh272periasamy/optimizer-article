# Optimizer in Deep Learning

---

## Introduction

Deep learning is a subset of Machine Learning that primarily uses neural networks to perform complex tasks. These neural networks consist of multiple hidden layers and each contains multiple neurons for processing and identifying complex patterns in a large dataset. Deep learning is powerful for handling unstructured data such as images, audio, and text, making it widely applicable across various fields like computer vision, natural language processing, and speech recognition.

---

## What is an Optimizer in Deep Learning?

As we all know, deep learning is designed to perform complex problems, and optimizers play an important role in improving the accuracy of these models. One of the foundational techniques used across many optimizers is the gradient-based approach, which is a common method in all optimizers.

In deep learning, an optimizer is a crucial element that dynamically fine-tunes parameters in the neural networks during training. Its main objective is to minimize the model’s loss function, thereby improving performance. This is done by iteratively updating the parameters based on the feedback (errors) received from the model’s predictions.

There are various types of optimization algorithms, each employing unique strategies to efficiently guide the model toward the best possible parameter values. In the following sections, we will explore the different types of optimizers and their specific applications in deep learning.

---

## Important Terms

- **Weights**: Represent the importance assigned to input features during training.
- **Biases**: An additional parameter that helps the model make accurate predictions by shifting the activation function.
- **Epoch**: One complete pass of the entire dataset through the learning algorithm during training.
- **Sample**: A single data point or row in the dataset.
- **Batch**: A group of samples selected from the dataset used to train the model in a single iteration.
- **Learning Rate**: Controls how much the model’s weights are updated in response to the calculated error.

---

## Types of Optimizers

### Gradient Descent Optimizer

- **Working**: Calculates the gradient using the entire dataset at once.
- **Weight Update**: Occurs once per epoch.
- **Update Rule**: `w = w - lr * gradient`
- **Example**: Like finding the lowest point in a valley by averaging every rock's slope — accurate but slow.

**Advantages**:
- Produces a stable and accurate gradient.
- Converges smoothly.

**Disadvantages**:
- Slower training time.
- Computationally expensive for large datasets.

---

### Stochastic Gradient Descent (SGD)

- **Working**: Updates weights for each individual data point.
- **Weight Update**: Performed per sample.
- **Stochastic Nature**: Introduces randomness.

**Advantages**:
- Faster updates and learning.
- Can escape local minima.

**Disadvantages**:
- Noisy and less stable updates.
- Needs careful tuning.

---

### Mini-Batch Gradient Descent

- **Working**: Divides dataset into small batches.
- **Efficiency**: Combines GD's accuracy and SGD's speed.

**Advantages**:
- Faster than GD.
- More stable than SGD.
- Leverages vectorized computation.

**Disadvantages**:
- Needs tuning of batch size and learning rate.

---

### SGD with Momentum

- **Concept**: Like a ball rolling downhill, retains velocity from past updates.
- **Formulas**:
  - `V = γ * V_old - lr * gradient`
  - `W = W_old + V`

**Advantages**:
- Faster convergence.
- Reduces oscillations.

**Disadvantages**:
- More complex to tune.

---

### Nesterov Accelerated Gradient (NAG)

- **Concept**: Similar to momentum but looks ahead.
- **Formulas**:
  - `V = γ * V_old - lr * gradient`
  - `W = W_old + γ * V - lr * gradient`

**Advantages**:
- Anticipates next move.
- Faster and more accurate.

**Disadvantages**:
- Slightly more overhead.

---

### Adagrad (Adaptive Gradient)

- **Concept**: Adapts learning rate; slows down on frequent features.

**Advantages**:
- Good for sparse data (e.g., NLP).
- No need for manual learning rate tuning.

**Disadvantages**:
- Learning rate decreases continuously.

---

### RMSprop (Root Mean Square Propagation)

- **Concept**: Fixes Adagrad’s decreasing learning rate using moving averages.

**Advantages**:
- Stable learning rate.
- Good for RNNs.

**Disadvantages**:
- Sensitive to hyperparameters.

---

### Adam (Adaptive Moment Estimation)

- **Concept**: Combines momentum and RMSprop.
- **Maintains**: Mean and variance of gradients.

**Advantages**:
- Works well across many tasks.
- Fast and stable convergence.

**Disadvantages**:
- Can be biased in early training (correctable).

---

### AdamW

- **Concept**: Adam + correct weight decay.
- **Decouples**: Weight decay from gradient update.

**Advantages**:
- Better regularization.
- Improves model accuracy.

**Disadvantages**:
- Additional hyperparameter to tune.
- Not always better than SGD.

---

## Important Notes


- In frameworks like PyTorch, TensorFlow, and Keras, “SGD” usually means **Mini-Batch SGD**, not the theoretical 1-sample version.
- When you see terms like **“SGD with Momentum”**, it usually refers to mini-batch variants.

![image.png](attachment:b4065dee-1c0f-43e5-9129-e087d1fdb30e.png)

![image.png](attachment:2a3acddf-4d24-4404-a6c0-12f7743c785f.png)

---

### What is Weight Decay (L2 Regularization)?

Weight decay (L2 regularization) is a technique to penalize large weights, promoting simpler models. It helps reduce overfitting and improves generalization.

---

## Conclusion

Optimizers are essential in deep learning for minimizing the loss function and improving model accuracy. Depending on the task, different optimizers and techniques like momentum or L2 regularization can make a big difference. A solid understanding of these techniques helps in building models that are both faster and more accurate.
