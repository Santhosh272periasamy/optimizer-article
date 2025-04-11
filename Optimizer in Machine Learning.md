# Optimizer in Deep Learning

## Introduction

Deep learning is a subset of Machine Learning that primarily uses Neural Networks to perform complex tasks. These networks consist of multiple hidden layers, each containing multiple neurons that process and learn patterns in large datasets. Deep Learning excels at handling unstructured data such as images, audio, and text, making it widely applicable across fields like computer vision, natural language processing, and speech recognition.

## What is an Optimizer in Deep Learning?

Optimizers play a critical role in improving the accuracy of deep learning models. A common foundational technique is the **gradient-based approach**, used in nearly all optimizers.

An optimizer dynamically adjusts the model's parameters during training to **minimize the loss function**. This is achieved by **iteratively updating the parameters** based on feedback from the model's prediction errors.

There are various optimization algorithms, each with unique strategies to guide the model toward optimal parameter values. The following sections describe the types of optimizers and their roles.

## Important Terms

- **Weights**: Importance assigned to input features during training.
- **Biases**: Additional parameter that shifts the activation function for better prediction accuracy.
- **Epoch**: One complete pass of the entire dataset through the learning algorithm.
- **Sample**: A single data point in the dataset.
- **Batch**: A subset of samples used in one training iteration.
- **Learning Rate**: Controls the size of the step taken during weight updates.

---

## Types of Optimizers

### 1. Gradient Descent (GD)

- **Working**: Uses the **entire dataset** to calculate gradients.
- **Update Frequency**: Once per epoch.
- **Formula**:  
  `w = w - lr * gradient`
- **Example**: Like finding the lowest point in a valley by averaging all terrain.
- **Advantages**:
  - Stable and accurate gradient estimates
  - Smooth convergence
- **Disadvantages**:
  - Slow training
  - Computationally expensive for large datasets

---

### 2. Stochastic Gradient Descent (SGD)

- **Working**: Updates weights **per sample**.
- **Update Frequency**: 1 update per sample (e.g., 100 updates per epoch if 100 samples).
- **Advantages**:
  - Faster updates
  - Helps escape local minima
- **Disadvantages**:
  - Noisy updates, less stable
  - Requires careful tuning of parameters

---

### 3. Mini-Batch Gradient Descent

- **Working**: Uses **small batches** of data per update.
- **Advantages**:
  - Faster than GD, more stable than SGD
  - Enables vectorized operations
- **Disadvantages**:
  - Requires tuning batch size and learning rate

---

### 4. SGD with Momentum

- **Concept**: Keeps velocity from past updates, like a ball rolling downhill.
- **Formula**:  
  `v = γv + lr * ∇Loss(w)`  
  `w = w - v`
- **Advantages**:
  - Faster convergence
  - Smoother updates
- **Disadvantages**:
  - Requires additional tuning

---

### 5. Nesterov Accelerated Gradient (NAG)

- **Concept**: Similar to momentum but looks ahead before updating.
- **Advantages**:
  - Anticipates future position
  - More accurate updates
- **Disadvantages**:
  - Slightly more computational overhead

---

### 6. Adagrad (Adaptive Gradient)

- **Concept**: Adapts learning rate based on frequency of parameters.
- **Advantages**:
  - Effective for sparse data (e.g., NLP)
  - No need to manually tune learning rate
- **Disadvantages**:
  - Learning rate decreases over time → may stop learning

---

### 7. RMSprop (Root Mean Square Propagation)

- **Concept**: Uses moving average of squared gradients to stabilize learning rate.
- **Advantages**:
  - Good for RNNs
  - Maintains steady learning rate
- **Disadvantages**:
  - Sensitive to hyperparameters

---

### 8. Adam (Adaptive Moment Estimation)

- **Concept**: Combines momentum and RMSprop.
- **Tracks**:
  - First moment (mean)
  - Second moment (variance)
- **Advantages**:
  - Fast and stable
  - Performs well on many tasks
- **Disadvantages**:
  - Biased in early training (bias-corrected internally)

---

## Practical Notes on Optimizer Usage

- In deep learning **frameworks** like PyTorch, TensorFlow, and Keras, **"SGD"** usually refers to **Mini-Batch SGD**, not the theoretical version (1 sample → 1 update).
- Variants like **"SGD with Momentum"** or **"SGD with Weight Decay"** typically use mini-batches under the hood.

---

## Weight Decay (L2 Regularization)

**Weight decay**, or **L2 regularization**, is a technique to reduce overfitting by penalizing large weights. It adds a term to the loss function to encourage smaller, simpler models that generalize better.

---

## Conclusion

Optimizers are essential for training deep learning models. Understanding their strengths, weaknesses, and use cases allows for smarter choices when building models. Whether it’s simple SGD, adaptive optimizers like Adam, or enhancements like momentum and weight decay, each has its place in improving accuracy and performance.
