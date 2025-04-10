**Title** : Optimizer in Deep Learning.

**Introduction** :

Deep learning is a subset of Machine Learning that primarily uses Neural networks to perform complex tasks. These Neural networks consist of multiple Hidden layers and each contains multiple neurons for processing and performing complex patterns in a large dataset . Deep Learning is powerful for handling unstructured data such as images, audio, and text, making it widely applicable across various fields like computer vision, natural language processing, and speech recognition.

**What is Optimizer in Deep Learning:**

	As we all know Deep Learning is designed to perform Complex problems and **Optimizer**  play an important role in improving accuracy of these models. One of the foundational techniques used across many optimizers is the **gradient-based approach,** which is one of the basic and common methods used in all Optimizers.

	In deep learning, an optimizer is a crucial element that dynamically fine-tunes parameters in the neural network’s during training. Its main objective is to **minimize the model’s loss function**, thereby improving performance. This is done by **iteratively updating the parameters** based on the feedback (errors) received from the model’s predictions.

There are various types of optimization algorithms each employing unique strategies to efficiently guide the model toward the best possible parameter values. In the following sections, we will explore the different types of optimizers and their specific applications in deep learning.

**Important terms**:

* **Weights**: Weights represent the importance assigned to input features during training. Higher weights indicate that the feature has a greater influence on the model’s predictions.

* **Biases**: Bias is an additional parameter that helps the model make accurate predictions by shifting the activation function. A high bias can indicate underfitting, leading to poor model performance.

* **Epoch**: An epoch refers to one complete pass of the entire dataset through the learning algorithm during training.

* **Sample**: A sample is a single data point or row in the dataset.

* **Batch**: A batch is a group of samples selected from the dataset used to train the model in a single iteration. It helps improve training efficiency and stability.

* **Learning Rate**: The learning rate controls how much the model’s weights are updated in response to the calculated error. A higher learning rate speeds up training but risks overshooting, while a lower rate improves accuracy but may slow down convergence.

**Types of Optimizer**:

1. Gradient Descent Optimizer:

   

* **Working**: Calculates the gradient using the **entire dataset** at once.   
* **Weight Update**: Occurs **once per epoch**, after processing all data. Move in the direction of the negative gradient to minimize loss.  
* Update rule : w \= w \- lr \* gradient  
* Ex : Imagine you're trying to find the lowest point in the valley by calculating the average slope from every rock in the valley \- its accurate but slow  
* **Advantages**:	  
  * Produces a more **stable and accurate** gradient.  
  * Tends to converge smoothly.  
*  **Disadvantages**:  
  * **Slower training** time since it waits for one full pass over the data.  
  * Can be **computationally expensive** for large datasets.  
       
2. Stochastic Gradient Descent  
* **Working**: Updates weights **for each individual data point**.  
* **Weight Update**: If an epoch has 100 samples, it performs **100 updates per epoch**.  
* **Stochastic Nature**: Introduces **randomness** as it selects one data point at a time.  
* **Advantages**:  
  * **Faster updates** and more frequent learning.  
  * Can help escape local minima due to randomness.  
* **Disadvantages**:  
  * Updates are **noisy and less stable**, which may lead to convergence oscillations.  
  * Needs **careful tuning** of learning rate and other parameters.

    

3.  Mini-Batch Gradient Descent:  
* Unlike the SGD and GD it performs with small batches within the epoch.  
* **Working**: Divides the dataset into small batches and updates weights for each batch. Combines the efficiency of GD with the speed of SGD.  
*  **Advantages**:

  * Faster than GD, more stable than SGD  
  * Leverages vectorization in computation

* **Disadvantages**:  
  * Still requires tuning of batch size and learning rate

4. SGD with Momentum:  
* Think of it like a ball rolling downhill , Keeps some speed from previous steps → smoother, faster descent.  
* **Momentum** \= Like pushing a rolling ball down a hill → gains speed, less bounce.  
* **Formula**:  
   v \= γv \+ lr \* ∇Loss(w)  
   w \= w \- v  
*  **Advantages**:  
  * Faster convergence  
  * Reduces zig-zagging  
* **Disadvantages**:  
  * Slightly more complex to tune

5. Nesterov Accelerated Gradient (NAG):  
* Like momentum but looks ahead before computing gradient.**More responsive than momentum**  
* **Advantages:**  
  * Anticipates next position  
  * Faster, more accurate descent  
* **Disadvantages:**  
  * Slightly more overhead  
      
      
6. Adagrad (Adaptive Gradient) :  
* Like walking cautiously → slows down where it's steep (frequent gradients), speeds up where it's flat (rare updates).  
*  **Advantages**:  
  * Good for sparse data (e.g., NLP)  
  * No need to manually tune learning rate  
* **Disadvantages**:  
  * Learning rate keeps decreasing — may stop learning  
      
7. RMSprop (Root Mean Square Propagation):  
* Fixes Adagrad’s shrinking learning rate problem by using moving averages of squared gradients.  
* **Advantages:**  
  * Maintains a steady learning rate Good for RNNs  
* **Disadvantages:**  
  * Sensitive to hyperparameters


8. Adam (Adaptive Moment Estimation):  
* Combines Momentum \+ RMSprop. The most widely used optimizer.  
* Maintains both first moment (mean) and second moment (variance) of gradients. **Stable and fast convergence**  
*  **Advantages:**  
  * Works well across many tasks  
  * Fast and stable convergence

* **Disadvantages:**  
  * Can be biased in early stages (usually corrected with bias correction)

* **Important Information :**  
  When it comes to deep learning **frameworks** (like PyTorch, TensorFlow, Keras), the term **"SGD"** usually refers to **Mini-Batch SGD**, not the strict theoretical SGD that uses **1 sample at a time**.  
    
* **Theoretical SGD** \= 1 sample → 1 update (rarely used in real-world DL)  
    
* **Practical / Common "SGD" in frameworks** \= **Mini-Batch SGD** → uses a small batch (like 32 or 64\) for each update.

  When we say things like **"SGD with Momentum"**, or **"SGD with Weight Decay"** in a framework, it’s usually a mini-batch behind the scenes.  
    
* What is Weight Decay (L2 Regularization)?

  **Weight decay**, often implemented as **L2 regularization**, is a technique used to penalize large model weights during training. By adding a term to the loss function or directly modifying the optimizer's update rule, it encourages the model to learn simpler, more general patterns, ultimately helping to reduce overfitting and improve performance on unseen data.

**Conclusion:**  
Optimizers are essential in deep learning for minimizing the loss function and improving model accuracy. Depending on the task, different optimizers and techniques like momentum or L2 regularization can make a big difference. A solid understanding of these techniques helps in building models that are both faster and more accurate