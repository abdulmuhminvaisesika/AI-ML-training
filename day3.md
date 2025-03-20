
# Neural Network

* Neural networks are machine learning models that mimic the complex functions of the human brain. 
* These models consist of interconnected nodes or neurons that process data, learn patterns, and enable tasks such as pattern recognition and decision-making.
* key components:
  1. Neurons: The basic units that receive inputs, each neuron is governed by a threshold and an activation function.
  2. Connections: Links between neurons that carry information, regulated by weights and biases.
  3. Weights and Biases: These parameters determine the strength and influence of connections.
  4. Propagation Functions: Mechanisms that help process and transfer data across layers of neurons.
  5. Learning Rule: The method that adjusts weights and biases over time to improve accuracy.
* Three-stage process
     1. Input Computation: Data is fed into the network.
     2. Output Generation: Based on the current parameters, the network generates an output.
     3. Iterative Refinement: The network refines its output by adjusting weights and biases, gradually improving its performance on diverse tasks. 
* Neural networks are pivotal in identifying complex patterns, solving intricate challenges, and adapting to dynamic environments. 
* Their ability to learn from vast amounts of data is transformative, impacting technologies like natural language processing, self-driving vehicles, and automated decision-making.
* Neural networks streamline processes, increase efficiency, and support decision-making across various industries. 
* As a backbone of artificial intelligence, they continue to drive innovation, shaping the future of technology.
* Layers
    1. input Layer(features)
    2. Hidden Layers
    3. Output Layer
## Working of Neural Networks

### Forward Propagation

* When data is input into the network, it passes through the  network in the forward direction, from the input layer through the hidden layers to the output layer. This process is known as forward propagation. 

  1. Linear Transformation
  2. Activation

### Backpropagation
  
* After forward propagation, the network evaluates its performance using a loss function, which measures the difference between the actual output and the predicted output. 
* The goal of training is to minimize this loss. 

1. Loss Calculation
2. Gradient Calculation
3. Weight Update

## Types of Neural Networks

### 1. Feedforward Neural Networks:


* A feedforward neural network is a simple artificial neural network architecture in which data moves from input to output in a single direction.
* connections between the nodes do not form cycles
*  The network consists of an input layer, one or more hidden layers, and an output layer.
*  one direction—from input to output
  
* Training a Feedforward Neural Network
    1. Forward Propagation:
    2. Loss Calculation
    3. Backpropagation


* Usage:
  * Used in supervised learning tasks like classification and regression.
  * Suitable for structured data processing, pattern recognition, and feature extraction.
     



### 2. Multilayer Perceptron (MLP):

* MLP is a type of feedforward neural network with three or more layers, including an input layer, one or more hidden layers, and an output layer. 
* It uses nonlinear activation functions.
* widely used for solving classification and regression tasks.
* MLP consists of fully connected dense layers that transform input data from one dimension to another. 
* The purpose of an MLP is to model complex relationships between inputs and outputs, making it a powerful tool for various machine learning tasks.
* Working of Multi-Layer Perceptron
    1. Step 1: Forward Propagation
       1. Weighted Sum:
       2. Activation Function
    2. Step 2: Loss 
    3. Step 3: Backpropagation
        1. Gradient Calculation:
        2. Error Propagation
        3. Gradient Descent
    4. Step 4: Optimization
        1. Stochastic Gradient Descent (SGD)
        2. Adam Optimizer




## Weights and Biases

### Weights (W)
* Weights represent the strength of the connection between neurons.

* They determine how much influence an input has on the neuron’s output.

* Initially, weights are random, but they get updated during training to improve predictions.
### Biases (b)
* Bias is an extra parameter added to neurons that shifts the activation function.
* It helps the model learn patterns that cannot be captured with just weights.
* Without bias, the neuron would always pass through the origin (0,0), which limits learning.

**✅ Formula for a Neuron Output**
* y = W . X + b
    * X = Input values
    * W = Weights
    * b = Bias
    * y = Output (before activation)

## Activation Functions
* Activation functions introduce non-linearity into the network, enabling it to learn and model complex data patterns. Common activation functions include:

  * Sigmoid: Binary classification (e.g., spam detection, yes/no problems).
  * Tanh: 
  * ReLU (Rectified Linear Unit):  Deep learning (e.g., CNNs, Image Classification).
  * Leaky ReLU
  * Softmax :  Multi-class classification (e.g., digit recognition).

* Activation functions decide whether a neuron should be activated or not by applying a transformation to the input.
* Usage:
  * Sigmoid and Softmax are preferred for classification.
  * ReLU and its variants (Leaky ReLU, PReLU) are widely used in deep learning.

## Loss Functions
* Loss functions in neural networks measure the difference between the actual output (ground truth) and the predicted output. 
* The goal of training a neural network is to minimize the loss by adjusting the model's weights using optimization algorithms like Gradient Descent.

### Loss Functions for Regression
* Regression loss functions are used when predicting continuous values, such as house prices, stock prices, or temperature.
1. Mean Squared Error (MSE)

    * MSE calculates the average squared difference between the actual and predicted values.
    * Used in regression tasks.
    * Penalizes larger errors more than smaller ones
2. Mean Absolute Error (MAE)
    * MAE calculates the absolute difference between actual and predicted values.
3. Huber Loss (Combination of MSE and MAE)
4. Root Mean Squared Error (RMSE)
    *  measures the average magnitude of the errors in a model’s predictions. 
    *  It is the square root of the Mean Squared Error (MSE).
    *  Similar to MSE but provides an error in the same unit as the target variable.





   
### Loss Functions for Classification
* Classification loss functions are used when predicting categories, such as spam vs. not spam or digit classification (0-9).
  1. Binary Cross-Entropy (For Two Classes)
    * bce = tf.keras.losses.BinaryCrossentropy()

  2. Categorical Cross-Entropy (For Multi-Class)
    * Requires one-hot encoded labels.
    * cce = tf.keras.losses.CategoricalCrossentropy()
  3. Sparse Categorical Cross-Entropy
    * Similar to Categorical Cross-Entropy, but does not require one-hot encoding
    * sparse_cce = tf.keras.losses.SparseCategoricalCrossentropy()

* Usage:
  * MSE & RMSE: Regression problems.
  * Cross-Entropy: Classification tasks like image classification and NLP.
  * Cross-Entropy Loss is used for text-based tasks.
 








## Gradient Descent

* Gradient Descent is an optimization algorithm used to minimize the loss function by iteratively updating the weights in the direction of the negative gradient.
* Common variants of gradient descent include:
    1. Batch Gradient Descent: Updates weights after computing the gradient over the entire dataset.
    2. Stochastic Gradient Descent (SGD): Updates weights for each training example individually.
    3. Mini-batch Gradient Descent: Updates weights after computing the gradient over a small batch of training examples.

* Workflow
    1. 📌 Step 1: Initialize Weights and Biases
    2. 📌 Step 2: Forward Propagation
    3. 📌 Step 3: Compute Loss
    4. 📌 Step 4: Compute Gradients (Backpropagation)
    5. 📌 Step 5: Update Weights and Biases
    6. 📌 Step 6: Repeat Until Convergence



## Optimizers
* Optimizers help update the weights of a neural network to minimize the loss function and improve model performance.

1. Adam Optimizer

* Adam is an advanced optimization algorithm that is widely used in training deep learning models. 
* It combines the advantages of two other optimizers:

    * Momentum Optimization (which smooths the gradient updates)
    * RMSprop (which adapts learning rates for each parameter)

* Adam is a powerful optimizer that dynamically adjusts learning rates for each weight, helping neural networks converge faster and more efficiently. 
* optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
  
## RMSprop (Root Mean Square Propagation)

* RMSprop is an optimization algorithm that adjusts the learning rate for each parameter dynamically, making it effective for non-stationary objectives (like training deep neural networks)
* It is specifically designed to deal with vanishing and exploding gradients in deep networks.
* optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001)


* Usage:
  * Adam is the most commonly used optimizer in deep learning.
  * RMSprop is ideal for recurrent neural networks (RNNs).
  * Entropy-based optimizers are used for exploration in reinforcement learning.
  * Adam Optimizer is commonly used in NLP




## Basic training terminologies

* Epoch - One complete pass of the entire dataset through the neural network during training.
* Batch & Batch Size 
    * Batch: A small subset of the dataset used for training in one iteration.
    * Batch Size: The number of samples processed before the model updates its weights.
*  Learning Rate - Controls the step size during weight updates.
*  Overfitting - Model performs well on training data but poorly on unseen data.
*  Underfitting- Model is too simple and fails to capture patterns in the data.