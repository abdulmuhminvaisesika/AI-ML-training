# Artificial Neural Networks (ANN)

* (ANN), is a group of multiple perceptrons or neurons at each layer. 
* ANN is also known as a Feed-Forward Neural network because inputs are processed only in the forward direction. 
* This type of neural networks are one of the simplest variants of neural networks. 
* They pass information in one direction, through various input nodes, until it makes it to the output node. 
* The network may or may not have hidden node layers, making their functioning more interpretable.
* Advantages:
    * Storing information on the entire network.
    * Ability to work with incomplete knowledge.
    * Having fault tolerance.
    * Having a distributed memory.
* Disadvantages:

    * Hardware dependence.
    * Unexplained behavior of the network.
    * Determination of proper network structure.

# Convolutional Neural Network (CNN):


* (CNN) are one of the most popular models used today. 
* This neural network computational model uses a variation of multilayer perceptrons and contains one or more convolutional layers that can be either entirely connected or pooled. 
* These convolutional layers create feature maps that record a region of image which is ultimately broken into rectangles and sent out for nonlinear processing.
* Advantages:

    * Very High accuracy in image recognition problems.
    * Automatically detects the important features without any human supervision.
    * Weight sharing.
* Disadvantages:

    * CNN do not encode the position and orientation of object.
    * Lack of ability to be spatially invariant to the input data.
    * Lots of training data is required.
  
# Recurrent Neural Network (RNN):

* (RNN) are more complex. 
* They save the output of processing nodes and feed the result back into the model (they did not pass the information in one direction only). 
* This is how the model is said to learn to predict the outcome of a layer. 
* Each node in the RNN model acts as a memory cell, continuing the computation and implementation of operations. 
* If the network’s prediction is incorrect, then the system self-learns and continues working towards the correct prediction during backpropagation.
* Advantages:

    * An RNN remembers each and every information through time. It is useful in time series prediction only because of the feature to remember previous inputs as well. This is called Long Short Term Memory.
    * Recurrent neural network are even used with convolutional layers to extend the effective pixel neighborhood.
    * RNNs allow the network to “remember” past information by feeding the output from one step into next step.
    * This helps the network understand the context of what has already happened and make better predictions based on that.
* Disadvantages:

    * Gradient vanishing and exploding problems
    * Training an RNN is a very difficult task.
    * It cannot process very long sequences if using tanh or relu as an activation function.



### How does RNN work?

* At each time step RNNs process units with a fixed activation function. 
* These units have an internal hidden state that acts as memory that retains information from previous time steps. 
* This memory allows the network to store past knowledge and adapt based on new inputs.

### Types Of Recurrent Neural Networks

1. One-to-One RNN
    * single input and a single output. 
    * It is used for straightforward classification tasks such as 
    * binary classification where no sequential data is involved.

2. One-to-Many RNN
    * single input to produce multiple outputs over time.
    * This is useful in tasks where one input triggers a sequence of predictions (outputs). 
    * For example in image captioning a single image can be used as input to generate a sequence of words as a caption.
3. Many-to-One RNN
    * receives a sequence of inputs and generates a single output. 
    * useful when the overall context of the input sequence is needed to make one prediction.
    * In sentiment analysis the model receives a sequence of words (like a sentence) and produces a single output like positive, negative or neutral.

4. Many-to-Many RNN
    * processes a sequence of inputs and generates a sequence of outputs.
    * In language translation task a sequence of words in one language is given as input, and a corresponding sequence in another language is generated as output.

## Variants of Recurrent Neural Networks (RNNs)


### 1. Vanilla RNN
* simplest form of RNN consists of a single hidden layer where weights are shared across time steps. 
* Vanilla RNNs are suitable for learning short-term dependencies but are limited by the vanishing gradient problem, which hampers long-sequence learning.


### 2. Bidirectional RNNs
* Bidirectional RNNs process inputs in both forward and backward directions, capturing both past and future context for each time step. 
* This architecture is ideal for tasks where the entire sequence is available, such as named entity recognition and question answering.
* Applications:- Sentiment Analysis,Named Entity Recognition, Machine Translation, Speech Recognition.
* Advantages of BRNNs
    * Enhanced context understanding:
    * Improved accuracy
    * Better handling of variable-length sequences
    * Increased robustness
* Challenges of BRNNs
    * High computational cost
    * Longer training time
    * Limited real-time applicability
    * Less interpretability


### 3. Long Short-Term Memory Networks (LSTMs)

* introduce a memory mechanism to overcome the vanishing gradient problem.
* Each LSTM cell has three gates:
  
    1. Input Gate: Controls how much new information should be added to the cell state.
    2. Forget gate: Determines what information is removed from the memory cell.

    3. Output Gate: Regulates what information should be output at the current step.
   
* Problem with Long-Term **Dependencies** in RNN
    1. Vanishing Gradient:
        * When training an RNN, the gradients get smaller and smaller as they travel backward through many time steps.
        * This makes it hard for the model to learn long-term dependencies because early information fades away.
        * Imagine passing a message in a long chain of people, and by the end, it's almost lost!
    2. Exploding Gradient:
        *  Sometimes, gradients become too large and cause instability in training.
        *  This makes the model's updates erratic and unpredictable, leading to failure in learning.
        *  It's like a microphone picking up its own sound and creating a loud, uncontrollable noise (feedback loop).


* type of LSTM
    1. Bidirectional LSTM Model
        *  variation of normal LSTM which processes sequential data in both forward and backward directions. 
        *  Bi LSTMs are made up of two LSTM networks one that processes the input sequence in the forward direction and one that processes the input sequence in the backward direction. 
        *  The outputs of the two LSTM networks are then combined to produce the final output.

* Applications of LSTM
  
    * Language Modeling-machine translation,text summarization 
    * speech recognition- transcribing speech to text
    * Time Series Forecasting- predicting stock prices, weather and energy consumption.
    * Recommender Systems - suggesting movies, music and books.
    * Video Analysis - object detection, activity recognition and action classification. When combined with Convolutional Neural Networks (CNNs) they help analyze video data and extract useful information.


### 4. Gated Recurrent Units (GRUs)
* simplify LSTMs by combining the input and forget gates into a single update gate and streamlining the output mechanism. 
* This design is computationally efficient, often performing similarly to LSTMs, and is useful in tasks where simplicity and faster training are beneficial.

* > However LSTMs are very complex structure with higher computational cost. To overcome this Gated Recurrent Unit (GRU) where introduced which uses LSTM architecture by merging its gating mechanisms offering a more efficient solution for many sequential tasks without sacrificing performance.


* The core idea behind GRUs is to use gating mechanisms to selectively update the hidden state at each time step allowing them to remember important information while discarding irrelevant details.
* GRUs aim to simplify the LSTM architecture by merging some of its components and focusing on just two main gates: the update gate and the reset gate.

* GRU consists of two main gates
    1. Update Gate 
        * This gate decides how much information from previous hidden state should be retained for the next time step.
    2. Reset Gate
        * This gate determines how much of the past hidden state should be forgotten.

* How GRUs Solve the Vanishing Gradient Problem
    * GRUs help mitigate this issue by using gates that regulate the flow of gradients during training ensuring that important information is preserved and that gradients do not shrink excessively over time. 
    * By using these gates, GRUs maintain a balance between remembering important past information and learning new, relevant data.
* GRU vs LSTM
    * GRUs are more computationally efficient because they combine the forget and input gates into a single update gate. 
    * GRUs do not maintain an internal cell state as LSTMs do, instead they store information directly in the hidden state making them simpler and faster.



## Regularization Techniques in Neural Networks

* Regularization is a set of techniques used to prevent overfitting in neural networks by improving their ability to generalize to new data.
* Regularization techniques in neural networks help prevent overfitting by controlling the complexity of the model.
* The role of regularization is to help a machine learning model learn better by avoiding overfitting
  


* The commonly used regularization techniques are : 


    1. Lasso Regularization – (L1 Regularization)
    2. Ridge Regularization – (L2 Regularization)
    3. Elastic Net Regularization – (L1 and L2 Regularization)
  

### What are Overfitting and Underfitting?

* Overfitting and underfitting are terms used to describe the performance of machine learning models in relation to their ability to generalize from the training data to unseen data.


* **Overfitting** is a phenomenon that occurs when a Machine Learning model is constrained to training set and not able to perform well on unseen data.
* That is when our model learns the noise in the training data as well. 
  
* **Underfitting** on the other hand is the case when our model is not able to learn even the basic patterns available in the dataset. 
* In the case of underfitting model is unable to perform well even on the training data hence we cannot expect it to perform well on the validation data.


### What are Bias and Variance?

* Bias-  errors which occur when we try to fit a statistical model on real-world data which does not fit perfectly well on some mathematical model.
* Variance-  error value that occurs when we try to make predictions by using data that is not previously seen by the model.

### 1. Lasso Regression

* LASSO(Least Absolute Shrinkage and Selection Operator) regression
* adds the “absolute value of magnitude” of the coefficient as a penalty term to the loss function(L). 
* adds the absolute value of coefficients as a penalty
* shrinking some to zero for feature selection.
* It helps in reducing complexity by eliminating less important features, making the model simpler and more interpretable.



### 2. Ridge Regression
* A regression model that uses the L2 regularization technique is called Ridge regression. 
* Ridge regression adds the “squared magnitude” of the coefficient as a penalty term to the loss function(L).

### 3. Elastic Net Regression

* Elastic Net Regression is a combination of both L1 as well as L2 regularization. 
* That implies that we add the absolute norm of the weights as well as the squared measure of the weights. 
* With the help of an extra hyperparameter that controls the ratio of the L1 and L2 regularization.


### Benefits of Regularization
* Prevents Overfitting
* improves Interpretability
* Enhances Performance
* Stabilizes model
* prevent Complexity
* handles Multicollinearity
* Allows Fine-Tuning
* promotes Consistency
  
> Regularization techniques like L1 (Lasso), L2 (Ridge) and Elastic Net play a important role in improving model performance by reducing overfitting and ensuring better generalization. By controlling complexity, selecting important features and stabilizing models these techniques help making more accurate predictions especially in large datasets.