# CS301 Project1 - Milestone3 - hyperparameter Optimization 
Members: ***Ahmed Tanvir*** & ***Munavvarhusain Bunglawala***
We will be implementing the Milestone-4 work in Google Colab for the environment of the project.  

In this assignment we needed to compress the model to fit a computer that may not have accelator GPU or enough memory. To compress we need to implement Knowledge - distillation.

# Introduction to Knowledge Distillation  
Knowledge Distillation is a procedure for model compression, in which a small (student) model is trained to match a large pre-trained (teacher) model. Knowledge is transferred from the teacher model to the student by minimizing a loss function, aimed at matching softened teacher logits as well as ground-truth labels.  
The logits are softened by applying a "temperature" scaling function in the softmax, effectively smoothing out the probability distribution and revealing inter-class relationships learned by the teacher.  

![milestone4](https://user-images.githubusercontent.com/98997616/205421446-78c0a7e5-becd-4d9b-94bd-ee089765b405.JPG)


# Construct Distiller() class

The custom Distiller() class, overrides the Model methods train_step, test_step, and compile(). In order to use the distiller, we need:  

A trained teacher model  
A student model to train  
A student loss function on the difference between student predictions and ground-truth  
A distillation loss function, along with a temperature, on the difference between the soft student predictions and the soft teacher labels  
An alpha factor to weight the student and distillation loss  
An optimizer for the student and (optional) metrics to evaluate performance  
In the train_step method, we perform a forward pass of both the teacher and student, calculate the loss with weighting of the student_loss and distillation_loss by alpha and 1 - alpha, respectively, and perform the backward pass. Note: only the student weights are updated, and therefore we only calculate the gradients for the student    weights.  


# Resources   
https://arxiv.org/pdf/1503.02531.pdf

The implemention was done using the code from Milestone-2. With segmentation code obtain from github. A student model was includent to match with teacher model we had in the program. After implemnting all the classes for knoledge distillation we faced problem of making it compatible with our previous code. After that only 10 images were sampled from large dataset. We trained the model with teacher and student model. After that the sample were validated through the student class compiging the student loss functions and the prediction. After many tries with 5 epoch we got our result of segmented images and training and validations graphs. 
