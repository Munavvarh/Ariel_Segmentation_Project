# CS301 Project1 - Milestone3 - hyperparameter Optimization 
Members: ***Ahmed Tanvir*** & ***Munavvarhusain Bunglawala***
We will be implementing the Milestone-4 work in Google Colab for the environment of the project.  

In this assignment we needed to compress the model to fit a computer that may not have accelator GPU or enough memory. To compress we need to implement Knowledge - distillation.

# Compression methods
With large datasets and increasing parameter of neural network and with its model complexity it take up big memory and computation cost. To reduce this problem there are several methods to compress model such as pruning nd qunatuzation knowledge distillation. In this program we will try to utilize the knowledge distillation in order to compress our program without any computational error. 

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

We alrealdy had the coding done by milestone 2. The code was used from seed repo consisting of images of ariel Dubai and mask images based on specification of the lands. With segmentation code obtain from github a student model was includent to match with teacher model we had in the program. After implemnting all the classes for knoledge distillation we faced problem of making it compatible with our previous code. After that only 10 images were sampled from large dataset. We trained the model with teacher and student model. After that the sample were validated through the student class combining the student loss functions and the prediction. After many tries with 50 epoch we got our result of segmented images and training and validations graphs. The loss was same as previous code but this time it didn't require as much as gpu. We were able to run the program with half the ram as previous implementation. The compression with distillation functions helped to minimise gpu use. The training validation and validaiton loss vs epochs and the precision and recall values were close to the data from milestone 2.

# 10 Segmented Images
![image](https://user-images.githubusercontent.com/113075133/205519536-7c0a1c0b-25bd-4fc5-b2bb-c4710f8fa2fd.png)
![image](https://user-images.githubusercontent.com/113075133/205519614-1aa9beae-767d-4219-8c15-98c348259c9a.png)
![image](https://user-images.githubusercontent.com/113075133/205519571-03fc1709-ced0-48f3-9fcc-1616a1750553.png)

The images were similar to the images gotten in Milestone 2. 


# Training and Validation loss vs epochs
![ml4-plo1](https://user-images.githubusercontent.com/98997616/205514304-9545df34-e8ec-4926-92bc-35b5a88d0703.JPG)  
There was slight validation loss but overall it was pretty close to the result obtained in milestone 2.

![ml4-plot2](https://user-images.githubusercontent.com/98997616/205514308-4f345b2d-69ab-406a-bb8c-28d09c492179.JPG)  

# Precision vs Recall
![ml4-plot3](https://user-images.githubusercontent.com/98997616/205514310-e88478c4-6206-4311-bd8a-280c6687333f.JPG)  
Precision and recall value are very close to precision and recall from milestone 2.
