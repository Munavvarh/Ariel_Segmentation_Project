# CS301 Project1 - Milestone3 - hyperparameter Optimization 
Members: ***Ahmed Tanvir*** & ***Munavvarhusain Bunglawala***
We will be implementing the Milestone-4 work in Google Colab for the environment of the project.  


# Introduction to Knowledge Distillation  
Knowledge Distillation is a procedure for model compression, in which a small (student) model is trained to match a large pre-trained (teacher) model. Knowledge is transferred from the teacher model to the student by minimizing a loss function, aimed at matching softened teacher logits as well as ground-truth labels.  
The logits are softened by applying a "temperature" scaling function in the softmax, effectively smoothing out the probability distribution and revealing inter-class relationships learned by the teacher.  
