# CS301 Project1 - Milestone3 - hyperparameter Optimization 
Members: ***Ahmed Tanvir*** & ***Munavvarhusain Bunglawala***
We will be implementing the milestone3 work in Google Colab for the environment of the project.  

# Hyperparameter Optimization


The main agenda of Milestone 3 was to incorporate a hyperparameter optimization to the previous program of MIlestone 2. Most importantly the main program was done using the seed repo of Ariel Semantic Segmentation. It included the code to produce the segmented images. The code follows as it the images were taken from kaggle and downloaded to our local machine then it was uploaded to Google Drive. Through colab the files were accesed and iterated to go through each images. The files included the airel images and the mask images. After iterations the images were croped and patchified with the vlaue of 256. Later the images were given different colours for pixel associated with diffrent area of land such as building, land, vegitation, water and unidentified. The images were trained and with the help of 100 epochs. Lastly 10 segmented images wre produced with mask that was hgihleted with diffrent colours based on the diognostic of land from the ariel image. There was a problem that it didn't 
quite produced the segementation accurately. The program required some tuning. 

Due to Tuning Hyperparamater Optimization comes into place. In this Milestone many different methods were assigned and our group had incroporate Hyperband implementation. Hyperband is a hyperparameter algorithm that selectes data samples and allocates to randomly configuratites sampled. After it trains the model of each configuration and then stops whichever has poor result to improve those configurations. The main part of hyperband is to halving processes because budget. When the budet is depleted hald configurations are taken out of the performance and other half are trained and it continues until the last configurations. One things is that more the configuration the more the budget which is why it needs to thought out and proceed with certain confifiuration that can run for long time.

![image](https://user-images.githubusercontent.com/113075133/202832875-64a204d3-b23c-4571-924b-fdd993dfc42f.png)
