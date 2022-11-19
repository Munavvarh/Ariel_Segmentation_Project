# CS301 Project1 - Milestone3 - hyperparameter Optimization 
Members: ***Ahmed Tanvir*** & ***Munavvarhusain Bunglawala***
We will be implementing the milestone3 work in Google Colab for the environment of the project.  

# Hyperparameter Optimization


The main agenda of Milestone 3 was to incorporate a hyperparameter optimization to the previous program of MIlestone 2. Most importantly the main program was done using the seed repo of Ariel Semantic Segmentation. It included the code to produce the segmented images. The code follows as it the images were taken from kaggle and downloaded to our local machine then it was uploaded to Google Drive. Through colab the files were accesed and iterated to go through each images. The files included the airel images and the mask images. After iterations the images were croped and patchified with the vlaue of 256. Later the images were given different colours for pixel associated with diffrent area of land such as building, land, vegitation, water and unidentified. The images were trained and with the help of 100 epochs. Lastly 10 segmented images wre produced with mask that was hgihleted with diffrent colours based on the diognostic of land from the ariel image. There was a problem that it didn't 
quite produced the segementation accurately. The program required some tuning. 

Due to the tuning, Hyperparamater Optimization comes into place. In this Milestone many different methods were assigned and our group had incroporate Hyperband implementation. Hyperband is a hyperparameter algorithm that selectes data samples and allocates to randomly configuratites sampled. After it trains the model of each configuration and then stops whichever has poor result to improve those configurations. The main part of hyperband is to halving processes because budget. When the budet is depleted hald configurations are taken out of the performance and other half are trained and it continues until the last configurations. One things is that more the configuration the more the budget which is why it needs to thought out and proceed with certain confifiuration that can run for long time.

![image](https://user-images.githubusercontent.com/113075133/202832875-64a204d3-b23c-4571-924b-fdd993dfc42f.png)

# Explaning our Assigned method Hyperband 

Hyperband is a sophisticated algorithm for hyperparameter optimization. The creators of the method framed the problem of hyperparameter optimization as a pure-exploration, non-stochastic, infinite armed bandit problem. When employing Hyperband, a resource (such as iterations, data samples, or features) is chosen and allocated to randomly sampled configurations. The model is then trained with each configuration, stopping training combinations that perform badly and providing more resources to potential configurations. Hyperband is a variation of random search, but with some explore-exploit theory to find the best time allocation for each of the configurations. It is described in details in [1]. 

***successive halving:***  

Randomly sample 64 hyper-parameter sets in the search space.  
Evaluate after 100 iterations the validation loss of all these.  
Discard the lowest performers of them to keep only a half.  
Run the good ones for 100 iterations more and evaluate.  
Discard a half.  
Run the good ones for 200 iterations more and evaluate.  
Repeat till you have only one model left.  

![hyperband](https://user-images.githubusercontent.com/98997616/202865044-09cea9e9-2bbe-418d-a627-6ab22f9b449f.JPG)

This approach requires a repetition specification (which can be set so that the first model assessment occurs after a couple of epochs) and a total budget of repetitions (which will set the total number of explored configurations).Successive halving suffers from what is called the “n vs B/n” trade-off in [1]. B is the total budget to be allocated for hyper-parameter search, and n denotes the number of configurations to be searched. The average amount of resource allocated to a certain configuration is denoted by B/n. For a given budget, it is unclear whether to search for a big number of configurations (large n) in a short period of time, or to investigate a small number of configurations while spending a large amount of resources to them (large B/n). If hyper-parameter configurations can be prejudiced rapidly for a given problem (if the dataset converges quickly, bad structures reveal themselves rapidly, or the hyper-parameter search space is not chosen wisely enough so that a randomly chosen one is likely to be very bad), then n should be large. If they are sluggish to identify (or if the search space is limited but the best configuration is sought with strong certainty), B/n should be large (at the expense of the number of tested configurations).

Hyperband solves this problem by evaluating several alternative values of n for a fixed B, essentially executing a grid search across viable values of n [1].
This allows for the optimal allocation approach to be found in an indefinite time limit.
On several real-world tests,[1] shows that a basic successive halving with a correct balance of n vs B/n is quite effective (and in many cases better) than Hyperband, however it may not generalize for every problem. On a more practical level, utilizing Hyperband allows you to checkpoint your model during learning, allowing it to be paused and resumed.

# Resources
[1] Li, Lisha, Kevin Jamieson, Giulia DeSalvo, Afshin Rostamizadeh, and Ameet Talwalkar. “Hyperband: A novel bandit-based approach to hyperparameter optimization.” arXiv preprint arXiv:1603.06560 (2016).

# Result of images Tile 5 Method (Hyperband)

# Result of Training and validation loss vs. epochs:  

# Result of Precision and Recall Values:  


