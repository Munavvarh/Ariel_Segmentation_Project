
### Tanvir Ahmed & Munavvarhusain Bunglawala  
### CS301-101 - Introduction to Data Science  
### Project : Semantic-Segmentation-of-Satellite-Imagery  
### November 6th, 2022

Milestone 2 is directod to perform Baseline Semantic. To execute the program we were provded with the seed repo. The seed repo was 228_seemantic_segmentation_of_ariel_imagery_using_unet. It provided with all the necessary code to go through the images and masks and rescale and print 10 segmented images from validation sets.

From kaggle were able to get data containing the images and mask of satelight images of Dubai. After downloading the contents, it was uploaded to Google drive. Then through collab, after installing all the dependencies and giving access to Google drive, the seed repo was able to exeute the files from kaggle. Although there were some problem. We got the image and the mask image after running the seed repo but it was only printing 1 image and 1 mask image. After some modification using a for loop we were able to produce the 10 images including the mask image. 

The seed repo was simple. It stores the directory of image in collab by acessing it through google drive and iterates through the images and crops the image to 256 x 256 x3. The patchify size was 256 as shown in the repo and the video. Same goes for the masks. After that using hexadecimal the RGB was incorported to produce the mask image. The pixels of the mask are replaced if its found and is labeled for each type of images such as "Building", "Land", "Road", "Vegetation", "Water", and "Unlabeled". Then the images are processed for cateofirical loss and dicess or segementation models loss. Along the way the the data of the mages are trained and the compiled with 100 epochs. After that training and validation accuracy graph is produced based on the result of the categorical croosentropy. To predict images built keras is function is used then s 10 images were produced with the image and mask image and prediciton image.

The importat thing in the milestone was the unet. It demostrate the the localize of the network and trainnind data in terms of data. To produce a 32 x 32 pixel, the network is slow to run each patches that it effects the localization accuracy. Large patches requires more max pooling layers and reduces accuracy. With convualtional network the 2x2 max pooling wuth rectified ReLU the features is doubled and it haves the number of feature channels with a cropped feature map. In order to display a precise segmentaion Unet is used to learn segmentaion end to end setting and able to precisely localize and distinguish border. Its important to classify every pizel as that the input and output have the same size. Also U-net has fewer annotaion of images to train reducing GPU time compared to other network.

![UNET img](https://user-images.githubusercontent.com/98997616/200198638-ea5aec85-d58f-49aa-8726-e6969e93ed54.PNG)


## Result of the plots 

![plt3](https://user-images.githubusercontent.com/98997616/200198120-e8714754-e8c7-4b28-a666-a26a752146b1.PNG)  


![plt1](https://user-images.githubusercontent.com/98997616/200198118-3a652ffd-6a43-4bfc-8955-bcd28ff408d4.PNG)  
![plt2](https://user-images.githubusercontent.com/98997616/200198119-038305cc-7a11-4007-9ed1-e745acbe3cf4.PNG)  
## Result of the images and masks 


![result1](https://user-images.githubusercontent.com/98997616/200198299-9bb3fc5f-df8c-4c07-8d35-c3177e6afa98.PNG)  
![result2](https://user-images.githubusercontent.com/98997616/200198300-49641745-1380-4b3a-80cf-246f892c5b7c.PNG)  

## Modifying the images and masks (extras)
![img3](https://user-images.githubusercontent.com/98997616/200198392-8220ab7b-a0f2-4a59-b0ae-c905390bfd0c.PNG)  

