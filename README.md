# Milestone1-301

CS 301 Project for Semantic Segmentation of Satellite Imagery by Ahmed Tanvir and Munavvarhusain Bunglawala.\
We will be implementing this project via google colab. 

# Step 1: 
Create a new repo under one Github account of a member of your team.\
We have created new repo under Github account of @tanvirA25.\
here is the link of the repo : https://github.com/tanvirA25/Milestone-1/tree/Milestone1

# step 2:
Uploading the given seed repo for this project\
228_training_aerial_imagery.py\
simple_multi_unet_model.py

# step 3:  
Install all required dependencies in colab.\
!pip install patchify \
!pip install keras \
!pip install segmentation_models \
!pip install keract \
!pip install tensorflow==2.7.0 

# step 4: 
Installing NNI and testing it. \
! pip install nni # install nni \
! wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip    # download ngrok and unzip it \
! unzip ngrok-stable-linux-amd64.zip \
! mkdir -p nni_repo \
! git clone https://github.com/microsoft/nni.git nni_repo/nni  #clone NNI's offical repo to get examples\  
! ./ngrok authtoken 2GrWNANhHuh5SQg394e6DSTvdYT_88PY6QjcgoTVbK7x1emu4   # Registerd a ngrok account , then connected to our account using genrated authtoken.\

Now, Starting an NNI example on a port bigger than 1024, then start ngrok with the same port.\
! nnictl create --config nni_repo/nni/examples/trials/mnist-pytorch/config.yml --port 5000 & get_ipython().system_raw('./ngrok http 5000 &')\
Finally, checking the INNI UI in public url.\
! curl -s http://localhost:4040/api/tunnels # don't change the port number 4040\

Which sucesfully executed and below is attached picture of NNI UI:

![INNI-UI](https://user-images.githubusercontent.com/113075133/198894843-c4c649de-a6a7-434d-8af8-c5ae8b6b72bd.png)

