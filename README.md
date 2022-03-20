# Potato-Disease-Classification

### Introduction
 Farmers every year face economic loss and crop waste due to various diseases in potato plants. We will use image classification using CNN and built a model and when  a input is given the model will predict if the plant has a disease or not .
 
 ### Technology Stack
1. Model Building: tensorflow, CNN, data augmentation, tf dataset
2. Backend Server and ML Ops: tf serving, FastAPI, Postman
3. Model Optimization: Quantization, Tensorflow
4. Deployment: GCP (Google cloud platform, GCF (Google cloud functions)

### Convolution Layers

![image](https://github.com/ask-santosh/Potato-Disease-Classification-/blob/main/Screenshot%202021-08-30%20at%209.45.15%20PM.png)

Total parameters to be trained on: 1,83,747

### Training Graph

![image](https://github.com/ask-santosh/Potato-Disease-Classification-/blob/main/Screenshot%202021-08-30%20at%209.44.59%20PM.png)

After training I got the loss=0.0329 , accuracy=0.9884, validation loss= 0.2217, validation accuracy= 0.9427

### Predicted Output

![image](https://github.com/ask-santosh/Potato-Disease-Classification-/blob/main/Screenshot%202021-08-30%20at%209.44.46%20PM.png)

### Running The Program

1. Place all the leaf images in PlantImages folder and images are available in kaggle website.
2. Change your folder path in code .
3. run training.py file .
4. You can change batch size and epochs as per your requirements.
5. Run main.py file for FastAPI and test it through Postman.


### Deployment To GCP(Google Cloud Platform)

1. Create a GCP account.
2. Deploy the model to GCP using google cloud function(GCF).
3. Create a link of that GCF model and by using postman test with your images.
