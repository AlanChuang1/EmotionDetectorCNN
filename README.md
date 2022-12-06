# Emoji Prediction Using CNN - Group 4 - ECS 171 FQ 2022

Group Project for ECS 171, Fall Quarter 2022, at UC Davis under Dr. Solares.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1mFxnk88tvQjZwlSB8vthuq_pnYSQRe-D?usp=sharing)


## Group Members:

| Member Name  | GitHub Username |
| ------------- | ------------- |
| Alan Chuang  | AlanChuang1  |
| Daksh Jain  | DakshJ4033  |
| Sohum Goel  | SohumGoel  |
| Arnav Rastogi  | Arnav33R |
| Grisha Bandodkar  | grishaab |


## Project Abstract

### Overview and Approach:

Emojis are widely used for modern communication, and offer a way to express emotions visually. Through computer vision and machine learning, the aim of our project is to be able to predict the emoji that best fits a given facial expression. Since there are many emojis, to keep our project simple and achievable within the given time constraint, we are going to use just a few: Happy, Sad, and Neutral. If time permits, we will move onto more complex emotions: Angry, Disgust, Fear, and Surprise. Our approach is to use a Convolution Neural Network for facial recognition, and we are going to classify these expressions into a category of emotions. From there, we will simply output the emoji that matches that particular emotion. 

Dataset: https://www.kaggle.com/datasets/msambare/fer2013 

### Description of Dataset:

The dataset contains images of faces that are 48x48 pixel grayscale and centered, thus occupying almost the same amount of space for each image, making recognition easier. The training set consists of 28,709 examples and the public test set consists of 3,589 examples.


### Data Exploration

There are 4,254 observations in the dataset, with 1774 happy observations, 1233 neutral observations, and 1247 sad observations. Each “observation”, or image file, is a 48x48 pixel-sized grayscale image of expressions on faces. Sizes are all standardized, so they don’t need to be cropped or require any further changes. However, some images need to be flipped horizontally, and some need to be rescaled, which can be accomplished through keras preprocessing (ImageDataGenerator). In order to normalize the images' pixels, we need to rescale the RGB coefficients to be in the range of 0 and 1. 

# Write-Up

## Introduction

Our group stumbled upon the FER-2013 dataset as we were searching for ideas for the final project on Kaggle and other dataset websites. We wanted to choose a dataset that allowed us to utilize convolutional layers and accurately classify some information. We found this dataset to be cool because it gives us a baby-introduction to how facial recognition works on our phones and other applications. Having a model that accurately classifies images with certain emotions, features, age, race, and etc is important as it can help improve facial recognition further and potentially catch criminals from an existing database. 

![image](https://user-images.githubusercontent.com/82127623/205781977-7bb5af90-6261-40ae-8ced-cc6c81751df2.png)

Figure 1: Examples of images associated with fear

![image](https://user-images.githubusercontent.com/82127623/205782283-ee7f3f80-7913-4174-8f93-76fe9a94574a.png)

Figure 2: Examples of images associated with happy


## Methods

### Data Exploration

----- empty -----

### Preprocessing

----- empty -----

### Model 1

----- empty -----

### Model 2

----- empty -----

## Results

----- empty -----

## Discussion

----- empty -----

## Conclusion

----- empty -----

## Collaboration 

Alan Chuang: 

Daksh Jain: I was responsible partially for coding, debugging, and communicating with team members to setup meetings and manage deadlines. I also worked on the write up introduction and ____ section. We split tasks evenly and helped eachother when needed. I believe everyone in the group worked hard and together as a team!

Sohum Goel: 

Grisha Bandodkar: 

Arnav Rastogi: 
