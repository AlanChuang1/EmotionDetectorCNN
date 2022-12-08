# Emotion Prediction Using CNN - Group 4 - ECS 171 FQ 2022

Group Project for ECS 171, Fall Quarter 2022, at UC Davis under Dr. Solares.

Below is a link to our entire project notebook in Google Colab. We also have code blocks for certain sections located throughout this write-up

[![Open Entire Project Notebook In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1mFxnk88tvQjZwlSB8vthuq_pnYSQRe-D?usp=sharing)


## Group Members:

Here is a list of our group members, with their names, email addresses and GitHub usernames:

| Name | Email | GitHub Username |
| ------------- | ------------- | ------------- |
| Alan Chuang  | afchuang@ucdavis.edu | AlanChuang1 |
| Daksh Jain  | dajain@ucdavis.edu  | DakshJ4033 |
| Sohum Goel  | sohgoel@ucdavis.edu | SohumGoel |
| Arnav Rastogi  | arnrastogi@ucdavis.edu | Arnav33R |
| Grisha Bandodkar | gbandodkar@ucdavis.edu | grishaab |

# Write-Up

## Introduction

Our group stumbled upon the FER-2013 dataset as we were searching for ideas for the final project on Kaggle and other dataset websites. Since we enjoyed working with Neural Networks, we wanted to choose a dataset that allowed us to utilize convolutional layers and in doing so, accurately classify some information. When we came across this dataset, we were excited to use it for our final project because it would give us an introduction to how facial recognition works on our phones and other applications. Having a model that accurately classifies images with certain emotions, features, age, race, and etc is important as it can help improve facial recognition further and has many useful and applicable applications - such as, potentially catching/indetifying criminals from an existing database. 

Thus, the aim of our project is to be able to predict the emotion that best fits a given facial expression. Although there is a plethora of possible emotions, our project focuses on a few that are the most basic human expressions/emotions. These are categorized as Angry, Fear, Happy, Neutral, Sad, Surprise and Disgust. 

[Our Dataset](https://www.kaggle.com/datasets/msambare/fer2013)

This dataset contains images of faces that are 48x48 pixel grayscale and centered, thus occupying almost the same amount of space for each image, making recognition easier. Here are some examples of images contained in the dataset:


![image](https://user-images.githubusercontent.com/82127623/205781977-7bb5af90-6261-40ae-8ced-cc6c81751df2.png)

Figure 1: Examples of images associated with the emotion fear

![image](https://user-images.githubusercontent.com/82127623/205782283-ee7f3f80-7913-4174-8f93-76fe9a94574a.png)

Figure 2: Examples of images associated with the emotion happiness


## Methods

### Data Exploration

There are 4,254 observations in the dataset, with 1774 happy observations, 1233 neutral observations, and 1247 sad observations. Each “observation”, or image file, is a 48x48 pixel-sized grayscale image of expressions on faces. Sizes are all standardized, so they don’t need to be cropped or require any further changes. However, some images need to be flipped horizontally, and some need to be rescaled, which can be accomplished through keras preprocessing (ImageDataGenerator). In order to normalize the images' pixels, we need to rescale the RGB coefficients to be in the range of 0 and 1. 

----- empty -----

### Preprocessing

The "disgust" directory is deleted as it represents only a tiny portion of the entire dataset. This somewhat reduces the training time for the model.

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

Sohum Goel: I was responsible for the initial setup of the neural network, along with some trial runs. I also communicated with the team to set up meetings, and all of us split tasks evenly and added to each other's work. Everyone worked together to complete this project.

Grisha Bandodkar: 

Arnav Rastogi: 
