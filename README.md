# Facial Expression Prediction Using CNN - Group 4 - ECS 171 FQ 2022

Group Project for ECS 171, Fall Quarter 2022, at UC Davis under Dr. Solares.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1mFxnk88tvQjZwlSB8vthuq_pnYSQRe-D?usp=sharing)


## Group Members:

| Member Name  | GitHub Username |
| ------------- | ------------- |
| Alan Chuang  | AlanChuang1  |
| Daksh Jain  | Content Cell  |
| Sohum Goel  | SohumGoel  |
| Arnav Rastogi  | Arnav33R |
| Grisha Bandodkar  | grishaab |


## Introduction

Our group stumbled upon the FER-2013 dataset as we were searching for ideas for the final project on Kaggle and other dataset websites. Since we enjoyed working with Neural Networks, we wanted to choose a dataset that allowed us to utilize convolutional layers to accurately classify information. We were excited to use it for our final project because it would give us an introduction to facial recognition on our phones and other applications. Having a model that accurately classifies images with certain emotions, features, age, race, etc is important as it can help improve facial recognition further and has many useful and applicable applications - such as, potentially catching/identifying criminals from an existing database.

Thus, the aim of our project is to be able to predict the emotion that best fits a given facial expression. Although there is a plethora of possible emotions, our project focuses on a few that are the most basic human expressions/emotions. These are categorized as Angry, Fear, Happy, Neutral, Sad, Surprised, and Disgust.

Dataset: https://www.kaggle.com/datasets/msambare/fer2013 

## Methods

### Data Exploration

There are 4,254 observations in the dataset, with 1774 happy observations, 1233 neutral observations, and 1247 sad observations. Each “observation”, or image file, is a 48x48 pixel-sized grayscale image of expressions on faces. Sizes are all standardized, so they don’t need to be cropped or require further changes. However, some images need to be flipped horizontally, and some need to be rescaled, which can be accomplished through Keras preprocessing (ImageDataGenerator). In order to normalize the images' pixels, we need to rescale the RGB coefficients to be in the range of 0 and 1.

### Data Evaluation

Here we are creating a dataframe by going through the directory and setting each image with its corresponding expression and storing that in a pandas dataset which is going to be returned by the function. We use the returned dataset to display the number of samples for each expression in our train and test dataset.  


```python
train_dir = './train/'
test_dir = './test/'

# image size
row, col = 48, 48
# number of image classes: angry, sad, etc
classes = 7

def count_exp(path, set_):
    dict_ = {}
    for expression in os.listdir(path):
        dir_ = path + expression
        dict_[expression] = len(os.listdir(dir_))
    df = pd.DataFrame(dict_, index=[set_])
    return df

# number of observations
train_count = count_exp(train_dir, 'train')
test_count = count_exp(test_dir, 'test')
print(train_count)
print(test_count)
```

Here we go through our directory again and plot an example image with the corresponding expression using our test data for the y-axis for simplicity. 

```python

test_count.transpose().plot(kind='bar',figsize=(12, 10))

plt.figure(figsize=(14,22))
i = 1
for expression in os.listdir(train_dir):
    img = load_img((train_dir + expression +'/'+ os.listdir(train_dir + expression)[1]))
    plt.subplot(1,7,i)
    plt.imshow(img)
    plt.title(expression)
    plt.axis('off')
    i += 1
plt.show()

```

### Preprocessing
We now move on to preprocessing our data by eliminating a class and reformatting the structure of all images.
We remove disgust from the dataset. Then we create a single data frame consisting of all the images with their corresponding class (label).


```python

shutil.rmtree( './train/disgust')
shutil.rmtree( './test/disgust')

# train dataset
images = []
labels = []
for subset in os.listdir(train_dir):
  image_list = os.listdir(os.path.join(train_dir,subset))  # all the names of images in the directory
  image_list = list(map(lambda x:os.path.join(subset,x),image_list))
  images.extend(image_list)
  labels.extend([subset]*len(image_list)) 

df = pd.DataFrame({"Images":images,"Labels":labels})      
df = df.sample(frac=1).reset_index(drop=True) # this will shuffle the data
samplesize = int(int(df.size)/14)  # sample size used for modelling 
print(samplesize)
df_train = df.head(samplesize)  

```



