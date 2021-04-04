# Reci-py Recommender

Reci-py Recommender is a recipe recommendation system with the purpose of reducing the amount of food waste. So the user can put list of ingredients to use and the recommender system will recommend a user few recipes that contains most of ingredients given by the user.

## Method Used
* Data Visualization
* Content based filtering system
* Collaborative filtering system

## Technologies
* Python
* Pandas
* Jupyter
* Numpy
* Matplotlib
* SpaCy
* Gensim
* etc.

## Project Description
The data was downloaded from [kaggle.com](https://www.kaggle.com/shuyangli94/food-com-recipes-and-user-interactions), which collected data from food.com. From the dataset, only the raw data were used. Data was processed using Gensim and SpaCy for tokenization and text cleaning. The goal for this project is to recommend recipes to users based on the list of ingredients they provide, in order to reduce the amount of food waste. For the recommendation system, vectorized list of ingredients to find recipes with similar ingredients and used Surprise recommender for collaborative filtering system.

