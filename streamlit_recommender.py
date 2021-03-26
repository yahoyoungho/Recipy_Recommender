import streamlit as st
import numpy as np
import pandas as pd
import pickle
import sklearn as skl
import os

from gensim import similarities
import spacy
import os

import surprise
from surprise import SVD
from surprise import Dataset, Reader

######### intializing variables ###############
pp_recipes = pickle.load(open("src/data/pkls/recipe_ingrients_pp.pkl","rb"))
ingr_corpus = pickle.load(open("src/data/pkls/ingr_corpus.pkl","rb"))
ingr_dict = pickle.load(open("src/data/pkls/ingr_dictionary.pkl","rb"))
index = similarities.SparseMatrixSimilarity(ingr_corpus, num_features = len(ingr_corpus))
interaction_df = pd.read_csv("src/data/kaggle_food_data/RAW_interactions.csv")



rating_path = "src/data/user_rating_dir/user_ratings.csv"
alg = SVD(n_factors=100, n_epochs=20, lr_all= 0.005, reg_all= 0.025, random_state= 789,verbose=True)
curr_user_id = "special_user"


############# Functions #######################
def prep_ingr(ingredients):
    """preprocess formatting of the list of ingredients
    
    will remove string 'and' and '&' if present
    
    Args:
        ingredients (list of strings): list of ingredients
    
    Returns:
        list: list of formatted ingredients 
    """
    toreturn = []
    for ingr in ingredients:
        
        # remove 'and' or '&' if exsits
        if "and" in ingr or "&" in ingr:
            ingr = ingr.replace("and", "").replace("&","") #remove
            ingr = ingr.split(" ")
            # remove empty strings
            while "" in ingr:
                ingr.remove("")
                
            for i in ingr:
                toreturn.append(i)
        else:
            toreturn.append("_".join(ingr.split(" ")))
    return toreturn

def get_rating_prediction(recipe_ids, alg):
	toreturn = []
	for recipe_id in recipe_ids:
		toreturn.append(alg.predict(curr_user_id, recipe_id).est/5)

	return toreturn


def content_base_recommends(input_ingr,num_recs ,index = index, dct = ingr_dict, recipe_df = pp_recipes, alg = None):
    "returns certain amount of recipes that is similar to list of input ingredients"
    
    nlp = spacy.load("en_core_web_sm")
    ingrs = nlp(input_ingr)
    ingrs = " ".join([ingr.lemma_.lower()for ingr in ingrs])
    # format ingredients
    ingrs = prep_ingr(ingrs.split(" , "))
    
    # format ingredients in bow
    ingrs_bow = dct.doc2bow(ingrs)
    
    # get the n_closest recipes
    # if alg is not None use it for rating prediction
    if alg:
    	toreturn = recipe_df.iloc[index[ingrs_bow].arg_sort()[-num_recs:]]
    	recipe_ids = recipe_df.iloc[index[ingrs_bow].arg_sort()[-num_recs:]]["id"].values.tolist()
    	scaled_ratings = get_rating_prediction(recipe_ids, alg)
    	toreturn["pref_ratings"] = scaled_ratings # create new column to rank depending on user preference
    	toreturn.sort_values("pref_ratings", ascending = False, inplace = True) # rank by preference
    	toreturn.drop(columns = ["pref_ratings"], inpalce = True) # drop the column for better visualization
    	return toreturn

    return recipe_df.iloc[index[ingrs_bow].argsort()[-num_recs:]]
    
def create_new_interaction_df(new_df, inter_df = interaction_df):
	inter_df = inter_df.copy()
	"""append user interaction to pre-existing df
	
	Args:
	    inter_df (pd.DataFrame, optional): df to append to
	    new_df (pd.DataFrame): df to append
	
	Returns:
	    pd.DataFrame: appended dataframe with index resetted
	"""
	inter_df = inter_df.append(new_df)
	inter_df.reset_index(drop=True, inplace = True)
	return inter_df

def check_filled(file=rating_path):
	"""Check if the file is empty
	
	Args:
	    file (string, optional): path to the file
	
	Returns:
	    bool : returns True if the file is not empty
	"""
	with open(file,"r") as f:
		if len(f.readlines()):
			return True
	return False



################## Webapp main interface #######################
can_update_ratings = False
if check_filled():
	f = open(rating_path,"r")
	if len(f.readlines()) >5:
		can_update_ratings = True
rating_is_empty = check_filled(rating_path) # True if the file is filled
pred_alg = None
if can_update_ratings: # if it can update load the prediction alg
	pred_alg = pickle.load(open("src/data/user_rating_dir/collaborative_algorithm.pkl","rb"))


st.title("Welcome to Reci-Py Recommender!")

st.header("This recommender will give you recipe recommendation depending on what ingredients you put in!")
st.text("Please type in the list of ingredients you want to use separated by comma!")
ingredients = st.text_input(label = "Ingredients input")

st.text("Please select how many recommendations you want to get! (max: 20)")
n_recs = st.number_input(label = "Number of recipes to recieve", min_value = 2, max_value = 20, value = 5)
n_recs = int(n_recs)

# getting recipe recommendations
if st.button("Get recommendations"):
	# if the input is too small don't run
	if ingredients: # if valid input
		recs = content_base_recommends(ingredients, n_recs, pred_alg)["name id steps description ingredients".split()]
		st.write(recs)

		# get the rating of the rceipe
		st.write("What recipe did you choose?")
		
		ids = recs["id"].values.tolist()

		selected_id = st.select_slider(label= "Select the id of the recipe",options = ids)
		
		st.write("Please give the rating for the selected recipe from 0 to 5.")
		st.write("0 being the worst and 5 being the best.")
		
		recipe_rating_by_user = st.number_input("Rating of the recipe", min_value = 0,\
			max_value = 5, value = 0, step = 1)
		
		user_data = {"user_id": curr_user_id,
			"recipe_id": selected_id,
			 "date":None, "rating":recipe_rating_by_user,
			  "review":None}
		collected_data_df = pd.DataFrame(user_data, index = [0])
		

		if st.button("Save Rating"):# check if custom user rating already exists
			if check_filled(): # if there is information already recorded
				# merge and save
				collected_data_df = create_new_interaction_df(collected_data_df, pd.read_csv(rating_path))
				collected_data_df.to_csv(rating_path, index = False)
			else:	
				# just save
				collected_data_df.to_csv(rating_path, index = False)


	else:
		st.write("Please have more ingredients in the input!")

st.text("This will update your preference and chagne the outcome of the recommendations.")
st.markdown("_this will work after you provide some ratings to recipes you have recieved_")

# updating preference
if st.button("Update my perference"): # will run collaborative filtering system
	# first check if the number of rating is sufficient
	if check_filled():
	# if not enough rating don't run
		if len(user_pref_df) < 5:
			st.write("Not enough ratings. Please provide more ratings to update your preference.")
			st.write(f"current number of ratings: {len(user_pref_df)}. {5-len(user_pref_df)} needed.")
		else: #if there are enough number of ratings given by the user
			st.write("Updating the preference. This will take a little bit of time.")
			# training SVD for collaborative filtering
			col_alg = SVD(n_factors = 100, n_epochs = 20, lr_all = 0.005, reg_all = 0.025, random_state = 789)
			reader = Reader(rating_scale = (0,5))
			surprise_dataset = Dataset.load_from_df(create_new_interaction_df(pd.read_csv(rating_path))["user_id recipe_id rating".split(" ")], reader  )
			col_alg.fit(surprise_dataset.uild_full_trainset())
			# save prediction model
			pickle.dump(col_alg, "src/data/user_rating_dir/collaborative_algorithm.pkl")
			st.write("Preference updated. Your preference will be used in recommending from next use.")

	else:
		st.write("not enough rating is provided")
		

	
