import json
from tqdm import tqdm
import pandas as pd
from collections import Counter
import re
import ast
import spacy

class dataset_generation():

    def __init__(df_names_full, df_train, df_valid, df_test, lemma="True", authrorized_pos= ['PROPN', 'PRON', 'ADJ', 'ADV', 'NOUN', 'NUM', 'VERB']):
        self.nlp = spacy.load("en_core_web_sm")
        self.lemma = lemma
        self.authrorized_pos = authrorized_pos

        ##Datasets
        self.df_names_full = df_names_full
        self.df_train = df_train
        self.df_valid = df_valid
        self.df_test = df_test
        
    def getting_lists(self, string_list):

        list_ = []
        for elem in string_list:
            list_.append(ast.literal_eval(elem))
    
        assert len(list_) == len(string_list)
        
        return list_

    def text_cleaning(self, recipe_list):

        clean_list = []
        for i in tqdm(range(len(recipe_list))):
            recipe_doc = nlp(str(recipe_list[i]))
            temp_list = []
            for token in recipe_doc:
                if token.pos_ in authrorized_pos:
                    if not token.is_stop:
                        if self.lemma:
                            temp_list.append(token.lemma_)
                        else:
                            temp_list.append(token)
            clean_text = ' '.join(temp_list)
            clean_list.append(clean_text)
        
        return clean_list

    def recipe_gathering(self, id_list, df_, mode="var"):
        """
        Function to get the text from id_list
        """
        if mode == "KB": 
            recipe_list, ingr_list = [], []
            for id in id_list:
                df_id = df_[df_['Unnamed: 0'] == id]
                recipe_list.append(df_id['directions'])
                temp_list = ast.literal_eval(df_id['NER'])
                ingr_list.extend(temp_list)
    
            ingr_list_f = list(set(ingr_list)) ## We just need to keep the unique list of all ingredient ever encountered for the KB
            
            return recipe_list, ingr_list
        else: 
            recipe_list, country_list, ingr_list = [], [], []
            for id in id_list:
                df_id = df_[df_['Unnamed: 0'] == id]
                recipe_list.append(df_id['directions'])
                country_list.append(df_id['countries'])
                ingr_list.append(df_id['NER'])  ## We append string here for now
    
            return recipe_list, country_list, ingr_list

    def get_ids(self):

        string_KB = list(self.df_names_full['ids_LongList'])
        self.list_KB = self.getting_lists(string_KB)
        string_train_var = list(self.df_names_full['train_variations'])
        self.list_train_var = self.getting_lists(string_train_var)
        string_valid_var = list(self.df_names_full['valid_variations'])
        self.list_valid_var = self.getting_lists(string_valid_var)
        string_test_var = list(self.df_names_full['test_variations'])
        self.list_test_var = self.getting_lists(string_test_var)

    def get_elems(self):
        self.recipe_KB, self.ingr_KB = self.recipe_gathering(self.list_KB, self.df_train, mode="KB")  ## The KB is formed with the trained dataset only
        self.recipe_train, self.country_train, self.ingr_train = self.recipe_gathering(self.list_train_var, self.df_train)
        self.recipe_valid, self.country_valid, self.ingr_valid = self.recipe_gathering(self.list_valid_var, self.df_valid)
        self.recipe_test, self.country_test, self.ingr_test = self.recipe_gathering(self.list_test_var, self.df_test)


    def get_clean_recipe(self):

        self.recipe_KB_clean = self.text_cleaning(self.recipe_KB)
        self.recipe_train_clean = self.text_cleaning(self.recipe_train)
        self.recipe_valid_clean = self.text_cleaning(self.recipe_valid)
        self.recipe_test_clean = self.text_cleaning(self.recipe_test)

    def json_create(self, file_path='.'):

        recipes_names = list(self.df_names_full['Short Name'])
        recipes_country = list(self.df_names_full['Country'])

        self.get_ids()
        self.get_elems()
        self.get_clean_recipe()
        
        count = 0
        print('============Starting saving procedure============')
        for i, name in enumerate(recipes_names):
            name_save = name.split(' | ')[0]
            
            #Instanciating the dictionnary to save the information regarding this recipe
            dict_ = {}
            dict_['Recipe_Name'] = name_save
            dict_['Name variations'] = name
            dict_['Country'] = recipes_country[i]
            dict_['Reference_Base'] = {}
            dict_['Reference_Base']['AllIngredients'] = ingr_KB
            for i, id_ in enumerate(list_KB):
                dict_['Reference_Base'][str(id_)]= {}
                dict_['Reference_Base'][str(id_)]['recipe_raw'] = recipe_KB[i]
                dict_['Reference_Base'][str(id_)]['recipe_clean'] = recipe_KB_clean[i]
            
            dict_['Train_Variations'] = {}
            for i, id_ in enumerate(list_train_var):
                dict_['Train_Variations'][str(id_)]= {}
                dict_['Train_Variations'][str(id_)]['recipe_raw'] = recipe_train[i]
                dict_['Train_Variations'][str(id_)]['recipe_clean'] = recipe_train_clean[i]
                dict_['Train_Variations'][str(id_)]['country'] = country_train[i]
                dict_['Train_Variations'][str(id_)]['ingredient_list'] = ingr_train[i]
    
            dict_['Valid_Variations'] = {}
            for i, id_ in enumerate(list_valid_var):
                dict_['Valid_Variations'][str(id_)]= {}
                dict_['Valid_Variations'][str(id_)]['recipe_raw'] = recipe_valid[i]
                dict_['Valid_Variations'][str(id_)]['recipe_clean'] = recipe_valid_clean[i]
                dict_['Valid_Variations'][str(id_)]['country'] = country_valid[i]
                dict_['Valid_Variations'][str(id_)]['ingredient_list'] = ingr_valid[i]
    
            dict_['Test_Variations'] = {}
            for i, id_ in enumerate(list_test_var):
                dict_['Test_Variations'][str(id_)]= {}
                dict_['Test_Variations'][str(id_)]['recipe_raw'] = recipe_test[i]
                dict_['Test_Variations'][str(id_)]['recipe_clean'] = recipe_test_clean[i]
                dict_['Test_Variations'][str(id_)]['country'] = country_test[i]
                dict_['Test_Variations'][str(id_)]['ingredient_list'] = ingr_test[i]
    
            ## Saving the JSON for each recipe :
            file_name = file_path + name_save +'recipe_{}.json'.format(i)
            with open(file_name, "w") as outfile:
                json.dump(dict_ = {}, outfile)
    
            del dict_
            count += 1
        
        print('============ALL {} FILES SAVED============'.format(count))