import pandas as pd
import ast
from collections import Counter
import re
from sklearn.model_selection import train_test_split
import json
from tqdm import tqdm
import sys

import data_config as config

sys.path.append("GlobalFusion_Generation")
from KBEB_extraction import ID_linkages
from json_dataset import dataset_generation


import argparse
import logging


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "True", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "False","f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def tokenize(text):
    return [tok.text for tok in nlp.tokenizer(str(text))]

def country_matching(title_list, state_list = ['new jersey', 'new england', 'louisianna', 'kentucky', 'southern', 'maine', 'maryland', 'pennsylvania', 'new orleans'])
    
    country_flag = []
    for i, title in enumerate(title_list):
        title_toks = tokenize(str(title).lower())
        natio_interesect = list(set(title_toks) & set(low_natio_list))
        country_interesect = list(set(title_toks) & set(low_country_list))
        if len(natio_interesect) > 0:
            nationality = natio_interesect[0]
            index = low_natio_list.index(nationality)
            country = country_list[index].lower()
        elif len(country_interesect) > 0:
            country = country_interesect[0]
            if country in state_list:
                country = "united states"
            if country == 'jersey':
                country = "united kingdom"
        else:
            country = '<UNK>'
        
        country_flag.append(country)

    return country_flag

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generating GlobalFusion Dataset for Cultural Novelty Evaluation")

    parser.add_argument("--root", default=config.path["root"], help="ROOT Directory where model is located")
    parser.add_argument("--selected_recipes", default=config.path["selected_recipes"], help="File path the manually currated list of selected recipes")
    parser.add_argument("--recipes_paired_ids", default=config.path["recipes_paired_ids"], help="File path to the matching IDs for recipes in NLG Recipes")
    parser.add_argument("--countries", default=config.path["countries"], help="File path to country list and matching nationalties")
    parser.add_argument("--recipes_train", default=config.path["recipes_train"], help="File path to final split for generating GlobalFusion")
    parser.add_argument("--recipes_valid", default=config.path["recipes_valid"], help="File path to final split for generating GlobalFusion")
    parser.add_argument("--recipes_test", default=config.path["recipes_test"], help="File path to final split for generating GlobalFusion")
    parser.add_argument("--nlg", default=config.path["nlg"], help="File path to NLG Recipes -- source file")
    parser.add_argument("--nlg_known", default=config.path["nlg_known"], help="File path to NLG Recipes with paired countries added")
    
    parser.add_argument("--KownCountries", default=config.steps["KownCountries"], type=str2bool, help="Does the file nlg_knwon exists")
    parser.add_argument("--split_know", default=config.steps["split_know"], type=str2bool, help="Does the files nlg_knwon train,valid,test exist")
    parser.add_argument("--ID_pairing", default=config.steps["ID_pairing"], type=str2bool, help="Does the files nlg_knwon train,valid,test exist")

    ########################## IMPORTING FILES ##########################
    df_recipes = pd.read_csv(args.nlg)
    title_list = list(df_recipes['title'])
    df_names = pd.read_excel(args.selected_recipes)
    
    ########################## CHECKING IF THE COUNTRY MATCHING AND THE SPLITS ALREADY EXISTS ##########################
    if not args.split_know: 
        ########################## CHECKING IF THE COUNTRY MATCHING ALREDY EXISTS ##########################
        if not args.KownCountries:
            df_countries = pd.read_csv(args.nlg)
            country_list = list(df_countries['Name'])
            natio_list = list(df_countries['Nationality'])
            
            #Lower_casing all countries
            low_country_list = []
            low_natio_list = []
            for i in range(len(country_list)):
                low_country_list.append(country_list[i].lower())
                low_natio_list.append(natio_list[i].lower())
        
            country_matched = country_matching(title_list)
            df_known = df_recipes[df_recipes['countries'] != '<UNK>']
            df_known.to_csv(args.nlg_known, index=False)
        else:
            df_known = pd.read_csv(args.nlg_known)
            
            df_train, temp_df = train_test_split(df_known, test_size=0.3, random_state=42)
            df_valid, df_test = train_test_split(temp_df, test_size=0.5, random_state=42)
            df_train.to_csv(args.recipes_train, index=False)
            df_valid.to_csv(args.recipes_valid, index=False)
            df_test.to_csv(args.recipes_test, index=False)
    else: 
        df_train = pd.read_csv(args.recipes_train, index=False)
        df_valid = pd.read_csv(args.recipes_valid, index=False)
        df_test = pd.read_csv(args.recipes_test, index=False)

    ########################## CHECKING IF THE IDS PAIRING between names and NLG is exisitng ##########################
    if not args.ID_pairing:
        collect_KB = ID_linkages(df_names)
        df_names_long = collect_KB.ID_list_longKB(df_train)
        df_names_long_trainvar = collect_KB.train_variations(df_names_long, df_train)
        df_names_long_validvar = collect_KB.test_variations(df_names_long_trainvar, df_valid, col_name='valid_variations')
        df_names_long_testvar = collect_KB.test_variations(df_names_long_validvar, df_test, col_name='test_variations')

        df_names_long_testvar.to_csv(args.recipes_paired_ids,index=False)
    else:
        df_names_long_testvar = pd.read_csv(args.recipes_paired_ids, index=False)

    ############## Generating the GlobalFusion Dataset
    gen_instance = dataset_generation(df_names_long_testvar, df_train, df_valid, df_test, lemma="True", authrorized_pos= ['PROPN', 'PRON', 'ADJ', 'ADV', 'NOUN', 'NUM', 'VERB'])
    gen_instance.json_create(file_path=args.save_path)