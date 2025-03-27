from os import walk
import json
import re
from tqdm import tqdm

import score_config as config

sys.path.append("Novelty_Scoring")
from utils import data_analysis, pmi, pmi_to_dict, docs_distribution, new_distribution, get_info, get_new_ingr
from Scoring import compute_scores


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


if __name__ == "__main__":

    filenames = list(set(next(walk(save_path), (None, None, []))[2]))
    parser = argparse.ArgumentParser(description="Generating Novelty and Surprise scoring for all variation recipes")

    parser.add_argument("--root", default=config.path["root"], help="ROOT Directory where model is located")
    parser.add_argument("--gf_path", default=config.path["gf_path"], help="File path to folder of json files for each recipes")

    
    for k in tqdm(range(len(filenames))):
        recette = filenames[k]
        file_path = my_path + recette
        with open(file_path) as json_file:
            recipe_dict = json.load(json_file)
            
        ## We set to 0 the distance here for each recipe -- difference needs to estimate distance between all points. 
        #This serves as optim to not calculate for each varaitions but only once since it is the same distance for all KB
        neighboroud_distance  = 0. 
        
        #### Skipping errors if there are no reference recipes in the dataset
        KB_recettes, _ = data_analysis(recipe_dict)
        if len(KB_recettes) <= 0:
            continue

        ### Transforming KB into distribution
        KB_texts = ' '.join(KB_recettes).split()
        EB_PMI = pmi(KB_texts)
        dict_know_pmi = pmi_to_dict(EB_PMI)

        train_recettes, train_indexes  = data_analysis(recipe_dict, ref=False, col_name='Train_Variations')
        valid_recettes, valid_indexes  = data_analysis(recipe_dict, ref=False, col_name='Valid_Variations')
        test_recettes, test_indexes  = data_analysis(recipe_dict, ref=False, col_name='Test_Variations')

        recette_variations = train_recettes + valid_recettes + test_recettes
        KB_matrix,  KB_dist, Count_matrix = docs_distribution(KB_recettes, recette_variations)
        KB_size = list(range(KB_matrix.shape[0]))

        ### Before computing novelty scores - initializing information for metadata
        KB_country = recipe_dict['Country']
        recipe_country.append(KB_country)
        #Nb of variation names used in the data collection
        nb_names.append(len(recipe_dict['Name variations'].split(' | ')))
    
        KB_ingr_list = recipe_dict['Reference_Base']['AllIngredients']
        KB_ingr_size.append(len(KB_ingr_list))


        ########### Train recipes
        country_list_train, ingr_list_train, train_raw_len, train_clean_len, train_raw_uniq, train_clean_uniq = get_info(train_indexes, recipe_dict, KB_country, index_name='Train_Variations')
        new_ingr_train = get_new_ingr(ingr_list_train, KB_ingr_list)
        for i in range(len(train_recettes)):
            select_variation = KB_size + [len(KB_size)+i]
            NewKB_dist, variation_dist = new_distribution(Count_matrix, select_variation)

            KB_updated = [train_recettes[i]]
            updated_text = ' '.join(KB_updated).split()
            New_EB_PMI = pmi(updated_text)

            results_train = compute_scores(KB_matrix, KB_dist, NewKB_dist, variation_dist, EB_PMI, dict_know_pmi, New_EB_PMI, config.types, config.thrs)
            
            current_index = train_indexes[i]
            for key, value in results_train.items():
                recipe_dict['Train_Variations'][current_index][key] = value
            #Computing metadata
            recipe_dict['Train_Variations'][train_indexes[i]]['nb_ingredients'] = len(ingr_list_train[i])
            recipe_dict['Train_Variations'][train_indexes[i]]['nb_new_ingredients'] = new_ingr_train[i]
            recipe_dict['Train_Variations'][train_indexes[i]]['text_length'] = train_raw_len[i]
            recipe_dict['Train_Variations'][train_indexes[i]]['clean_text_length'] = train_clean_len[i]
            recipe_dict['Train_Variations'][train_indexes[i]]['nb_uniq_tokens'] = train_raw_uniq[i]
            recipe_dict['Train_Variations'][train_indexes[i]]['clean_nb_uniq_tokens'] = train_clean_uniq[i]

        ######## Validation
        country_list_valid, ingr_list_valid, valid_raw_len, valid_clean_len, valid_raw_uniq, valid_clean_uniq = get_info(valid_indexes, recipe_dict, KB_country, index_name='Valid_Variations')
        new_ingr_valid = get_new_ingr(ingr_list_valid, KB_ingr_list)
        for i in range(len(valid_recettes)):
            select_variation = KB_size + [(len(KB_size)+len(train_recettes))+i]
            NewKB_dist, variation_dist = new_distribution(Count_matrix, select_variation)
    
            KB_updated = [valid_recettes[i]]
            updated_text = ' '.join(KB_updated).split()
            New_EB_PMI = pmi(updated_text)
            
            #Computing novelty scores
            results_valid = compute_scores(KB_matrix, KB_dist, NewKB_dist, variation_dist, EB_PMI, dict_know_pmi, New_EB_PMI, config.types, config.thrs)

            current_index = valid_indexes[i]
            for key, value in results_valid.items():
                recipe_dict['Valid_Variations'][current_index][key] = value

            recipe_dict['Valid_Variations'][valid_indexes[i]]['nb_ingredients'] = len(ingr_list_valid[i])
            recipe_dict['Valid_Variations'][valid_indexes[i]]['nb_new_ingredients'] = new_ingr_valid[i]
            recipe_dict['Valid_Variations'][valid_indexes[i]]['text_length'] = valid_raw_len[i]
            recipe_dict['Valid_Variations'][valid_indexes[i]]['clean_text_length'] = valid_clean_len[i]
            recipe_dict['Valid_Variations'][valid_indexes[i]]['nb_uniq_tokens'] = valid_raw_uniq[i]
            recipe_dict['Valid_Variations'][valid_indexes[i]]['clean_nb_uniq_tokens'] = valid_clean_uniq[i]
            
        ############## Test
        country_list_test, ingr_list_test, test_raw_len, test_clean_len, test_raw_uniq, test_clean_uniq = get_info(test_indexes, recipe_dict, KB_country, index_name='Test_Variations')
        new_ingr_test = get_new_ingr(ingr_list_test, KB_ingr_list)
        for i in range(len(test_recettes)):

            select_variation = KB_size + [(len(KB_size)+len(train_recettes)+len(valid_recettes))+i]
            NewKB_dist, variation_dist = new_distribution(Count_matrix, select_variation)
    
            KB_updated = [test_recettes[i]]
            updated_text = ' '.join(KB_updated).split()
            New_EB_PMI = pmi(updated_text)
    
            results_test = compute_scores(KB_matrix, KB_dist, NewKB_dist, variation_dist, EB_PMI, dict_know_pmi, New_EB_PMI, config.types, config.thrs)

            current_index = test_indexes[i]
            for key, value in results_test.items():
                recipe_dict['Test_Variations'][current_index][key] = value
            recipe_dict['Test_Variations'][test_indexes[i]]['nb_ingredients'] = len(ingr_list_test[i])
            recipe_dict['Test_Variations'][test_indexes[i]]['nb_new_ingredients'] = new_ingr_test[i]
            recipe_dict['Test_Variations'][test_indexes[i]]['text_length'] = test_raw_len[i]
            recipe_dict['Test_Variations'][test_indexes[i]]['clean_text_length'] = test_clean_len[i]
            recipe_dict['Test_Variations'][test_indexes[i]]['nb_uniq_tokens'] = test_raw_uniq[i]
            recipe_dict['Test_Variations'][test_indexes[i]]['clean_nb_uniq_tokens'] = test_clean_uniq[i]

        file_name = save_path + recette
        with open(file_name, "w") as outfile:
            json.dump(recipe_dict, outfile)

        del recipe_dict