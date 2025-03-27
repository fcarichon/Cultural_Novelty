
ROOT_DIR = '.'

path = {
    "root": ROOT_DIR,
    "selected_recipes": f"{ROOT_DIR}/datasets/GlobalFusion/Selected_recipes.xlsx",
    "recipes_paired_ids" : f"{ROOT_DIR}/datasets/GlobalFusion/RecipesNamesFull_withKBAndVar.csv"
    "countries": f"{ROOT_DIR}/datasets/countries.csv", 
    "recipes_train": f"{ROOT_DIR}/datasets/GlobalFusion/NLGRecipe_KownCountriesOnly_Train.csv", 
    "recipes_valid": f"{ROOT_DIR}/datasets/GlobalFusion/NLGRecipe_KownCountriesOnly_Valid.csv", 
    "recipes_test": f"{ROOT_DIR}/datasets/GlobalFusion/NLGRecipe_KownCountriesOnly_Test.csv", 
    "nlg": f"{ROOT_DIR}/datasets/GlobalFusion/full_dataset_NLGRecipe.csv",
    "nlg_known": f"{ROOT_DIR}/datasets/GlobalFusion/NLGRecipe_KownCountriesOnly.csv",
    "save_path": f"{ROOT_DIR}/datasets/GlobalFusion/Recipes"
}

#If the pre-treated files from Recipe NLG have already been generated
steps = {
    "KownCountries": True,
    "split_know": True,
    "ID_pairing":True
}