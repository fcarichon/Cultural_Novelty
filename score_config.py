
ROOT_DIR = '.'

path = {
    "root": ROOT_DIR,
    "gf_path": f"{ROOT_DIR}/datasets/GlobalFusion/Recipes",
    "save_path": f"{ROOT_DIR}/datasets/GlobalFusion/Recipes_with_scores/"
}

# Default thrsholds for going from Novelty scores to binary scores
# Estimated as average in Train datasets -- See file thrshold estimate for details
thrs = {"newness_div": 0.00092,
        "newness_prob":14.64,
        "new_extremes":0.0014,
        "novelty_new":0.0014,
        "newness_rank":10,
        "uniq_dist":0.527,
        "uniq_proto":0.00358,
        "diff_global":0.897,
        "diff_local":0.614,
        "neighbors":3,
        "new_surprise":0.0104,
        "dist_surprise":0.00256
       }

# Default option to estimate novelty and surprise
types = {"newness_div": True,
         "newness_prob": False,
         "new_extremes": False,
         "new_rank":False,
         "uniq_dist":True,
         "uniq_proto":False,
         "diff_global":False,
         "diff_local":True,
         "new_surprise":True,
         "dist_surprise":True}