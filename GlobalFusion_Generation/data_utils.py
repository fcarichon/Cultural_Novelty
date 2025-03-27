
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