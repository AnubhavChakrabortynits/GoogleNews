from joblib import load
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet




def setup_nltk():
    import nltk
    nltk.download('wordnet')
    return 'NLTK resources downloaded successfully.'

def handle_country_abbr(text):
    country_abbreviations = {
    'U.S.A.': 'United States',
    'USA': 'United States',
     'US': 'United States' , 
    'UK': 'United Kingdom',   
    'U.K.': 'United Kingdom',
    'AUS.': 'Australia',
     'GBR': 'United Kingdom',
     'AUS' : 'Australia',
     'FRA' : 'France',
     'DEU' : 'Germany',
     'JPN' : 'Japan', 
    'CHN' : 'China',
    'IND' : 'India',
    'IN': 'India',   
    'BRA' : 'Brazil',
    'KOR': 'Korea' ,
     'MEX' : 'Mexico',
    'ITA' : 'Italy',
    'ESP' : 'Spain',
    'RUS' : 'Russia',
    'ZAF' : 'South Africa',
    'ARG' : 'Argentina',
    'CAN' : 'Canada',
    'NLD' : 'Netherlands',
    'SWE' : 'Sweden',
    'NZL': 'New Zealand',
    'CHE': 'Switzerland',
    'AUT': 'Austria',
    'NOR': 'Norway',
    'DNK': 'Denmark',
    'POL': 'Poland',
    'BEL': 'Belgium',
    'TUR': 'Turkey',
    'THA': 'Thailand',
    'ISR': 'Israel',
    'GRC': 'Greece',
    'IRL': 'Ireland',
    'SGP': 'Singapore',
    'EGY': 'Egypt',
    'SAU': 'Saudi Arabia',
    'MYS': 'Malaysia',
    'NGA': 'Nigeria',
    'ZWE': 'Zimbabwe',
    'COL': 'Colombia',
    'PER': 'Peru',
    'PRT': 'Portugal',
    'CZE': 'Czech Republic',
    'HUN': 'Hungary',
    'FIN': 'Finland',
    'IDN': 'Indonesia',
    'VNM': 'Vietnam',
    'PHL': 'Philippines',
    'IRN': 'Iran',
    'UKR': 'Ukraine',
    'KEN': 'Kenya',
    'CHL': 'Chile',
    'PAK': 'Pakistan',
    'MAR': 'Morocco',
    'SUI': 'Switzerland'  ,
    'GHA': 'Ghana',
    'UGA': 'Uganda',
    'MNG': 'Mongolia',   
    'LBN': 'Lebanon',
    'VEN': 'Venezuela',
    'LUX': 'Luxembourg',    
}   
    prevtext = text
    if text != 'us' and text != 'in':
        text = text.upper()
    if text not in country_abbreviations:
        return prevtext
    return country_abbreviations[text] 

def change_abbr(val):
    tempval = val.split(' ')
    for i in range(len(tempval)):
        tempval[i] = handle_country_abbr(tempval[i])
    val = ' '.join(tempval)
    return val

def tokenize(text):
    tokenized_words = word_tokenize(text)
    return tokenized_words

def remove_unwanted_chars(text):  
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b','',text)
    text = re.sub(r'\bhttps?://[^\s]+|\bwww\.[^\s]+','', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^\w]+',' ',text)
    pattern = re.compile('[\U0001F600-\U0001F64F]|[\U0001F300-\U0001F5FF]|[\U0001F680-\U0001F6FF]|[\U0001F700-\U0001F77F]|[\U0001F780-\U0001F7FF]|[\U0001F800-\U0001F8FF]|[\U0001F900-\U0001F9FF]|[\U0001FA00-\U0001FA6F]|[\U0001FA70-\U0001FAFF]|[\U00002702-\U000027B0]|[\U000024C2-\U0001F251]|[\U0001F004-\U0001F0CF]')
    text = pattern.sub(' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text
    
def remove_stop_words(val):
    stop_words = set(stopwords.words("english"))
    filtered_list = []
    for word in val:
        if word.casefold() not in stop_words:
            filtered_list.append(word)
    return filtered_list

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    
def lemmatize(val):
    lemmatizer = WordNetLemmatizer()
    val = nltk.pos_tag(val)
    lemmatized_words = [lemmatizer.lemmatize(word[0],get_wordnet_pos(word[1])) for word in val]
    return lemmatized_words

def make_sentence(val):
    return ' '.join(val)


def predict(text,model,vectorizer):
    cleaned_abbr = [change_abbr(string) for string in text]
    cleaned_array = [remove_unwanted_chars(string) for string in cleaned_abbr]
    processed_array = [tokenize(string) for string in cleaned_array]
    processed_array = [remove_stop_words(string) for string in processed_array]
    lemmatizer = WordNetLemmatizer()
    lemmatized_array = [[lemmatizer.lemmatize(string)  for string in val] for val in processed_array]
    sentences = [make_sentence(val) for val in lemmatized_array]
    print(sentences)
    tfidf_val = vectorizer.transform(sentences)
    proba_predictions = [model.predict_proba(value) for value in tfidf_val]
    
    proba_results = []
    for i in range(len(proba_predictions)):
        keys = ['Business','Entertainment','Headlines','Health','Science','Sports','Technology','Worldwide']
        proba_dict = dict(zip(keys,proba_predictions[i][0]))
        proba_dict = dict(sorted(proba_dict.items(), key=lambda item: item[1],reverse=True))
        proba_results.append([sentences[i],proba_dict])
            
    return proba_results