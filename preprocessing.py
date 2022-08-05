import collections
from operator import itemgetter

import nltk
from matplotlib import pyplot as plt, ticker
from nltk.corpus import stopwords
import matplotlib
from nltk import FreqDist, WordNetLemmatizer, ConditionalFreqDist
from nltk.draw import dispersion_plot
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from spellchecker import SpellChecker
from nltk import WhitespaceTokenizer

def preprocessing():

    #open text file you want to analyze
    f = open('teslafinance.txt', 'r', encoding='utf8')
    raw = f.read()

    #tokenize by words and make into nltk text
    tokens = nltk.word_tokenize(raw)
    text = nltk.Text(tokens)

    return text

#does all of the cleaning (punctuation and stopwords)
def get_cleared_text(text):
    cleared = lower_case(text)

    cleared = filter_punctuation(cleared)
    cleared = filter_stopwords(cleared)
    #cleared = spellchecker(cleared)
    print(type(cleared))

    #
    # f = open('file.txt', 'a', encoding='utf8')
    # f.write("\n")
    # for word in cleared:
    #     f.write(word + ' ')
    # f.close()
    return cleared

#tags Parts of Speech
def pos_tagger(cleared_text):
    result  = nltk.pos_tag(cleared_text)
    return result

def lower_case(cleared_text):
    cleared = []
    for word in cleared_text:
        cleared.append(word.lower())
    return cleared

#should do spellchecking but actually makes it worse
def spellchecker(cleared_text):
    result = []
    spell = SpellChecker()
    for word in cleared_text:
        correct_word = spell.correction(word)
        result.append(correct_word)

    return result



#capitalizes each word
def capitalize(cleared_text):
    cleared = []
    for word in cleared_text:
        cleared.append(word.capitalize())
    return cleared


#filters out anything not alphabetical
def filter_punctuation(nltk_text):
    text = [word.lower() for word in nltk_text if word.isalpha()]
    return text

#filters english stopwords
def filter_stopwords(list_to_be_cleared):
    stop_words = set(stopwords.words("english"))
    #stop_words = set(stopwords.words("english_reddit"))
    # empty list für das Ergebnis
    filtered_list = []

    for word in list_to_be_cleared:
        if word.casefold() not in stop_words:
            filtered_list.append(word)

    return filtered_list