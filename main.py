# This is a sample Python script.
import collections
from operator import itemgetter

import nltk
from matplotlib import pyplot as plt, ticker
from nltk.corpus import stopwords
import matplotlib
from nltk import FreqDist, WordNetLemmatizer
from nltk.draw import dispersion_plot
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from itertools import islice

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
#nltk.download('wordnet')
#nltk.download('omw-1.4')
#nltk.download('vader_lexicon')
#nltk.download('averaged_perceptron_tagger')
#nltk.download("maxent_ne_chunker")
#nltk.download("words")
from numpy import take


def preprocessing():


    #open text file you want to analyze
    f = open('allfinance.txt', 'r', encoding='utf8')
    raw = f.read()

    #tokenize by words and make into nltk text
    tokens = nltk.word_tokenize(raw)
    text = nltk.Text(tokens)

    return text

def get_cleared_text(text):
    cleared = filter_punctuation(text)
    cleared = filter_stopwords(cleared)
    return cleared

def pos_tagger(cleared_text):
    result  = nltk.pos_tag(cleared_text)
    return result

def capitalize(cleared_text):
    cleared = []
    for word in cleared_text:
        cleared.append(word.capitalize())
    return cleared

def extract_ne(cleared_list):
    tags = nltk.pos_tag(cleared_list)
    tree = nltk.ne_chunk(tags,binary=True)
    return set(
        " ".join(i[0] for i in t)
        for t in tree
        if hasattr(t, "label") and t.label() == "NE"
    )

def dispersion_plot_vanilla(nltk_text):
    words = ["good", "bad", "buy", "sell"]
    plt.ion()
    dispersion_plot(nltk_text, words)
    plt.ioff()
    plt.savefig('dispersion_plot.png')
    plt.show(block=False)
    plt.pause(1)
    plt.close()



def dispersion_plotting(nltk_text):
    #words to filter for
    words = ["good", "bad", "buy", "sell"]

    #step 1: iterate over all of the nltk_text and
    #step 2: compare with the given words and then save offset
    points = [(x, y) for x in range(len(nltk_text))
              for y in range(len(words)) if nltk_text[x] == words[y]]

    #zip aggregates 0 or more iteratables into a tuple
    if points:
        x, y = zip(*points)
    else:
        x = y = ()

    plt.plot(x, y, "rx", scalex=1)
    plt.yticks(range(len(words)), words, color="g")
    plt.xticks()
    plt.ylim(-1, len(words))
    plt.title("Lexical Dispersion Plot")
    plt.xlabel("Word Offset")


    plt.savefig('disp_plot')

    plt.show()






def filter_punctuation(nltk_text):
    text = [word.lower() for word in nltk_text if word.isalpha()]
    return text

def filter_stopwords(list_to_be_cleared):
    stop_words = set(stopwords.words("english"))

    # empty list für das Ergebnis
    filtered_list = []

    for word in list_to_be_cleared:
        if word.casefold() not in stop_words:
            filtered_list.append(word)

    return filtered_list



def frequency_dist(cleared_list):
    frequencydist = FreqDist(cleared_list)
    print(type(frequencydist))
    print(frequencydist)
    frequencydist.plot(20, cumulative=True)

#method that returns a dictionary representing the 20 most often used words
def frequency_dist_dict(cleared_list):
    frequency_dist = FreqDist(cleared_list)

    #dictionaries cant be sorted so its getting sorted as a list and then cast back into a dict
    od = dict(sorted(frequency_dist.items(), key=lambda item: item[1], reverse=True))

    #dictionaries cant be sliced so conversion to list in order to slice the first 20 instances (can be changed)
    first_twenty = list(od.items())[:20]

    #conversion back into a dict
    final_dict = {}
    final_dict.update(first_twenty)

    return final_dict



def collocations(cleared_list):

    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in cleared_list]
    new_text = nltk.Text(lemmatized_words)
    new_text.collocations()



def sentiment_anaylsis(cleaned_list):
    sia = SentimentIntensityAnalyzer()
    number = 0
    pos = 0
    neg = 0
    neu = 0
    with open('wallstreetbetsentiment.txt', 'w', encoding='utf-8') as f:
        for string in cleaned_list:
            s = sia.polarity_scores(string)
            pos = pos + s['pos']
            neg = neg + s['neg']
            neu = neu + s['neu']
            number = number +1

        pos_avg = pos / number
        neg_avg = neg / number
        neu_avg = neu/number
        print("Positive = " + str(pos_avg))
        print("Negative = " + str(neg_avg))
        print("Neutral =  " + str(neu_avg))


if __name__ == '__main__':
    text = preprocessing()
    dispersion_plot_vanilla(text)

    cleared = get_cleared_text(text)
    frequency_dist_dict(cleared)
    #dispersion_plot(text)
    #collocations(cleared)
    # print(type(cleared))
    # l = capitalize(cleared)
    # l = pos_tagger(l)
    # tree = nltk.ne_chunk(l)
    # print(type(tree))
    # counter = 1
    # while counter > 0:
    #     print(tree[counter])
    #     counter = counter -1

    #collocations(cleared)
