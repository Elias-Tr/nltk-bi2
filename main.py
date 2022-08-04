# This is a sample Python script.
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


#gets installed with pyspellchecker

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
#nltk.download('wordnet')
#nltk.download('omw-1.4')
#nltk.download('vader_lexicon')
#nltk.download('averaged_perceptron_tagger')
#nltk.download("maxent_ne_chunker")
#nltk.download("words")


#opens the file and tokenizes it by words
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
    # f = open('wallstreetbetsentiment.txt', 'a', encoding='utf8')
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

#attempts to extract Named Entities from a cleared list
def extract_ne(cleared_list):
    tags = nltk.pos_tag(cleared_list)
    tree = nltk.ne_chunk(tags,binary=True)
    return set(
        " ".join(i[0] for i in t)
        for t in tree
        if hasattr(t, "label") and t.label() == "NE"
    )

#plots via the nltk dispersion plotter, then saves it as a .png
#matplotlib needs to show graphs once in order to be able to save it
#which is why this method shows it, and then auto closes it after a second
def dispersion_plot_vanilla(nltk_text):
    words = ["good", "bad", "buy", "sell"]
    plt.ion()
    dispersion_plot(nltk_text, words)
    plt.ioff()
    plt.savefig('dispersion_plot.png')

    #comment in for proper picture saving with no manual closing
    #plt.show(block=False)
    # plt.pause(1)
    plt.show()
    plt.close()



#this method doesnt use the nltk dispersion plotter
#and does some it itself
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

def condition_prediction(cleared_txt):

    cfdist= ConditionalFreqDist()

    #filling of the conditional frequency
    for word in cleared_txt:
        condition = len(word)
        cfdist[condition][word] += 1

    #get values back from the frequency distribution
    for condition in cfdist:
        for word in cfdist[condition]:
            if cfdist[condition].freq(word) > 0.02 and 10 > condition > 2:
                print("Cond. frequency of", word, cfdist[condition].freq(word), "[condition is word length =", condition, "]")




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


#somewhat deprecated
#plots a frequency distribution within nltk
def frequency_dist(cleared_list):
    fd = FreqDist(cleared_list)
    # print(type(frequencydist))
    # print(frequencydist)
    fd.plot(20, cumulative=True)
    print("hello")

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


#displays collocations
def collocations(cleared_list):

    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in cleared_list]
    new_text = nltk.Text(lemmatized_words)
    new_text.collocations()


#calculates overall sentiment and prints the positive, negative and neutral score
def sentiment_analysis():
    f = open('teslafinance.txt', 'r', encoding='utf8')
    raw = f.read()

    # tokenize by sentences and make into nltk text

    tokens = nltk.sent_tokenize(raw)
    text2 = nltk.Text(tokens)



    sia = SentimentIntensityAnalyzer()
    number = 0
    pos = 0
    neg = 0
    neu = 0

    for string in text2:

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


def all_analysis():
    text = preprocessing()
    cleared = get_cleared_text(text)

    dispersion_plot_vanilla(text)

    frequency_dist(cleared)

    collocations(cleared)

    sentiment_analysis()

if __name__ == '__main__':
    all_analysis()





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

