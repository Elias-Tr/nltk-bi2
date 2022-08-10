import nltk
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from nltk import FreqDist, WordNetLemmatizer, ConditionalFreqDist, BigramCollocationFinder, TrigramCollocationFinder
from nltk.draw import dispersion_plot
from nltk.sentiment import SentimentIntensityAnalyzer

import preprocessing as pre


# attempts to extract Named Entities from a cleared list
def extract_ne(cleared_list):
    tags = nltk.pos_tag(cleared_list)
    tree = nltk.ne_chunk(tags, binary=True)
    return set(
        " ".join(i[0] for i in t)
        for t in tree
        if hasattr(t, "label") and t.label() == "NE"
    )



# plots via the nltk dispersion plotter, then saves it as a .png
# matplotlib needs to show graphs once in order to be able to save it
# which is why this method shows it, and then auto closes it after a second
def dispersion_plot_vanilla(nltk_text):
    words = ["good", "bad", "buy", "sell"]
    plt.ion()
    dispersion_plot(nltk_text, words)
    plt.ioff()
    plt.savefig('dispersion_plot.png')

    # comment in for proper picture saving with no manual closing
    # plt.show(block=False)
    # plt.pause(1)
    plt.show()
    plt.close()


# this method doesnt use the nltk dispersion plotter
# and does some it itself
def dispersion_plotting(nltk_text):
    # words to filter for
    words = ["good", "bad", "buy", "sell"]

    # step 1: iterate over all of the nltk_text and
    # step 2: compare with the given words and then save offset
    points = [(x, y) for x in range(len(nltk_text))
              for y in range(len(words)) if nltk_text[x] == words[y]]

    # zip aggregates 0 or more iteratables into a tuple
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
    cfdist = ConditionalFreqDist()

    # filling of the conditional frequency
    for word in cleared_txt:
        condition = len(word)
        cfdist[condition][word] += 1

    # get values back from the frequency distribution
    for condition in cfdist:
        for word in cfdist[condition]:
            if cfdist[condition].freq(word) > 0.02 and 10 > condition > 2:
                print("Cond. frequency of", word, cfdist[condition].freq(word), "[condition is word length =",
                      condition, "]")


# somewhat deprecated
# plots a frequency distribution within nltk
def frequency_dist(cleared_list):
    verbs = return_specified_pos('NN', cleared_list)

    #fd = FreqDist(cleared_list)
    fd = FreqDist(verbs)

    fd.plot(20, cumulative=True)

def return_specified_pos(pos_part, text):
    pos_tags = nltk.pos_tag(text)
    pos_list = []
    for word, pos in pos_tags:
        if pos.startswith(pos_part):
            pos_list.append(word)
    return pos_list


# method that returns a dictionary representing the 20 most often used words
def frequency_dist_dict(cleared_list):
    frequency_dist = FreqDist(cleared_list)

    # dictionaries cant be sorted so its getting sorted as a list and then cast back into a dict
    od = dict(sorted(frequency_dist.items(), key=lambda item: item[1], reverse=True))

    # dictionaries cant be sliced so conversion to list in order to slice the first 20 instances (can be changed)
    first_twenty = list(od.items())[:20]

    # conversion back into a dict
    final_dict = {}
    final_dict.update(first_twenty)

    return final_dict


# displays collocations
def collocations(cleared_list):
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in cleared_list]
    new_text = nltk.Text(lemmatized_words)

    new_text.collocations()

def trigram_collocations(cleared_list):
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in cleared_list]
    new_text = nltk.Text(lemmatized_words)
    trigram_measures = nltk.collocations.TrigramAssocMeasures()
    privacy_filter = lambda *w: 'tesla' not in w

    finder = TrigramCollocationFinder.from_words(
        new_text)
    # only trigrams that appear 3+ times
    finder.apply_freq_filter(20)
    # only trigrams that contain 'creature'
    finder.apply_ngram_filter(privacy_filter)
    # return the 10 n-grams with the highest PMI
    # print (finder.nbest(trigram_measures.likelihood_ratio, 10))
    for i in finder.score_ngrams(trigram_measures.likelihood_ratio):
        print(i)




# calculates overall sentiment and prints the positive, negative and neutral score
def sentiment_analysis(path):
    f = open(path, 'r', encoding='utf8')
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
        number = number + 1

    pos_avg = pos / number
    neg_avg = neg / number
    neu_avg = neu / number
    print("Positive = " + str(pos_avg))
    print("Negative = " + str(neg_avg))
    print("Neutral =  " + str(neu_avg))

