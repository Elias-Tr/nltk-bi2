# This is a sample Python script.
import nltk
from nltk.corpus import stopwords
import matplotlib
from nltk import FreqDist, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
#nltk.download('wordnet')
#nltk.download('omw-1.4')
#nltk.download('vader_lexicon')

def preprocessing():


    #open text file you want to analyze
    f = open('wallstreetbet.txt', 'r', encoding='utf8')
    raw = f.read()

    #tokenize by words and make into nltk text
    tokens = nltk.word_tokenize(raw)
    text = nltk.Text(tokens)

    return text

def get_cleared_text(text):
    cleared = filter_punctuation(text)
    cleared = filter_stopwords(cleared)
    return cleared


def dispersion_plot(nltk_text):
    nltk_text.dispersion_plot(["good", "bad", "buy", "sell"])

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
    frequencydist.plot(20, cumulative=True)

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

    cleared = get_cleared_text(text)
    collocations(cleared)

