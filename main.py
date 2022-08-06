# This is a sample Python script.

import nltk
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from nltk import FreqDist, WordNetLemmatizer, ConditionalFreqDist
from nltk.draw import dispersion_plot
from nltk.sentiment import SentimentIntensityAnalyzer

import preprocessing as pre
import nltk_methods as nm


# necessary nltk downloads - or use nltk.download() to download everything

# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('vader_lexicon')
# nltk.download('averaged_perceptron_tagger')
# nltk.download("maxent_ne_chunker")
# nltk.download("words")


def all_reddit_analysis():
    text = pre.preprocessing('teslafinance.txt')
    cleared = pre.get_cleared_text(text)

    nm.dispersion_plot_vanilla(text)
    nm.frequency_dist(cleared)
    nm.collocations(cleared)
    #nm.trigram_collocations(cleared)
    nm.sentiment_analysis()


def all_news_analysis():
    text = pre.preprocessing('tesla_news.txt')
    cleared = pre.get_cleared_text(text)

    nm.dispersion_plot_vanilla(text)
    nm.frequency_dist(cleared)
    nm.collocations(cleared)
    #nm.trigram_collocations(cleared)
    nm.sentiment_analysis()


if __name__ == '__main__':
    all_reddit_analysis()
