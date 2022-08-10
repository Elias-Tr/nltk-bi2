import datetime
import pprint

import requests
import requests.auth
import praw
from psaw import PushshiftAPI


def reddit():
    api = PushshiftAPI()

    #query parameters
    gen = api.search_comments(q='tesla',score = '>5', subreddit=['PersonalFinance', 'investing', "Finance", "financialplanning"])

    #maximum amount of responses returned
    max_response_cache = 5000
    cache = []
    for c in gen:
        cache.append(c)
        # Omit this test to actually return all results. Wouldn't recommend it though: could take a while, but you do you.
        if len(cache) >= max_response_cache:
            break

    with open('test.txt', 'w', encoding='utf-8') as f:
        for comment in cache:
            if(comment.body != ""):
                body = str(comment.body)
                time = comment.created_utc
                body = body.replace("\n", "")
                body = body.replace(";", "")
                forcsv = body
                #optional additional paramteres extracted from the answer
                #+ body + ";" + str(comment.score) + ";" + str(timebetter)
                f.write(forcsv + "\n")


if __name__ == '__main__':
    reddit()