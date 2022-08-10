from datetime import timedelta, datetime
from datetime import *
import time
import requests
from currentsapi import CurrentsAPI
from dateutil.relativedelta import relativedelta


def getnews():
    #obtain an api key by signing up on https://currentsapi.services/en
    api_key = 'insert-your-api-key'

    #set endtime to current time
    end = datetime.now()

    #amount of timewindows that should be collected from - actual number of results may vary depending on how many news articles were released
    loop =100

    while loop > 0:

        api = CurrentsAPI(api_key=api_key)
        response = api.search(limit=200, keywords="tesla", page_number=10, end_date=end, language='en')
        newslist = (response['news'])
        #move the new enddate to 12 hours before the last one
        end = end - relativedelta(hours=12)


        with open('tesla_news.txt', 'a', encoding='utf-8') as f:
            for i in newslist:
                #only needed if the title is also needed/wanted
                # title = i['title']
                # title = title.replace(";", "")

                description = i['description']
                description = description.replace("\n", "")
                #result = str(title) + ";" + str(description) + ";" + str(i['category'])
                #result = result.replace("\n", "")
                f.write(description + '\n')
        loop = loop - 1

        #sleep call in order to obey the call limits per second
        time.sleep(2)


if __name__ == '__main__':
    print("start")
    getnews()
