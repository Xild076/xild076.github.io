import yfinance
import re
import numpy as np
import string
import requests
from nltk.tokenize import sent_tokenize
import math
#from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup
from requests_html import HTMLSession
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import json
import datetime
#import newspaper
from colorama import Fore
from pydub import AudioSegment
from pydub.playback import play
import traceback
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
from datetime import datetime, timedelta
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlencode
from Util import Util


class BasicSentAnalyzer:

    @staticmethod
    def fetch_stock_news_between_dates(stock_symbol, start_date, end_date, batch_size=5):
        base_url = 'https://news.google.com/rss/search?'
        news_articles = []

        with ThreadPoolExecutor() as executor:
            futures = []
            current_date = start_date

            while current_date <= end_date:
                date_str = current_date.strftime('%Y-%m-%d')
                params = {
                    'q': f'{stock_symbol} stock news',
                    'hl': 'en-US',
                    'gl': 'US',
                    'ceid': 'US:en',
                    'geo': 'US',
                    'ts': f'{date_str}T00:00:00Z'
                }
                url = base_url + urlencode(params)
                futures.append(executor.submit(BasicSentAnalyzer.fetch_news_from_url, url))
                current_date += timedelta(days=1)

            for future in futures:
                news_articles.extend(future.result()[:batch_size])

        return news_articles

    @staticmethod
    def fetch_news_from_url(url):
        session = requests.Session()
        try:
            response = session.get(url)
        except:
            return []

        news_articles = []
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, features='xml')
            items = soup.find_all('item')

            for item in items:
                title = item.title.text
                description = item.description.text

                news_articles.append({
                    'title': title,
                    'description': description,
                })

        return news_articles

    @staticmethod
    def calculate_average_sentiment(news_articles):
        total_sentiment = 0
        num_articles = len(news_articles)

        sid = SentimentIntensityAnalyzer()

        for article in news_articles:
            text = article['title'] + ' ' + article['description']

            sentiment = sid.polarity_scores(text)['compound']
            total_sentiment += sentiment

        if num_articles > 0:
            average_sentiment = total_sentiment / num_articles
            return average_sentiment
        else:
            return 1

    @staticmethod
    def get_news_sentiment(stock_symbol, start_date, end_date):
        news_articles = BasicSentAnalyzer.fetch_stock_news_between_dates(stock_symbol, start_date, end_date)
        average_sentiment = BasicSentAnalyzer.calculate_average_sentiment(news_articles)
        return average_sentiment
