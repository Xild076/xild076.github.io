import yfinance
import re
import numpy as np
import string
import requests
from nltk.tokenize import sent_tokenize
import math
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup
from requests_html import HTMLSession
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import json
import datetime
import newspaper
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



class TextPreparer:
    def __init__(self):
        pass

    def get_body_text(self, url):
        url_j = newspaper.Article(url="%s" % (url), language='en')
        url_j.download()
        url_j.parse()
        return url_j.text

    def cleaner(self, text, sections):
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        #text = [w for w in text if w.isalpha()]
        def chunk_into_n(lst, n):
            size = math.ceil(len(lst) / n) - 1
            return list(map(lambda x: lst[x * size: x * size + size], list(range(n))))
        token_text = list(chunk_into_n(text, sections))
        text = model.encode(token_text)
        text = np.reshape(text, (-1))
        return text
    
    def spreadsheet_preparer(self, id):
        df = pd.read_csv(f"https://docs.google.com/spreadsheets/d/{id}/export?format=csv")
        return df
    
    def get_spec_row(self, df, i):
        return df.iloc[i].tolist()

    def prepare_list(self, df, size):
        print("Gathering :) all data...")
        train_in, train_out = [], []
        for i in range(len(df)):
            line = self.get_spec_row(df, i)
            train_in.append(self.cleaner(self.get_body_text(str(line[0])), size))
            train_out.append(int(line[1]))
            Util.progress_bar(i + 1, len(df), 'Data')
        print()
        return np.array(train_in), np.array(train_out)

    def simple_url(self, url, size):
        return self.cleaner(self.get_body_text(url), size)


class BasicSentAnalyzer:

    @staticmethod
    def fetch_stock_news_between_dates(stock_symbol, start_date, end_date, batch_size=5):
        base_url = f'https://news.google.com/rss/search?q={stock_symbol}+stock+news&hl=en-US&gl=US&ceid=US:en'
        news_articles = []

        with ThreadPoolExecutor() as executor:
            futures = []
            current_date = start_date

            while current_date <= end_date:
                date_str = current_date.strftime('%Y-%m-%d')
                url = f"{base_url}&hl=en-US&gl=US&ceid=US:en&geo=US&ts={date_str}T00:00:00Z"
                futures.append(executor.submit(BasicSentAnalyzer.fetch_news_from_url, url))
                current_date += timedelta(days=1)

            for future in futures:
                news_articles.extend(future.result()[:batch_size])

        return news_articles

    @staticmethod
    def fetch_news_from_url(url):
        response = requests.get(url)
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
        num_articles = 0

        # Initialize the VADER sentiment analyzer
        sid = SentimentIntensityAnalyzer()

        for article in news_articles:
            text = article['title'] + ' ' + article['description']

            # Get the sentiment polarity using VADER
            sentiment = sid.polarity_scores(text)['compound']
            total_sentiment += sentiment
            num_articles += 1

        if num_articles > 0:
            average_sentiment = total_sentiment / num_articles
            return average_sentiment
        else:
            return 0

    @staticmethod
    def get_news_sentiment(stock_symbol, start_date, end_date):
        news_articles = BasicSentAnalyzer.fetch_stock_news_between_dates(stock_symbol, start_date, end_date)
        average_sentiment = BasicSentAnalyzer.calculate_average_sentiment(news_articles)
        return average_sentiment


class SupervisedLearning(object):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = 0

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _bce_loss(self, y_pred, y_true):
        epsilon = 1e-7
        loss = -np.mean(y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon))
        return loss
    
    def _train(self, _data, _labels, _epochs, _log = False):
        num_samples, num_features = _data.shape
        weights = self.weights if self.weights != None else np.random.randn(num_features)
        bias = 0 if self.bias == 0 else self.bias

        losslist = []

        for i in range(_epochs):
            pre_zigmoid = np.dot(_data, weights) + bias
            mapped_numbers = self.sigmoid(pre_zigmoid)
            loss = self._bce_loss(mapped_numbers, _labels)

            losslist.append(loss)

            dW = (1 / num_samples) * np.dot(_data.T, (mapped_numbers - _labels) * loss)
            db = (1 / num_samples) * np.sum((mapped_numbers - _labels) * loss)

            weights -= self.learning_rate * dW
            bias -= self.learning_rate * db

            Util.progress_bar(i + 1, _epochs, 'Iterations', 0) if _log else None
        
        if _log:
            print()

            x = np.arange(0, _epochs)
            y = np.array(losslist)

            plt.plot(x, y)
            plt.show()
        
        return weights, bias
    
    def train(self, data, labels, epochs, log = False):
        self.weights, self.bias = self._train(data, labels, epochs, log)
    
    def _predict(self, intake, weights, bias):
        pre_zigmoid = np.dot(intake, weights) + bias
        mapped_numbers = self.sigmoid(pre_zigmoid)
        prediction = np.round(mapped_numbers)
        return prediction
    
    def predict(self, intake):
        return self._predict(intake, self.weights, self.bias)