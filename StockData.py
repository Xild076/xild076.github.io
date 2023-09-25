import yfinance as yf
from datetime import datetime, timedelta
import pandas_datareader
from pandas_datareader.data import DataReader
import pandas as pd
import numpy as np
import random
from pathlib import Path
import time
import Util
import math
import os
from Policy import PolicyAlgorithm
import colorama
import cProfile
import pstats
import tuna
from Util import Util
from ArticleSorter import BasicSentAnalyzer
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')


class FinanceManager(object):
    def __init__(self, stock_keys, data_keys, start_day, end_day):
        self.stock_keys = stock_keys
        self.data_keys = data_keys
        self.start_day = start_day
        self.end_day = end_day

        self.dict_of_stock_files = {}
        self.dict_of_fred_data = {}

    def get_stock_data(self, keys, start_day, end_day):
        #Getting data based on days

        ticker = yf.Ticker(keys)
        ticker_hist = ticker.history(period='max', start=start_day.strftime('%Y-%m-%d'), end=end_day.strftime('%Y-%m-%d'), interval='5d')
        avg_total = []
        indexi = ticker_hist.index.values
        time_list = []
        for i in range(len(indexi)):
            time_list.append(Util.dt64todt(indexi[i]))
            avg_total.append(ticker_hist.values[i][3])
        return np.array(time_list), np.array(avg_total)
    
    def get_stock_daycount_data(self, key, day, day_count):
        # Get stock data from file or cache
        if key not in self.dict_of_stock_files:
            all_array = self.load_stock_data(key)
            self.dict_of_stock_files[key] = all_array
        else:
            all_array = self.dict_of_stock_files[key]

        dates, prices = all_array[0], all_array[1]
        nearest_date, index = Util.find_nearest(dates, day)
        return dates[index : index + day_count], prices[index : index + day_count]

    def stock_data_dump(self, keys, start_day, end_day, force = False):
        #Dumps stock data

        for key in keys:
            print(colorama.Fore.MAGENTA + f"{key}... ", end="") 
            if not force:
                if not Util.file_exists(f'FinanceAIClean\saved_stocks\{key}.npy'):
                    time_list, data_list = self.get_stock_data(key, end_day, start_day)
                    Util.save_load(2, f'FinanceAIClean\saved_stocks\{key}.npy', np.array([time_list, data_list]))
            else:
                time_list, data_list = self.get_stock_data(key, end_day, start_day)
                Util.save_load(2, f'FinanceAIClean\saved_stocks\{key}.npy', np.array([time_list, data_list]))
        print("")

    def get_fred_data(self, code, start_day, end_day):
        #['T10YIE', 'REAINTRATREARAT10Y', 'UNRATE']

        data_source = 'fred'
        data = DataReader(name=code, data_source=data_source, start=end_day, end=start_day)
        data_np = np.array(data)
        data_list = []
        mean = 0
        for i in range(len(data_np)):
            if not math.isnan(data_np[i][0]):
                mean += data_np[i][0]
        mean /= len(data_np)
        for i in range(len(data_np)):
            if math.isnan(data_np[i][0]):
                data_np[i][0] = mean
            data_list.append(data_np[i][0])
        return np.array(data.index), np.array(data_list)
    
    def fred_data_dump(self, codes, start_day, end_day, force = False):
        #Dumps fred data

        for code in codes:
            if not force:
                if not Util.file_exists(f'FinanceAIClean\saved_freds\{code}.npy'):
                    time_list, data_list = self.get_fred_data(code, start_day, end_day)
                    Util.save_load(2, f'FinanceAIClean\saved_freds\{code}', np.array([time_list, data_list]))
            else:
                time_list, data_list = self.get_fred_data(code, start_day, end_day)
                Util.save_load(2, f'FinanceAIClean\saved_freds\{code}', np.array([time_list, data_list]))
    
    def get_fred_day_data(self, codes, day):
        # Get fred data from file or cache
        temp_fred_data = {}
        for code in codes:
            if code not in self.dict_of_fred_data:
                all_array = self.load_fred_data(code)
                self.dict_of_fred_data[code] = all_array
            else:
                all_array = self.dict_of_fred_data[code]

            dates, datas = all_array[0], all_array[1]
            nearest_date, index = Util.find_nearest(dates, day)
            temp_fred_data[code] = np.array([nearest_date, datas[index]])
        
        myKeys = list(temp_fred_data.keys())
        myKeys.sort()
        sorted_dict = {i: temp_fred_data[i] for i in myKeys}

        return sorted_dict

    def get_fred_daycount_data(self, codes, days):
        temp_fred_data = {}
        for code in codes:
            if code not in self.dict_of_fred_data.keys():
                all_array = Util.save_load(3, f'FinanceAIClean\saved_freds\{code}.npy', None)
                dates, datas = all_array[0], all_array[1]

                if type(dates[0]) == type(10):    
                    dates = np.array([datetime.fromtimestamp(ts / 1e9) for ts in dates])
                
                nearest_day_list = []
                nearest_index_list = []
                for day in days:
                    nearest_date, index = Util.find_nearest(dates, day)
                    nearest_day_list.append(nearest_date)
                    nearest_index_list.append(datas[index])
                
                self.dict_of_fred_data[code] = np.array([dates, datas])
                temp_fred_data[code] = np.array([nearest_day_list, nearest_index_list])
            else:
                all_array = self.dict_of_fred_data[code]
                dates, datas = all_array[0], all_array[1]

                if type(dates[0]) == type(10):
                    dates = np.array([datetime.fromtimestamp(ts / 1e9) for ts in dates])

                nearest_day_list = []
                nearest_index_list = []
                for day in days:
                    nearest_date, index = Util.find_nearest(dates, day)
                    nearest_day_list.append(nearest_date)
                    nearest_index_list.append(datas[index])
                
                temp_fred_data[code] = np.array([nearest_day_list, nearest_index_list])
        
        myKeys = list(temp_fred_data.keys())
        myKeys.sort()
        sorted_dict = {i: temp_fred_data[i] for i in myKeys}

        return sorted_dict
    
    def load_stock_data(self, key):
        file_path = f'FinanceAIClean\saved_stocks\{key}.npy'
        if not Util.file_exists(file_path):
            time_list, data_list = self.get_stock_data(key, self.end_day, self.start_day)
            Util.save_load(2, file_path, np.array([time_list, data_list]))
        else:
            all_array = Util.save_load(3, file_path, None)
            return np.array(all_array)
    
    def load_fred_data(self, code):
        file_path = f'FinanceAIClean\saved_freds\{code}.npy'
        if not Util.file_exists(file_path):
            time_list, data_list = self.get_fred_data(code, self.start_day, self.end_day)
            Util.save_load(2, file_path, np.array([time_list, data_list]))
        else:
            all_array = Util.save_load(3, file_path, None)
            return np.array(all_array)

    def dump_total_data(self):
        self.stock_data_dump(self.stock_keys, self.start_day + timedelta(days=365), self.end_day)
        self.fred_data_dump(self.data_keys, self.start_day + timedelta(days=365), self.end_day)

class FinanceEnv(object):
    def __init__(self, stock_code_list, fred_code_list, day_count, scale, act_detail, day_start=Util.date_after_now(years=2), day_end=Util.date_after_now(years=10)):
        self.stock_code_list = stock_code_list
        self.fred_code_list = fred_code_list
        self.day_count = day_count
        self.day_start = day_start
        self.day_end = day_end
        self.scale = scale
        self.observation_space = day_count * (1 + 1 + len(fred_code_list)) + 2
        self.act_detail = act_detail
        self.action_space = self.act_detail * 2 + 1
        self.counter = 0
        self.last_action = []

        self.finance_manager = FinanceManager(self.stock_code_list, self.fred_code_list, self.day_start, self.day_end)
        self.finance_manager.dump_total_data()
    
    def get_random_prepped_data(self):
        list_of_days = []
        list_of_avg_datas = []
        for stock_code in self.stock_code_list:
            date = Util.random_date(self.day_end, self.day_start)
            day_list, data_list = self.finance_manager.get_stock_daycount_data(stock_code, date, self.day_count)
            list_of_days.append(day_list)
            list_of_avg_datas.append(data_list)

        avg_list_days = Util.average_by_func(list_of_days, Util.avg_dates)
        self.avg = avg_list_days
        
        list_fred = self.finance_manager.get_fred_daycount_data(self.fred_code_list, avg_list_days)


        return np.array(list_of_days), np.array(list_of_avg_datas), list_fred
    
    def reset(self):
        self.counter = 0
        self.day_list, self.data_list, self.fred_list = self.get_random_prepped_data()

        return self.get_state()
    
    def get_state(self):
        self.current_code_index = random.randint(0, len(self.stock_code_list) - 1)
        last_day = self.day_list[self.current_code_index][-1]
        self.afterwards_date = Util.random_date(last_day + timedelta(days=1), last_day + timedelta(days=15))

        new_sent = BasicSentAnalyzer.get_news_sentiment(self.stock_code_list[self.current_code_index],
                                                        self.day_list[self.current_code_index][-2],
                                                        last_day)

        int_day_list = Util.convert_secs(self.day_list[self.current_code_index])

        data_to_concat = [self.data_list[self.current_code_index], int_day_list]

        for fred_code in self.fred_code_list:
            data_to_concat.append(self.fred_list[fred_code][1])

        data_to_concat.append(np.array([Util.convert_secs([self.afterwards_date])]).reshape(1,))
        data_to_concat.append(np.array([new_sent]).reshape(1,))

        entire_state = np.concatenate(data_to_concat)

        return entire_state

    def step(self, action, test=False):
        _, new_recent_data = self.finance_manager.get_stock_daycount_data(self.stock_code_list[self.current_code_index], self.afterwards_date, 1)
        percent_change = new_recent_data[0] / self.data_list[self.current_code_index][-1]
        assumed_action = round(round(math.tanh((percent_change - 1) * 8) + 1, 1) * self.act_detail, 0)
        reward = 0
        if assumed_action == action:
            reward += 5 * self.scale
            self.last_action = []
        else:
            reward -= 5 * self.scale * self.day_count // 2
            """if math.fabs(assumed_action - action) <= (self.day_count // 10 + 1):
                reward += 5 * (self.scale)"""
            if action > self.action_space // 2 and self.action_space // 2 <= assumed_action:
                reward -= 5 * self.scale * self.day_count // 2
            if action <= self.action_space // 2 and self.action_space > self.action_space // 2:
                reward -= 5 * self.scale * self.day_count // 2
            for i in range(len(self.last_action)):
                    if self.last_action[len(self.last_action) - i - 1] == action:
                        reward -= 5 * self.day_count // 2
                    else:
                        break
        self.counter += 1
        self.last_action.append(action)
        if test:
            print(colorama.Fore.RED + "Assumed Action:", assumed_action)
            print("Taken Action:", action)
            print("PChange:", percent_change)

        return self.get_state(), reward, self.counter == len(self.stock_code_list) * self.scale , None

    def prep_data(self, day_count):
        #print("PREPDATA")
        list_of_days = []
        list_of_avg_datas = []
        for i in self.list_core:
            date = FinanceManager.random_date(datetime.now() + timedelta(days=-(365 * 8)), datetime.now() + timedelta(days=-(365 * 4)))
            a, b = self.fr.get_stock_data(i, date, day_count)
            list_of_days.append(a)
            list_of_avg_datas.append(b)
        return np.array(list_of_days), np.array(list_of_avg_datas)

print(colorama.Fore.CYAN + "HI THERE, [[HYPERLINK BLOCKED]], WE ARE GOING TO [[Get a discount]] ON THIS AMAZING [[Bad]] CODE!!!")

sdata = ['AAPL', 'ADBE', 'ADSK', 
                        'AMZN', 'ANSS', 'BIIB', 
                        'BKNG', 'BMY', 'CHTR', 
                        'CMCSA', 'COIN', 'COST', 
                        'CRM', 'CSCO', 'DOCU', 
                        'GOOG', 'GOOGL', 'INTC', 
                        'INTU', 'ISRG', 'JNJ', 
                        'JPM', 
                        'MA', 'META', 
                        'MMM', 
                        'MS', 'NFLX',
                        'PFE', 'PYPL', 
                        'TSLA']

fin_env = FinanceEnv(['AAPL'], ['T10YIE', 'T10Y2Y', 'MORTGAGE30US', 'SNDR', 'MMTY', 'DGS10', 'TB3MS', 
                                'FEDFUNDS', 'SOFR', 'BAMLH0A0HYM2', 'DTWEXBGS', 'BOGMBASE', 'WRESBAL', 
                                'RCCCBBALTOT', 'PCE', 'GDP', 'CPIAUCSL', 'INDPRO', 'REAINTRATREARAT10Y', 
                                'RSXFS', 'COMPOUT'], 5, 2, 10)

p_alg = PolicyAlgorithm(fin_env, 0.01, 0.99, 0.99, 0.05, 24)
#p_alg.load_model('chad_god-20230903210847')

profile = cProfile.Profile()
profile.enable()

p_alg.train(50, 5, 50)

profile.disable()
stats = pstats.Stats(profile).sort_stats('cumtime')
stats.print_stats(10)
stats = pstats.Stats(profile).sort_stats('tottime')
stats.print_stats(10)


#p_alg.train_render(1000, 5, 50)

for i in range(10):
    p_alg.test(50)

print(colorama.Fore.CYAN + "BYE THERE, [[Error 404]] HAS HAPPENED YOU NERD. [[Just kidding]] THE PROGRAM HAS FINISHED")