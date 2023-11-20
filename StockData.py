import os
import time
import colorama
import numpy as np
from datetime import datetime, timedelta
from pandas_datareader.data import DataReader
import yfinance as yf
import random
import math
from pathlib import Path
from Util import Util
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from Policy import PolicyAlgorithm
from ArticleSorter import BasicSentAnalyzer
import pandas_datareader
import pstats
import cProfile
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
                if not Util.file_exists(f'saved_stocks\{key}.npy'):
                    time_list, data_list = self.get_stock_data(key, end_day, start_day)
                    Util.save_load(2, f'saved_stocks\{key}.npy', np.array([time_list, data_list]))
            else:
                time_list, data_list = self.get_stock_data(key, end_day, start_day)
                Util.save_load(2, f'saved_stocks\{key}.npy', np.array([time_list, data_list]))
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
                if not Util.file_exists(f'saved_freds\{code}.npy'):
                    time_list, data_list = self.get_fred_data(code, start_day, end_day)
                    Util.save_load(2, f'saved_freds\{code}', np.array([time_list, data_list]))
            else:
                time_list, data_list = self.get_fred_data(code, start_day, end_day)
                Util.save_load(2, f'saved_freds\{code}', np.array([time_list, data_list]))
    
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
                all_array = Util.save_load(3, f'saved_freds\{code}.npy', None)
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
        file_path = f'saved_stocks\{key}.npy'
        if not Util.file_exists(file_path):
            time_list, data_list = self.get_stock_data(key, self.end_day, self.start_day)
            Util.save_load(2, file_path, np.array([time_list, data_list]))
        else:
            all_array = Util.save_load(3, file_path, None)
            return np.array(all_array)
    
    def load_fred_data(self, code):
        file_path = f'saved_freds\{code}.npy'
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
    def __init__(self, stock_code_list, fred_code_list, day_count, scale, act_detail, up_to, day_start=Util.date_after_now(years=2), day_end=Util.date_after_now(years=10)):
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
        self.action_counter = {}
        self.afterwards_price = 0
        self.right_in_row = True
        self.up_to = up_to

        self.finance_manager = FinanceManager(self.stock_code_list, self.fred_code_list, self.day_start, self.day_end)
        self.finance_manager.dump_total_data()
    
    def get_random_prepped_data(self):
        list_of_days = []
        list_of_avg_datas = []
        for stock_code in self.stock_code_list:
            date = Util.random_date(self.day_end, self.day_start)
            day_list, data_list = self.finance_manager.get_stock_daycount_data(stock_code, date, self.day_count + 1)
            list_of_days.append(day_list)
            list_of_avg_datas.append(data_list)

        avg_list_days = Util.average_by_func(list_of_days, Util.avg_dates)
        self.avg = avg_list_days
        
        self.sents = []
        for scn in range(len(self.stock_code_list)):
            self.sents.append(BasicSentAnalyzer.get_news_sentiment(self.stock_code_list[scn], list_of_days[scn][-3], list_of_days[scn][-2]))
        
        list_fred = self.finance_manager.get_fred_daycount_data(self.fred_code_list, avg_list_days[:len(day_list)-1])

        return np.array(list_of_days), np.array(list_of_avg_datas), list_fred
    
    def reset(self):
        self.counter = 0
        self.day_list, self.data_list, self.fred_list = self.get_random_prepped_data()

        return self.get_state()
    
    def get_state(self):
        self.current_code_index = random.randint(0, len(self.stock_code_list) - 1)

        self.afterwards_date = self.day_list[self.current_code_index][-1]
        new_sent = self.sents[self.current_code_index]
        
        int_day_list = Util.convert_secs(self.day_list[self.current_code_index][:len(self.day_list[self.current_code_index])-1])
        
        int_day_list = int_day_list / (Util.round_to_nearest_max(np.max(int_day_list)))
        
        data_to_concat = [self.data_list[self.current_code_index][:len(self.day_list[self.current_code_index])-1] / (Util.round_to_nearest_max(np.max(self.data_list[self.current_code_index]))), int_day_list]
        
        for fred_code in self.fred_code_list:
            normalized_fred = self.fred_list[fred_code][1] / (Util.round_to_nearest_max(np.max(self.fred_list[fred_code][1])))
            data_to_concat.append(normalized_fred)
        

        after_date_secs = Util.convert_secs([self.afterwards_date])
        
        data_to_concat.append(np.array([after_date_secs / (Util.round_to_nearest_max(after_date_secs))]).reshape(1,))
        data_to_concat.append(np.array([new_sent]).reshape(1,))

        entire_state = np.concatenate(data_to_concat)
        
        return entire_state
        

    def step(self, action, test=False):
        percent_change = self.data_list[self.current_code_index][-1] / self.data_list[self.current_code_index][-2]
        mapped_value = math.floor((percent_change - 1) / (self.up_to / self.act_detail)) + self.act_detail + 1
        mapped_value = max(0, min(mapped_value, 2 * self.act_detail + 1))
        assumed_act = mapped_value 
        reward = 0
        if assumed_act == action:
            reward += 5
        else:
            reward -= 10

        if test:
            print(colorama.Fore.RED + "Assumed Action:", assumed_act)
            print("Taken Action:", action)
            print("PChange:", percent_change)

        done = self.counter == len(self.stock_code_list) * self.scale
        return self.get_state(), reward, done, None

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
