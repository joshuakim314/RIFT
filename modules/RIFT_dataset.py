import os
import dill
import lzma

import numpy as np
import pandas as pd
import pickle
import torch
import torch.utils.data as data

from modules.train_utils import seq_corr_1d, seq_corr_3d

from itertools import combinations


class RIFT_Dataset(data.Dataset):
    def __init__(self, ts_df, date_range, target_fns, days_lag=500, days_lead=250, sample_size=200, random_seed=2023):
        super().__init__()
        self.ts_df = ts_df # columns: ["ticker", "date", "price", "log_ret"]
        self.start_date, self.end_date = date_range
        self.target_fns = target_fns
        self.days_lag = days_lag
        self.days_lead = days_lead
        self.sample_size = sample_size  # number or percent of tickers to select from available tickers at a given time
        self.random_seed = random_seed
        np.random.seed(self.random_seed)

        self.cuda_available = torch.cuda.is_available()
        if self.cuda_available:
            self.current_device = torch.cuda.current_device()

        self.process_ts_df()
        self.create_ts_dict()
        self.create_pairs()
    
    def process_ts_df(self):
        self.ts_df['date'] = pd.to_datetime(self.ts_df['date'])
        self.ts_df['log_ret'] = self.ts_df['log_ret'].fillna(0)  # replace NaN with 0 for the first trading day of each ticker
        self.ts_df['rel_date_num'] = self.ts_df['date'].rank(method='dense').astype(int) - 1
        self.ts_df = self.ts_df.sort_values(by=['ticker', 'date'])

        self.trading_date_list = sorted(set(self.ts_df['date'].tolist()))
        self.rel_date_num_dict = dict(zip(self.ts_df['date'], self.ts_df['rel_date_num']))
    
    def create_ts_dict(self):
        tickers = sorted(set(self.ts_df['ticker'].tolist()))
        ignore = ['ticker', 'date', 'rel_date', 'rel_date_num']
        num_cols = self.ts_df.shape[1] - len(ignore)  # exclude ticker, date, rel_date_num
        self.ts_dict = {ticker: np.full((len(self.trading_date_list), num_cols), np.nan) for ticker in tickers}
        for ticker in tickers:
            ticker_df = self.ts_df[self.ts_df['ticker'] == ticker].sort_values(by='date')
            rel_date_nums = ticker_df['rel_date_num'].tolist()
            i = 0
            for c in ticker_df.columns:
                if c in ignore:
                    continue
                data = ticker_df[c].tolist()
                self.ts_dict[ticker][rel_date_nums, i] = data
                i += 1
    
    def create_pairs(self):
        self.pairs = []
        for date in [date for date in self.trading_date_list if pd.to_datetime(self.start_date) <= date <= pd.to_datetime(self.end_date)]:
            tickers = self.get_available_tickers(date)
            print(f"{date}: {len(tickers)} tickers available")
            if self.sample_size == "ALL":
                for t, s in combinations(tickers, 2):
                    self.pairs.append((date, self.rel_date_num_dict[date], t, s))
            else:
                np.random.shuffle(tickers)
                ticker_num = self.sample_size // 2
                if isinstance(self.sample_size, float):
                    ticker_num = int(len(tickers) * self.sample_size) // 2
                for t, s in zip(tickers[0:ticker_num], tickers[ticker_num:2*ticker_num]):
                    self.pairs.append((date, self.rel_date_num_dict[date], t, s))
        print(f"Total number of pairs: {len(self.pairs)}")

    def get_available_tickers(self, date):
        if isinstance(date, str):
            date = pd.to_datetime(date)
        rel_date_num = self.rel_date_num_dict[date]
        start_rel_date_num = rel_date_num - self.days_lag + 1  # include today as part of historical data
        end_rel_date_num = rel_date_num + self.days_lead
        filtered_ts_df = self.ts_df.loc[(self.ts_df['rel_date_num'] >= start_rel_date_num) & (self.ts_df['rel_date_num'] <= end_rel_date_num)]
        tickers_missing = filtered_ts_df.groupby('ticker')['price'].apply(lambda x: x.isna().any())
        return list(tickers_missing[~tickers_missing].index)

    def calc_target_functions(self, target_inputs_arr, dims_to_compute, functions_list):
        n_functions = len(functions_list)
        target_arr = np.zeros(n_functions * len(dims_to_compute))
        for i, dim in enumerate(dims_to_compute):
            # and for functions specified
            for j, fn in enumerate(functions_list):
                # catch NaN correlation - caused by security prices not changing over a period
                target_arr[i * n_functions + j] = self.zero_out_na(
                    fn(target_inputs_arr[0][:, dim], target_inputs_arr[1][:, dim]))
        return(self.transform(target_arr))
    
    def transform(self, array_or_tensor, dtype=torch.float32):
        if torch.is_tensor(array_or_tensor):
            tensor = array_or_tensor.type(dtype)
        else:
            tensor = torch.as_tensor(array_or_tensor).type(dtype)
        if self.cuda_available:
            return tensor.to(self.current_device)
        else:
            return tensor
    
    def __getitem__(self, index):
        corr_col = 1 #0 for price, 1 for log_ret
        date, rel_date_num, t, s = self.pairs[index]
        ts_lag_t = self.ts_dict[t][rel_date_num-self.days_lag+1:rel_date_num+1, :]
        ts_lead_t = self.ts_dict[t][rel_date_num+1:rel_date_num+self.days_lead+1, :]
        ts_lag_s = self.ts_dict[s][rel_date_num-self.days_lag+1:rel_date_num+1, :]
        ts_lead_s = self.ts_dict[s][rel_date_num+1:rel_date_num+self.days_lead+1, :]
        
        #if self.sample_size == "ALL": #TODO: What is this supposed to do? Ans: Legacy
        #    raise NotImplementedError
        #    return (self.transform(ts_lag_t), self.transform(ts_lag_s)), [date.strftime('%Y-%m-%d'), rel_date_num, t, s]

        historical_corr = seq_corr_1d(self.transform(ts_lag_t[:, corr_col]), self.transform(ts_lag_s[:, corr_col])).item()
        target = np.zeros((len(self.target_fns)))
        for i, target_fn in enumerate(self.target_fns):
            target[i] = target_fn(self.transform(ts_lead_t[:, corr_col]), self.transform(ts_lead_s[:, corr_col])).item()
        target = target - historical_corr # target as residual correlation by subtracting historical correlation
        
        return (self.transform(ts_lag_t), self.transform(ts_lag_s)), self.transform(target)
        
    def __len__(self):
        return len(self.pairs)


if __name__ == '__main__':
    reload = False
    if reload == True:
        target_fns = [
            lambda x1, x2: seq_corr_1d(x1, x2),
            #lambda x1, x2: seq_corr_1d(x1[0:120], x2[0:120]),
            #lambda x1, x2: seq_corr_1d(x1[0:60], x2[0:60]),
            #lambda x1, x2: seq_corr_1d(x1[0:20], x2[0:20]),
            #lambda x1, x2: seq_corr_1d(x1[0:10], x2[0:10]),
            lambda x1, x2: seq_corr_1d(x1[0:5], x2[0:5])
        ]

        ts_df = pd.read_csv("../data/ts_df/all_etf_data.csv.gz", encoding='utf-8-sig', compression='gzip')
        tickers = ts_df['ticker'].unique().tolist()[:100]
        ts_df = ts_df.query("ticker in @tickers")
        econ_df = pd.read_csv("../data/ts_df/econ_data.csv", encoding='utf-8')
        yield_df = pd.read_csv("../data/ts_df/yield_interpolated.csv", encoding='utf-8')
        yield_df.columns = ['date'] + ['yield'+str(c) for c in yield_df.columns.tolist()[1:]]

        ts_df = ts_df.merge(econ_df, on=['date'], how='left')
        ts_df = ts_df.merge(yield_df, on=['date'], how='left')
        ts_df['date'] = pd.to_datetime(ts_df['date']).dt.date
        data_dts = [pd.to_datetime(d).date() for d in ('2010-01-01', '2021-12-01')]
        ts_df = ts_df.loc[(ts_df['date'] >= data_dts[0]) & (ts_df['date'] <= data_dts[1])]

        rd = RIFT_Dataset(ts_df, ('2018-01-01', '2018-01-08'), days_lead=10, days_lag=20,
                          target_fns=target_fns, sample_size='ALL')
        with lzma.open('../data/dataset/dataset_mini.dill.xz', 'wb') as handle:
            dill.dump(rd, handle)
    else:
        with lzma.open('../data/dataset/dataset_mini.dill.xz', 'rb') as handle:
            rd = dill.load(handle)

    print(rd[0])

    breakpoint()
