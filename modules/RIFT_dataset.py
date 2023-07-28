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
    def __init__(self, ts_df, date_range, target_fns, days_lag=500, days_lead=250, sample_size=200, random_seed=2023,
                 norm_inputs=True):
        super().__init__()
        self.ts_df = ts_df # columns: ["ticker", "date", "price", "log_ret"]
        if norm_inputs:
            yield_cols = []
            other_cols = []
            for c in ts_df.columns:
                if c in ['ticker', 'date', 'rel_date', 'rel_date_num', 'price']:
                    continue
                elif c[:5] == 'yield':
                    yield_cols.append(c)
                else:
                    other_cols.append(c)

            yield_mean = (ts_df[yield_cols].values).mean()
            yield_std = (ts_df[yield_cols].values).std()
            ts_df[yield_cols] = (ts_df[yield_cols] - yield_mean) / yield_std
            for c in []: # px_cols:
                #group by ticker
                px_mean = ts_df.groupby('ticker')[c].transform('mean')
                px_std = ts_df.groupby('ticker')[c].transform('std')
                ts_df[c] = (ts_df[c] - px_mean) / px_std
            for c in other_cols:
                #No need to group
                px_mean = ts_df[c].mean()
                px_std = ts_df[c].std()
                ts_df[c] = (ts_df[c] - px_mean) / px_std


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
        self.remove_excess_from_ts(delete=True)
    
    def process_ts_df(self):
        self.ts_df['date'] = pd.to_datetime(self.ts_df['date'])
        self.ts_df['log_ret'] = self.ts_df['log_ret'].fillna(0)  # replace NaN with 0 for the first trading day of each ticker
        self.ts_df['rel_date_num'] = self.ts_df['date'].rank(method='dense').astype(int) - 1
        self.ts_df = self.ts_df.sort_values(by=['ticker', 'date'])

        self.trading_date_list = sorted(set(self.ts_df['date'].tolist()))
        self.rel_date_num_dict = dict(zip(self.ts_df['date'], self.ts_df['rel_date_num']))
    
    def create_ts_dict(self):
        tickers = sorted(set(self.ts_df['ticker'].tolist()))
        ignore = ['ticker', 'date', 'rel_date', 'rel_date_num', 'price']
        num_cols = self.ts_df.shape[1] - len(ignore)  # exclude ticker, date, rel_date_num
        self.ts_dict = {ticker: np.full((len(self.trading_date_list), num_cols), np.nan) for ticker in tickers}
        for ticker in tickers:
            ticker_df = self.ts_df.query(f"ticker == '{ticker}'")
            #rel_date_nums = ticker_df['rel_date_num'].tolist()
            self.ts_dict[ticker] = ticker_df[[c for c in ticker_df.columns if c not in ignore]].values
            # i = 0
            # for c in ticker_df.columns:
            #     if c in ignore:
            #         continue
            #     data = ticker_df[c].tolist()
            #     self.ts_dict[ticker][rel_date_nums, i] = data
            #     i += 1
    
    def create_pairs(self):
        self.pairs = []
        for date in [date for date in self.trading_date_list if pd.to_datetime(self.start_date) <= date <= pd.to_datetime(self.end_date)]:
            tickers = self.get_available_tickers(date)
            np.random.shuffle(tickers)
            print(f"{date}: {len(tickers)} tickers available")
            if self.sample_size == "ALL":
                for t, s in combinations(tickers, 2):
                    self.pairs.append((date, self.rel_date_num_dict[date], t, s))
            elif isinstance(self.sample_size, int):
                i = 0
                for t, s in combinations(tickers, 2):
                    if i >= self.sample_size:
                        break
                    self.pairs.append((date, self.rel_date_num_dict[date], t, s))
                    i += 1
                print(f"Sampled {i} pairs")
            elif isinstance(self.sample_size, float):
                i = 0
                max = int(self.sample_size * len(tickers))
                for t, s in combinations(tickers, 2):
                    if i >= max:
                        break
                    self.pairs.append((date, self.rel_date_num_dict[date], t, s))
                    i += 1
                print(f"Sampled {self.sample_size}=>{i} pairs")

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

    def remove_excess_from_ts(self, delete=False):
        if delete:
            self.ts_df = None
        else:
            dts = [date for date in self.trading_date_list if pd.to_datetime(self.start_date) <= date <= pd.to_datetime(self.end_date)]
            max_rel_date_num = self.rel_date_num_dict[dts[-1]] + self.days_lead
            min_rel_date_num = self.rel_date_num_dict[dts[0]] - self.days_lag + 1
            self.ts_df = self.ts_df.loc[(self.ts_df['rel_date_num'] >= min_rel_date_num) & (self.ts_df['rel_date_num'] <= max_rel_date_num)]

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
        corr_col = 0 #log return
        date, rel_date_num, t, s = self.pairs[index]
        ts_lag_t = self.ts_dict[t][rel_date_num-self.days_lag+1:rel_date_num+1, :]
        ts_lead_t = self.ts_dict[t][rel_date_num+1:rel_date_num+self.days_lead+1, :]
        ts_lag_s = self.ts_dict[s][rel_date_num-self.days_lag+1:rel_date_num+1, :]
        ts_lead_s = self.ts_dict[s][rel_date_num+1:rel_date_num+self.days_lead+1, :]

        historical_corr = seq_corr_1d(self.transform(ts_lag_t[:, corr_col]), self.transform(ts_lag_s[:, corr_col])).item()
        target = np.zeros((len(self.target_fns)) + 1)
        for i, target_fn in enumerate(self.target_fns):
            target[i] = target_fn(self.transform(ts_lead_t[:, corr_col]), self.transform(ts_lead_s[:, corr_col])).item()
        target = target - historical_corr
        target[-1] = historical_corr
        
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
        with lzma.open('../data/train/dataset_mini.dill.xz', 'wb') as handle:
            dill.dump(rd, handle)
    else:
        with lzma.open('../data/train/dataset_mini.dill.xz', 'rb') as handle:
            rd = dill.load(handle)


    dataset_length = len(rd)
    train_indices = list(range(0, int(0.2 * dataset_length)))
    val_indices = list(range(int(0.8 * dataset_length), dataset_length))

    import math
    # split the 2010 data into train/val (no shuffle), then sample 0.5% of each
    train_set, val_set = data.Subset(rd, train_indices), data.Subset(rd, val_indices)
    train_size = math.floor(0.1 * len(train_set))
    val_size = math.floor(0.1 * len(val_set))
    train_set, _ = data.random_split(train_set, [train_size, len(train_set) - train_size])
    val_set, _ = data.random_split(val_set, [val_size, len(val_set) - val_size])
    print(f"dataset size: {dataset_length}")
    print(f"train_set size: {len(train_set)}")
    print(f"val_set size: {len(val_set)}")

    score_loader = data.DataLoader(train_set, batch_size=32 * 1, drop_last=False,shuffle=False)

    print(rd[0])

    breakpoint()
