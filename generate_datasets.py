from RIFT_train import *

def generate_annual_datasets(y_s = 2010, y_e = 2020, reload_datasets = True, compress = True):
    torch.cuda.empty_cache()
    DAYS_LAG = 252
    DAYS_LEAD = 50

    ys = [str(y) for y in range(y_s, y_e+1)]
    target_fns = [
        lambda x1, x2: seq_corr_1d(x1[0:5], x2[0:5]),
        lambda x1, x2: seq_corr_1d(x1[0:10], x2[0:10]),
        lambda x1, x2: seq_corr_1d(x1[0:20], x2[0:20]),
        lambda x1, x2: seq_corr_1d(x1[0:30], x2[0:30]),
        lambda x1, x2: seq_corr_1d(x1[0:40], x2[0:40]),
        lambda x1, x2: seq_corr_1d(x1, x2),
    ]

    cuda_available = torch.cuda.is_available()
    print('CUDA available: ' + str(cuda_available))
    print(os.curdir)


    ts_df = pd.read_csv("data/ts_df/all_etf_data.csv.gz", encoding='utf-8-sig', compression='gzip')
    #tickers = ts_df['ticker'].unique().tolist()[:5] #TODO: remove this line
    #ts_df = ts_df.query("ticker in @tickers")
    econ_df = pd.read_csv("data/ts_df/econ_data.csv", encoding='utf-8')
    yield_df = pd.read_csv("data/ts_df/yield_interpolated.csv", encoding='utf-8')
    yield_df.columns = ['date'] + ['yield'+str(c) for c in yield_df.columns.tolist()[1:]]

    ts_df = ts_df.merge(econ_df, on=['date'], how='left')
    ts_df = ts_df.merge(yield_df, on=['date'], how='left')
    ts_df['date'] = pd.to_datetime(ts_df['date']).dt.date
    data_dts = [pd.to_datetime(d).date() for d in ('2010-01-01', '2021-12-01')]
    ts_df = ts_df.loc[(ts_df['date'] >= data_dts[0]) & (ts_df['date'] <= data_dts[1])]

    for y in ys:
        print(f"Generating dataset for year {y}")
        set = RIFT_Dataset(ts_df, (y+'-01-01', y+'-12-31'),
                                 target_fns=target_fns, days_lag=DAYS_LAG, days_lead=DAYS_LEAD, sample_size='ALL', norm_inputs=True)
        if compress:
            with lzma.open('data/train/train_set_'+y+'.dill.xz', 'wb') as handle:
                dill.dump(set, handle)
        else:
            with open('data/train/train_set_'+y+'.dill', 'wb') as handle:
                dill.dump(set, handle)
        if torch.cuda.is_available():
            del set
            torch.cuda.empty_cache()
        print(f"Finished generating dataset for year {y}")


if __name__ == "__main__":
    data = generate_annual_datasets(y_s = 2011, y_e = 2011, reload_datasets=True, compress=False)
    print(data.keys())

