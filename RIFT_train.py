import os
import pickle
import dill
import lzma
import itertools
import pandas as pd
import numpy as np

import torch
import torch.utils.data as data

from modules.RIFT_dataset import RIFT_Dataset
from modules.RIFT_model import RIFT_Model
from modules.RIFT_model_config import RIFT_Model_Config
from modules.train_model import Model_Trainer
from modules.radam import RAdam
from modules.train_utils import ret_seq_indices, shifted_diff, ts_moving_average, ts_moving_var, seq_corr_1d, seq_corr_3d


def get_input_ts_transform_list():
    input_ts_transform_list = []
    input_ts_transform_list.append(ret_seq_indices)
    for i in range(1, 11):
        def f_shifted_diff(x, i=i): return shifted_diff(x, i)
        input_ts_transform_list.append(f_shifted_diff)
    for i in range(10, 200, 20):
        def f_moving_avg(x, i=i): return ts_moving_average(x, i)
        input_ts_transform_list.append(f_moving_avg)
        def f_moving_var(x, i=i): return ts_moving_var(x, i)
        input_ts_transform_list.append(f_moving_var)
    return input_ts_transform_list


input_ts_transform_fns = get_input_ts_transform_list()

def train_RIFT_model(config, train_set, val_set, param_search_grid, random_seed=2023):
    np.random.seed(random_seed)
    available_devices = max(torch.cuda.device_count(), 1)
    print('available devices: ' + str(available_devices))
    if available_devices >= 1:
        torch.cuda.empty_cache()  # clear cache in case prior experiment did not finish properly

    train_loader = data.DataLoader(train_set, batch_size=config.batch_size*available_devices, drop_last=True, shuffle=True)
    val_loader = data.DataLoader(val_set, batch_size=config.batch_size*available_devices, drop_last=True, shuffle=True)
    score_loader = data.DataLoader(val_set, batch_size=config.batch_size*available_devices, drop_last=False, shuffle=False)
    print(f"train_loader size: {len(train_loader)}")
    print(f"train_loader size: {len(val_loader)}")
    
    param_search_list = []
    for key in param_search_grid:
        param_search_list.append([(key, value) for value in param_search_grid[key]])
    param_product = list(itertools.product(*param_search_list))
    np.random.shuffle(param_product)
    
    model_trainers = []
    res_dfs = []
    # loop through combinations of model params in random order
    for i, combination in enumerate(param_product):
        rs = i + random_seed
        torch.manual_seed(rs)
        for param_tuple in combination:
            config.__dict__[param_tuple[0]] = param_tuple[1]
            config.random_seed = rs
        print('Experiment combination: ' + str(combination))
        model = RIFT_Model(config)
        model_trainer = Model_Trainer(model, train_loader, val_loader)
        print('Experiment id: ' + model_trainer.experiment_id)
        # clear GPU memory if using CUDA
        if available_devices >= 1:
            torch.cuda.empty_cache()
        model_trainer.train()
        res_df = model_trainer.score_model(score_loader)
        model_trainers.append(model_trainer)
        res_dfs.append(res_df)

    return model_trainer, res_df


if __name__ == '__main__':
    DAYS_LAG = 252
    DAYS_LEAD = 50
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

    reload_datasets = True
    if reload_datasets:

        ts_df = pd.read_csv("data/ts_df/all_etf_data.csv.gz", encoding='utf-8-sig', compression='gzip')
        tickers = ts_df['ticker'].unique().tolist()[:5] #TODO: remove this line
        ts_df = ts_df.query("ticker in @tickers")
        econ_df = pd.read_csv("data/ts_df/econ_data.csv", encoding='utf-8')
        yield_df = pd.read_csv("data/ts_df/yield_interpolated.csv", encoding='utf-8')
        yield_df.columns = ['date'] + ['yield'+str(c) for c in yield_df.columns.tolist()[1:]]

        ts_df = ts_df.merge(econ_df, on=['date'], how='left')
        ts_df = ts_df.merge(yield_df, on=['date'], how='left')
        ts_df['date'] = pd.to_datetime(ts_df['date']).dt.date
        data_dts = [pd.to_datetime(d).date() for d in ('2010-01-01', '2021-12-01')]
        ts_df = ts_df.loc[(ts_df['date'] >= data_dts[0]) & (ts_df['date'] <= data_dts[1])]

        train_set = RIFT_Dataset(ts_df, ('2016-01-01', '2016-12-31'),
                                 target_fns=target_fns, days_lag=DAYS_LAG, days_lead=DAYS_LEAD, sample_size='ALL')
        val_set = RIFT_Dataset(ts_df, ('2017-01-01', '2017-02-01'),
                               target_fns=target_fns, days_lag=DAYS_LAG, days_lead=DAYS_LEAD, sample_size='ALL')
        # with open('data/train/train_set.dill', 'wb') as handle:
        #     dill.dump(train_set, handle)
        # with open('data/train/val_set.dill', 'wb') as handle:
        #     dill.dump(val_set, handle)

        with lzma.open('data/train/train_set.dill.xz', 'wb') as handle:
            dill.dump(train_set, handle)
        with lzma.open('data/train/val_set.dill.xz', 'wb') as handle:
            dill.dump(val_set, handle)
    else:
        with lzma.open('data/train/train_set.dill.xz', 'rb') as handle:
            train_set = dill.load(handle)
        with lzma.open('data/train/val_set.dill.xz', 'rb') as handle:
            val_set = dill.load(handle)
        # with open('data/train/train_set.dill', 'rb') as handle:
        #     train_set = dill.load(handle)
        # with open('data/train/val_set.dill', 'rb') as handle:
        #     val_set = dill.load(handle)

    #print(train_set[0])


    config = RIFT_Model_Config(
        model_type="RIFT",
        input_size=24,
        sequence_length=DAYS_LAG,
        n_targets=len(target_fns),
        input_ts_transform_list=[], #input_ts_transform_fns,
        tcn_num_channels=[450, 450, 450],
        tcn_kernel_size=3,
        tcn_dropout=0.1,
        positional_encoding=True,
        encoder_embed_dim=600,
        encoder_layers=3,
        add_tcn_timeseries_pool=True,
        take_embedding_conv=True,
        encoder_embedding_mean_pool=True,
        pre_encoder_fc_apply_layer_norm=True,
        fc_encoder_layers=[512, 256],
        fc_dropout=0.25,
        post_encoder_fc_layers=[512, 256],
        batch_size=32,
        sample_size=200,
        optimizer=RAdam,
        loss_function=torch.nn.L1Loss(),
        learning_rate=5e-4,
        l2_lambda=0.0,
        max_epochs=10,  # 10
        accumulation_steps=20,  # 20
        evaluate_every_n_steps=400,  # 400
        consecutive_losses_to_stop=3
    )

    param_search_grid = {
        #'tcn_num_channels': [[600, 600, 600]],  # [450, 450, 450]
        #'post_encoder_fc_layers': [[512, 256]]
    }
    model_trainers, res_dfs = train_RIFT_model(config, train_set, val_set, param_search_grid)
    breakpoint()
