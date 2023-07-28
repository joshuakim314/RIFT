import math
import os
import pickle
import sys
import math

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

def retrieve_annual_datasets(y_s, y_e):
    datasets = {}
    for y in [str(y) for y in range(y_s, y_e+1)]:
        with lzma.open('data/train/train_set_' + y + '.dill.xz', 'rb') as handle:
            datasets[y] = dill.load(handle)
    return datasets


if __name__ == '__main__':

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.empty_cache()
    print('CUDA available: ' + str(cuda_available))
    print(os.curdir)

    datasets = retrieve_annual_datasets(2012, 2012)
    dataset = datasets['2012']
    dataset_length = len(dataset)

    #train with frst 50%, validate with last 10% of year
    train_indices = list(range(0, int(0.5 * dataset_length)))
    val_indices = list(range(int(0.8 * dataset_length), int(0.9 * dataset_length)))

    #split the 2010 data into train/val (no shuffle), then sample 0.5% of each
    train_set, val_set = data.Subset(dataset, train_indices), data.Subset(dataset, val_indices)
    #train_size = math.floor(0.01 * len(train_set))
    #val_size = math.floor(0.00002 * len(val_set))
    #train_set, _ = data.random_split(train_set, [train_size, len(train_set)-train_size])
    #val_set, _ = data.random_split(val_set, [val_size, len(val_set)-val_size])
    print(f"dataset size: {dataset_length}")
    print(f"train_set size: {len(train_set)}")
    print(f"val_set size: {len(val_set)}")



    config = RIFT_Model_Config(
        model_type="RIFT",
        input_size=23,
        sequence_length=252,
        n_targets=6+1,
        input_ts_transform_list=[], #input_ts_transform_fns,
        tcn_num_channels=[600, 600, 600],
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
        max_epochs=100,
        accumulation_steps=20,  # 20
        evaluate_every_n_steps=10000,  # 400


        consecutive_losses_to_stop=5
    )

    param_search_grid = {
        #'tcn_num_channels': [[600, 600, 600]],  # [450, 450, 450]
        #'post_encoder_fc_layers': [[512, 256]]
    }
    model_trainers, res_dfs = train_RIFT_model(config, train_set, val_set, param_search_grid)
    with open('mlruns/results/rift_res_dfs.dill', 'wb') as handle:
        dill.dump(res_dfs, handle)
    #breakpoint()
