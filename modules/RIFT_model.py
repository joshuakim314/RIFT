import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.modules import LayerNorm
import numpy as np

from modules.tcn import TemporalConvNet, TemporalBlock
from modules.train_utils import seq_corr_1d, seq_corr_3d


class RIFT_Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.siamese_encoder = RIFT_Encoder(config)
        self.post_encoder = build_linear_layers_from_list(
            self.siamese_encoder.last_layer_size, 
            self.config.post_encoder_fc_layers, 
            self.config.fc_dropout
        )
        self.last_layer_size = self.config.post_encoder_fc_layers[-1] if len(self.post_encoder) > 0 else self.siamese_encoder.last_layer_size

        # output layer initialized as a function to enable smooth switch from pre-training
        self.initialize_output_layer(self.config.n_targets)

    def initialize_output_layer(self, output_dim):
        # function to call when switching from pre-training
        self.full_output_layer = nn.Linear(self.last_layer_size, output_dim)

    def get_final_outputs(self, x):
        y_pred = torch.tanh(self.full_output_layer(x))
        return y_pred
    
    def forward(self, input, return_embeddings=False):
        x = input.copy()
        x = self.siamese_encoder.forward(x)
        for layer in self.post_encoder:
            x = layer(x)
            x = F.relu(x)
        if return_embeddings:
            return x
        # only take the output from the final timetep
        # can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        y_pred = self.get_final_outputs(x)
        return y_pred


class RIFT_Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.input_ts_transform_list = self.config.input_ts_transform_list
        self.input_size_with_features = self.config.input_size + self.config.input_size * len(self.config.input_ts_transform_list)
        self.return_full_embeddings = False
        
        self.tcn_embedder = TemporalConvNet(
            self.input_size_with_features,
            self.config.tcn_num_channels,
            self.config.tcn_kernel_size,
            self.config.tcn_dropout,
        )
        self.config.encoder_embed_dim = self.config.tcn_num_channels[-1]
        self.last_layer_size = self.config.encoder_embed_dim
        self.encoder_layers = nn.ModuleList([])
        for i in range(self.config.encoder_layers):
            self.encoder_layers.extend(
                [
                    TemporalConvNet(
                        self.input_size_with_features if i == 0 else self.config.tcn_num_channels[-1],
                        self.config.tcn_num_channels,
                        self.config.tcn_kernel_size,
                        self.config.tcn_dropout,
                    )
                ]
            )
        
        if "encoder_layers" in self.__dict__:
            assert len(self.encoder_layers) == (self.config.encoder_layers), "Attention layers list not generated correctly"
        
        # add a conv-pooling layer alongside the time-series dimension
        if self.config.add_tcn_timeseries_pool:
            # parametrize this as the last layer of the TCN encoder; 2^(i+1) would be the next dilation size
            dilation_size = 2**(len(self.config.tcn_num_channels) + 1)
            self.tcn_pooling_layer = TemporalBlock(
                n_inputs=self.config.encoder_embed_dim,
                n_outputs=1, kernel_size=self.config.tcn_kernel_size, stride=1, dilation=dilation_size,
                padding=(self.config.tcn_kernel_size - 1) * dilation_size, dropout=self.config.tcn_dropout
            )
            self.last_layer_size += self.config.sequence_length
        
        if config.pre_encoder_fc_apply_layer_norm:
            self.layer_norm = LayerNorm(self.last_layer_size)
        
        # build the feedforward encoder layers for each encoder
        self.encoder_fc_layers = build_linear_layers_from_list(
            self.last_layer_size, 
            self.config.fc_encoder_layers, 
            self.config.fc_dropout
        )
        
        print(f"final size of concatenated embeddings within the encoder is: {self.last_layer_size}")
        
        """
        individual encoder layers end here; the rest is post-encoder
        """
        
        # given the two embeddings will be concatenated, the input into the final layers will be double of the dimension of the last layer
        self.last_layer_size = 2 * self.config.fc_encoder_layers[-1]
        
        # if embeddings convs are being taken, an additional sequence_length vector is appended to the embeddings
        if self.config.take_embedding_conv:
            self.last_layer_size += self.config.sequence_length
    
    def forward(self, input):
        encoder_emb_0, full_encoder_embedding_0 = self.encoder_forward(input[0])
        encoder_emb_1, full_encoder_embedding_1 = self.encoder_forward(input[1])
        
        x = torch.cat((encoder_emb_0, encoder_emb_1), 1)  # concatenate the embeddings
        if self.config.take_embedding_conv:
            embed_corr = seq_corr_3d(full_encoder_embedding_0.transpose(0, 1), full_encoder_embedding_1.transpose(0, 1))
            x = torch.cat((x, embed_corr), 1)
        # self.logger.debug(f"shape of x before adding the per security mlp_embs: {x.shape}")
        return x
    
    def encoder_forward(self, x):
        for func in self.input_ts_transform_list:
            for i in range(0, 2):  # hardcoded since we have 2 time-series (price, log-return)
                feature_val = func(x[:, :, i].view(-1, self.config.sequence_length))
                x = torch.cat((x, feature_val.view(-1, self.config.sequence_length, 1)), 2)
        
        x = self.tcn_forward(x)
        # save output of tcn pooling
        if self.config.add_tcn_timeseries_pool:
            tcn_pooling_output = torch.squeeze(self.tcn_pooling_layer(x))
        # transpose to make shapes consistent between different encoder types
        x = x.transpose(0, 1).transpose(0, 2)
        
        if self.config.take_embedding_conv:
            full_encoder_embedding = x.clone()

        # mean pool or slice the encoder embedding
        if self.config.encoder_embedding_mean_pool:
            x = x.mean(0).view(-1, self.config.encoder_embed_dim)
        else:
            x = x[-1].view(-1, self.config.encoder_embed_dim)

        # and then cat the tcn pooling layer output to x
        if self.config.add_tcn_timeseries_pool:
            x = torch.cat((x, tcn_pooling_output), dim=1)
        
        if self.layer_norm:
            x = self.layer_norm(x)

        for layer in self.encoder_fc_layers:
            x = layer(x)
            x = F.relu(x)

        # if we are taking transformer_conv, return the full encoder embedding
        # alongside the mean-pooled / sliced and feedforwarded one
        if self.config.take_embedding_conv or self.return_full_embeddings:
            return (x, full_encoder_embedding)
        else:
            return (x, None)
    
    def tcn_forward(self, x):
        """Apply TCN on on x

        Args:
            x (torch.Tensor): shape = (batch_size, seq_len, input_dim)

        Returns:
            torch.Tensor: shape = (batch_size, seq_len)
        """
        tcn_emb = self.tcn_embedder.forward(x.transpose(1, 2))
        for i, layer in enumerate(self.encoder_layers):
            if i == 0:
                x = layer.forward(x.transpose(1, 2))
                if self.config.positional_encoding:
                    x = self.pos_encode(x, transpose=False)
            else:
                x = layer.forward(x)
        return tcn_emb
    
    def pos_encode(self, x, transpose=True):
        # if using model different from transformers (e.g. TCN), dims might be arranged differently and transposing is not needed
        if transpose:
            x = x.transpose(1, 2)
        emb_dim = x.shape[1]
        # make placeholder tensors using x for device consistency
        pe = x.new_zeros(self.config.sequence_length, emb_dim)
        position = x.new_zeros(self.config.sequence_length, 1)
        position[:, :] = torch.arange(
            0, self.config.sequence_length, dtype=torch.float).unsqueeze(1)
        div_term = x.new_zeros(emb_dim // 2)
        div_term[:] = torch.exp((torch.arange(
            0, emb_dim // 2 * 2, 2, dtype=torch.float) * -(np.log(10000.0) / (emb_dim // 2 * 2))))
        # 0 "pad" for odd embedding dim
        pe[:, 0:emb_dim // 2 * 2][:, 0::2] = torch.sin(position * div_term)
        pe[:, 0:emb_dim // 2 * 2][:, 1::2] = torch.cos(position * div_term)
        if emb_dim % 2 == 1:
            pe[:, -1] = 0
        pe = pe.unsqueeze(0).transpose(1, 2)
        x = x + pe
        if transpose:
            x = x.transpose(1, 2)
        return (x)


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


def build_linear_layers_from_list(last_layer_size, layer_list, dropout=None):
    # builds block of linear layers; applies batch norm if dropout is none
    layer_module_list = nn.ModuleList()
    for i, layer_size in enumerate(layer_list):
        append_block = nn.Sequential(
            Linear(last_layer_size, out_features=layer_size),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout else nn.BatchNorm1d(layer_size)
        )
        layer_module_list.append(append_block)
        last_layer_size = layer_size
    return(layer_module_list)
