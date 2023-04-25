class RIFT_Model_Config():
    def __init__(
        self,
        model_type,
        input_size,
        sequence_length,
        n_targets,
        input_ts_transform_list,
        tcn_num_channels,
        tcn_kernel_size,
        tcn_dropout,
        positional_encoding,
        encoder_embed_dim,
        encoder_layers,
        add_tcn_timeseries_pool,
        take_embedding_conv,
        encoder_embedding_mean_pool,
        pre_encoder_fc_apply_layer_norm,
        fc_encoder_layers,
        fc_dropout,
        post_encoder_fc_layers,
        batch_size,
        sample_size,
        optimizer,
        loss_function,
        learning_rate,
        l2_lambda,
        max_epochs,
        accumulation_steps,
        evaluate_every_n_steps,
        consecutive_losses_to_stop
    ):
        self.model_type = model_type
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.n_targets = n_targets
        self.input_ts_transform_list = input_ts_transform_list
        self.tcn_num_channels = tcn_num_channels
        self.tcn_kernel_size = tcn_kernel_size
        self.tcn_dropout = tcn_dropout
        self.positional_encoding = positional_encoding
        self.encoder_embed_dim = encoder_embed_dim
        self.encoder_layers = encoder_layers
        self.add_tcn_timeseries_pool = add_tcn_timeseries_pool
        self.take_embedding_conv = take_embedding_conv
        self.encoder_embedding_mean_pool = encoder_embedding_mean_pool
        self.pre_encoder_fc_apply_layer_norm = pre_encoder_fc_apply_layer_norm
        self.fc_encoder_layers = fc_encoder_layers
        self.fc_dropout = fc_dropout
        self.post_encoder_fc_layers = post_encoder_fc_layers
        
        # training parameters
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda
        self.max_epochs = max_epochs
        self.accumulation_steps = accumulation_steps
        self.evaluate_every_n_steps = evaluate_every_n_steps
        self.consecutive_losses_to_stop = consecutive_losses_to_stop
