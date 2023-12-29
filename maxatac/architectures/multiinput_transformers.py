import logging
import numpy as np
import math
from scipy import stats
from maxatac.utilities.system_tools import Mute
from copy import deepcopy

with Mute():
    import tensorflow as tf
    from tensorflow.keras import backend as K
    from tensorflow.keras.callbacks import ModelCheckpoint
    from tensorflow.keras.layers import (
        Input,
        Conv1D,
        MaxPooling1D,
        AveragePooling1D,
        Lambda,
        BatchNormalization,
        Dense,
        Flatten,
    )
    from tensorflow.keras import initializers
    from tensorflow.keras.activations import relu, gelu
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam, AdamW, Lion

    # from maxatac.utilities.constants import KERNEL_INITIALIZER, INPUT_LENGTH, INPUT_CHANNELS, INPUT_FILTERS, \
    #    INPUT_KERNEL_SIZE, INPUT_ACTIVATION, OUTPUT_FILTERS, OUTPUT_KERNEL_SIZE, FILTERS_SCALING_FACTOR, DILATION_RATE, \
    #    OUTPUT_LENGTH, CONV_BLOCKS, PADDING, POOL_SIZE, ADAM_BETA_1, ADAM_BETA_2, DEFAULT_ADAM_LEARNING_RATE, \
    #    DEFAULT_ADAM_DECAY, NUM_HEADS, NUM_MHA, KEY_DIMS, D_FF, CONV_TOWER_CONFIGS, EMBEDDING_SIZE, POOL_SIZE_BEFORE_FLATTEN, \
    #    DOWNSAMPLE_METHOD_CONV_TOWER, INCEPTION_BRANCHES, WHOLE_ATTENTION_KWARGS, USE_RPE, DM_DROPOUT_RATE, CONV_TOWER_CONFIGS_FUSION

    from maxatac.utilities.constants import (
        KERNEL_INITIALIZER,
        INPUT_LENGTH,
        INPUT_CHANNELS,
        DNA_INPUT_CHANNELS,
        INPUT_FILTERS,
        INPUT_KERNEL_SIZE,
        INPUT_ACTIVATION,
        OUTPUT_FILTERS,
        OUTPUT_KERNEL_SIZE,
        FILTERS_SCALING_FACTOR,
        DILATION_RATE,
        OUTPUT_LENGTH,
        CONV_BLOCKS,
        PADDING,
        POOL_SIZE,
        ADAM_BETA_1,
        ADAM_BETA_2,
        DEFAULT_ADAM_LEARNING_RATE,
        DEFAULT_ADAM_DECAY,
        BASENJI_INPUT_KERNEL_SIZE,
        BASENJI_INPUT_FILTERS,
        ENFORMER_INPUT_KERNEL_SIZE,
        ENFORMER_INPUT_FILTERS,
        PREDICTION_HEAD_DROPOUT_RATE,
        RESIDUAL_CONNECTION_DROPOUT_RATE,
        DEFAULT_COSINEDECAYRESTARTS_FIRST_DECAY_STEPS,
        DEFAULT_COSINEDECAYRESTARTS_ALPHA,
        DEFAULT_COSINEDECAYRESTARTS_T_MUL,
        DEFAULT_COSINEDECAYRESTARTS_M_MUL,
        DEFAULT_COSINEDECAYRESTARTS_INITIAL_LR_MULTIPLIER,
        DEFAULT_COSINEDECAY_DECAY_STEPS,
        DEFAULT_COSINEDECAY_ALPHA,
        DEFAULT_COSINEDECAY_WARMUP_TARGET_MULTIPLIER,
        DEFAULT_COSINEDECAY_WARMUP_STEPS,
    )

    from maxatac.architectures.dcnn import (
        loss_function,
        loss_function_focal_class,
        dice_coef,
        dice_coef_class,
        loss_function_class,
        get_layer,
        get_residual_layer,
    )
    from maxatac.architectures.attention_module_TF import TransformerBlock
    import tensorflow_addons as tfa
    from tensorflow.keras.metrics import MeanMetricWrapper


def get_inception_block(inbound_layer, inception_branches, base_name):
    """
    Get an inception block. The motivation is that, the motif can have multiple lengths, so using the inception block with multiple paths may capture this pattern
    Let's say the input has shape (batch, seq_len, embed_dim), after inception it is (batch, seq_len, embed_dim)
    The 4 filter sizes can be 7, 10, 13, and 16, with padding same and stride 1
    We can use 1x1 conv to reduce the embed dim like in the paper
    """
    outputs = []
    embed_dim = inbound_layer.shape[-1]
    assert (
        embed_dim % len(inception_branches) == 0
    ), "Embed dim has to be a multiple of number of branches"
    for i, branch in enumerate(inception_branches):
        # Each branch is a conv block with a conv + batch_norm + activation
        temp_layer = inbound_layer
        for j, block in enumerate(branch):
            if block["name"] == "conv":
                temp_layer = get_conv_block(
                    temp_layer, block, f"{base_name}_branch_{i}_block_{j}"
                )
            else:
                temp_layer = AveragePooling1D(
                    pool_size=block["pool_size"],
                    strides=block["stride"],
                    padding=block["padding"],
                    name=f"{base_name}_avgpool_branch_{i}_block_{j}",
                )(temp_layer)

        outputs.append(temp_layer)

    # Reshape the output into the desired shape
    output_layer = tf.keras.layers.Concatenate(axis=-1)(outputs)
    return output_layer


def get_conv_block(
    inbound_layer,
    conv_block_config,
    base_name,
    dropout_rate=None,
    use_residual=False,
    suppress_activation=False,
    pre_activation=False,
    regularization=False,
    l1=0.0,
    l2=0.0,
    override_activation=None
):
    """
    Feed the input through some conv layers.
    This function is very similar to Tareian's get_layer function, with just some slight modification
    """
    for l in range(conv_block_config["num_layer"]):
        if override_activation==None:
            activation = tf.keras.layers.Activation(
                conv_block_config["activation"], name=base_name + f"_act_{l+1}"
            )
        else:
            activation = tf.keras.layers.Activation(
                override_activation, name=base_name + f"_act_{l + 1}"
            )
        if pre_activation:
            if l == 0:
                inbound_layer = activation(inbound_layer)
            else:
                if suppress_activation == False:
                    inbound_layer = activation(inbound_layer)
            inbound_layer = Conv1D(
                filters=conv_block_config["num_filters"],
                kernel_size=conv_block_config["kernel"],
                padding=conv_block_config["padding"],
                strides=conv_block_config["stride"],
                activation="linear",
                name=base_name + f"_conv_layer_{l+1}",
                kernel_regularizer=tf.keras.regularizers.L1L2(l1, l2)
                if regularization
                else None,
            )(inbound_layer)
        else:
            inbound_layer = Conv1D(
                filters=conv_block_config["num_filters"],
                kernel_size=conv_block_config["kernel"],
                padding=conv_block_config["padding"],
                strides=conv_block_config["stride"],
                activation="linear",
                name=base_name + f"_conv_layer_{l+1}",
            )(inbound_layer)
        inbound_layer = BatchNormalization(name=base_name + f"_batch_norm_{l+1}")(
            inbound_layer
        )
        if l == 0:
            if dropout_rate != None:
                inbound_layer = tf.keras.layers.Dropout(rate=dropout_rate)(
                    inbound_layer
                )
    if suppress_activation == False:
        inbound_layer = activation(inbound_layer)
    return inbound_layer


def get_conv_tower(
    inbound_layer,
    conv_tower_configs,
    downsample_method,
    base_name,
    use_residual=False,
    suppress_activation=False,
    pre_activation=False,
    residual_connection_dropout_rate=None,
    override_activation=None
):
    """
    Feed the input through the tower of conv layers
    Input has shape (batch_size, 1024, 16)
    """
    count = 0
    for conv_block_config in conv_tower_configs:
        count += 1
        # print(f"before conv block: {inbound_layer.shape}")
        if use_residual:
            inbound_layer = get_residual_layer(
                inbound_layer,
                get_conv_block(
                    inbound_layer,
                    conv_block_config,
                    base_name=f"{base_name}_B_{count}",
                    use_residual=use_residual,
                    suppress_activation=suppress_activation,
                    pre_activation=pre_activation,
                    dropout_rate=residual_connection_dropout_rate,
                    override_activation=override_activation
                ),
                activation="relu" if override_activation==None else override_activation
            )
        else:
            inbound_layer = get_conv_block(
                inbound_layer,
                conv_block_config,
                base_name=f"{base_name}_B_{count}",
                use_residual=use_residual,
                suppress_activation=suppress_activation,
                pre_activation=pre_activation,
                override_activation=override_activation
            )
        # print(f"after conv block: {inbound_layer.shape}")
        # After each conv block, use maxpooling to reduce seq len by 2
        # set option to downsample whether with maxpooling or conv1d stride 2
        if downsample_method == "maxpooling":
            inbound_layer = MaxPooling1D(
                pool_size=5,
                strides=2,
                padding="same",
                name=f"{base_name}_B_{count}_DS_maxpool",
            )(inbound_layer)
            inbound_layer = BatchNormalization()(inbound_layer)
        else:
            inbound_layer = Conv1D(
                filters=conv_block_config["num_filters"],
                kernel_size=conv_block_config["kernel"],
                strides=2,
                padding="same",
                name=f"{base_name}_B_{count}_DS_conv",
            )(inbound_layer)
            inbound_layer = BatchNormalization()(inbound_layer)

    return inbound_layer


@tf.keras.utils.register_keras_serializable()
class Swish(tf.keras.layers.Layer):
    def __init__(self, beta=1.0, *args, **kwargs):
        super(Swish, self).__init__(*args, **kwargs)
        self._beta = beta

    def build(self, input_shape):
        self.beta = tf.Variable(
            initial_value=tf.constant(self._beta, dtype="float32"), trainable=True
        )

    def call(self, inputs):
        return inputs * tf.sigmoid(self.beta * inputs)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "beta": self._beta,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable()
class SwiGlu(tf.keras.layers.Layer):
    def __init__(self, beta=1.0, units=None, *args, **kwargs):
        super(SwiGlu, self).__init__(*args, **kwargs)
        self._beta = beta
        self.units=units

    def build(self, input_shape):
        self.swish = Swish(beta=self._beta)
        self.W = tf.keras.layers.Dense(
            units=input_shape[-1] if self.units==None else self.units,
            use_bias=True,
            bias_initializer="glorot_uniform",
            name=f"{self.name}/W_c",
        )
        self.V = tf.keras.layers.Dense(
            units=input_shape[-1] if self.units==None else self.units,
            use_bias=True,
            bias_initializer="glorot_uniform",
            name=f"{self.name}/V_c",
        )

    def call(self, inputs):
        return self.V(inputs) * self.swish(self.W(inputs))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "beta": self._beta,
            }
        )
        return config


def get_positional_encoding(inbound_layer, seq_len, depth, n=10000):
    """
    Return a positional encoding for the transformer,
    Input is the matrix I of shape (None, seq_len, embed_size)
    The function generates an pos encoding matrix P with shape (None, seq_len, embed_size)
    The output is I + P
    """
    # Get a pos encoding layer (taken from https://www.tensorflow.org/text/tutorials/transformer#the_embedding_and_positional_encoding_layer)
    depth = depth / 2

    positions = np.arange(seq_len)[:, np.newaxis]  # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :] / depth  # (1, depth)

    angle_rates = 1 / (n**depths)  # (1, depth)
    angle_rads = positions * angle_rates  # (pos, depth)

    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)

    pos_encoding = tf.cast(pos_encoding, dtype=tf.float32)
    # Use Add layer to add the name to the layer
    inbound_layer = tf.keras.layers.Add(name="Add_positional_encoding")(
        [inbound_layer, pos_encoding[tf.newaxis, :seq_len, :]]
    )
    # inbound_layer = inbound_layer + pos_encoding[tf.newaxis, :seq_len, :]

    return inbound_layer


def get_multihead_attention(inbound_layer, key_dim, num_heads):
    """
    Get a multi head attention module
    """
    # Check correct dimensions
    embed_dim = inbound_layer.shape[-1]
    assert (
        embed_dim == key_dim * num_heads
    ), "Embedding size has to be equal to key_dim * num heads"
    mha, att_scores = tf.keras.layers.MultiHeadAttention(
        key_dim=key_dim, num_heads=num_heads
    )(query=inbound_layer, value=inbound_layer, return_attention_scores=True)
    layer_norm = tf.keras.layers.LayerNormalization()(mha)
    inbound_layer = tf.keras.layers.Add()([inbound_layer, layer_norm])

    return inbound_layer, att_scores


def get_feed_forward_nn(
    inbound_layer,
    d_ff,
    base_name,
    activation_function="relu",
    bias1=True,
    bias2=True,
):
    """
    Get the feed forward neural net right after multihead attention
    inbound_layer has shape (batch_size, seq_len, embed_dim)
    Implementation taken from https://nn.labml.ai/transformers/feed_forward.html
    From the original paper, the output from FFNN is also layer_normed and add residual
    """
    if activation_function == "relu":
        activation = tf.keras.layers.ReLU(name=base_name + "_relu")

    embed_dim = inbound_layer.shape[-1]
    dense_1 = tf.keras.layers.Dense(d_ff, use_bias=bias1, name=base_name + "_dense_1")
    dense_2 = tf.keras.layers.Dense(
        embed_dim, use_bias=bias2, name=base_name + "_dense_2"
    )
    layer_norm = tf.keras.layers.LayerNormalization(
        name=base_name + "_layernorm_in_ffnn"
    )
    residual = tf.keras.layers.Add(name=base_name + "_residual_in_ffnn")

    ffnn_output = activation(dense_1(inbound_layer))
    ffnn_output = dense_2(ffnn_output)
    inbound_layer = residual([inbound_layer, layer_norm(ffnn_output)])
    # inbound_layer = inbound_layer + layer_norm(ffnn_output)

    return inbound_layer


def get_multihead_attention_custom(
    inbound_layer, key_dim, num_heads, seq_len, base_name
):
    """
    Make a custom multi head attention layer
    inbound_layer has shape (batch_size, seq_len, embed_dim)
    Implementation taken from https://nn.labml.ai/transformers/mha.html
    """
    embed_dim = inbound_layer.shape[-1]
    assert (
        embed_dim == key_dim * num_heads
    ), "Embedding size has to be equal to key_dim * num heads"
    query = tf.keras.layers.Dense(embed_dim, name=base_name + "_Wq")
    key = tf.keras.layers.Dense(embed_dim, name=base_name + "_Wk")
    value = tf.keras.layers.Dense(embed_dim, name=base_name + "_Wv")

    wq = query(inbound_layer)
    wk = key(inbound_layer)
    wv = value(inbound_layer)
    # Reshape this from (batch_size, seq_len, embed_dim) to (batch_size, num_heads, seq_len, key_dim)
    wq = tf.reshape(wq, (-1, num_heads, seq_len, key_dim))
    wk = tf.reshape(wk, (-1, num_heads, seq_len, key_dim))
    wv = tf.reshape(wv, (-1, num_heads, seq_len, key_dim))

    # Take the dot product of wq and wk.T to produce (batch_size, num_heads, seq_len, seq_len)
    att_weights = tf.linalg.matmul(wq, tf.transpose(wk, perm=[0, 1, 3, 2]))
    att_weights = att_weights * (1 / math.sqrt(key_dim))
    att_weights = tf.keras.layers.Softmax(
        axis=1, name=base_name + "_softmax_att_weights"
    )(att_weights)

    # Multiply this with value to produce (batch_size, num_heads, seq_len, key_dim)
    output = tf.linalg.matmul(att_weights, wv)

    # Reshape to (batch_size, seq_len, embed_dim) and pass to another dense layer
    output = tf.reshape(output, (-1, seq_len, embed_dim))
    output = tf.keras.layers.Dense(embed_dim, name=base_name + "_dense_after_mha")(
        output
    )

    # Pass through layer norm and add residual
    layer_norm = tf.keras.layers.LayerNormalization(
        name=base_name + "_layernorm_in_mha"
    )(output)
    output = tf.keras.layers.Add(name=base_name + "_residual_in_mha")(
        [inbound_layer, layer_norm]
    )
    # inbound_layer = inbound_layer + layer_norm

    return output, att_weights


def get_multiinput_transformer(
    output_activation,
    model_config,
    adam_learning_rate=DEFAULT_ADAM_LEARNING_RATE,
    adam_decay=DEFAULT_ADAM_DECAY,
    input_length=INPUT_LENGTH,
    input_channels=INPUT_CHANNELS,
    input_filters=INPUT_FILTERS,
    input_kernel_size=INPUT_KERNEL_SIZE,
    input_activation=INPUT_ACTIVATION,
    output_filters=OUTPUT_FILTERS,
    output_kernel_size=OUTPUT_KERNEL_SIZE,
    filters_scaling_factor=FILTERS_SCALING_FACTOR,
    dilation_rate=DILATION_RATE,
    output_length=OUTPUT_LENGTH,
    conv_blocks=CONV_BLOCKS,
    padding=PADDING,
    pool_size=POOL_SIZE,
    adam_beta_1=ADAM_BETA_1,
    adam_beta_2=ADAM_BETA_2,
    target_scale_factor=1,
    dense_b=False,
    weights=None,
):
    """
    If weights are provided they will be loaded into created model
    """
    logging.debug("Building Dilated CNN model")

    # Current there are two inputs: one for the genome sequence, one for the ATAC-seq signal
    _input = Input(
        shape=(input_length, INPUT_CHANNELS),
    )

    genome_input = tf.keras.layers.Lambda(
        lambda x: x[:, :, :DNA_INPUT_CHANNELS], name="genome"
    )(_input)
    atacseq_input = tf.keras.layers.Lambda(
        lambda x: x[:, :, DNA_INPUT_CHANNELS:], name="atac"
    )(_input)

    # The current feature dim to the transformer is 64
    # Using 2 inputs, each input will be transformed to feature dim of 32
    # Then they are concatenated and passed through another conv layer to keep the same 64
    # genome_layer = genome_input
    atacseq_layer = atacseq_input
    filters = input_filters  # redefined in encoder/decoder loops

    if (
        "USING_BASENJI_KERNEL" in model_config.keys()
        and model_config["USING_BASENJI_KERNEL"]
    ):
        genome_layer = get_layer(
            inbound_layer=genome_input,
            filters=BASENJI_INPUT_FILTERS,
            kernel_size=BASENJI_INPUT_KERNEL_SIZE,
            activation="linear",
            padding=padding,
            dilation_rate=1,
            kernel_initializer=KERNEL_INITIALIZER,
            n=1,
            use_bias=False,
            name="injected_first_conv_kernel",
            skip_batch_norm=True,
        )

        if model_config["KERNEL_REPLACING"] == False:
            genome_layer_ext = get_layer(
                inbound_layer=genome_input,
                filters=filters,
                kernel_size=input_kernel_size,
                activation="linear",
                padding=padding,
                dilation_rate=1,
                kernel_initializer=KERNEL_INITIALIZER,
                n=1,
                name="first_conv_kernel",
                skip_batch_norm=True,
            )
            genome_layer = tf.keras.layers.Concatenate(axis=-1)(
                [genome_layer, genome_layer_ext]
            )

    elif (
        "USING_ENFORMER_KERNEL" in model_config.keys()
        and model_config["USING_ENFORMER_KERNEL"]
    ):
        genome_layer = get_layer(
            inbound_layer=genome_input,
            filters=ENFORMER_INPUT_FILTERS,
            kernel_size=ENFORMER_INPUT_KERNEL_SIZE,
            activation="linear",
            padding=padding,
            dilation_rate=1,
            kernel_initializer=KERNEL_INITIALIZER,
            n=1,
            use_bias=False,
            name="injected_first_conv_kernel",
            skip_batch_norm=True,
        )

        if model_config["KERNEL_REPLACING"] == False:
            genome_layer_ext = get_layer(
                inbound_layer=genome_input,
                filters=filters,
                kernel_size=input_kernel_size,
                activation="linear",
                padding=padding,
                dilation_rate=1,
                kernel_initializer=KERNEL_INITIALIZER,
                n=1,
                name="first_conv_kernel",
                skip_batch_norm=True,
            )
            genome_layer = tf.keras.layers.Concatenate(axis=-1)(
                [genome_layer, genome_layer_ext]
            )
    else:
        genome_layer = get_layer(
            inbound_layer=genome_input,
            filters=filters,
            kernel_size=input_kernel_size,
            activation="linear",
            padding=padding,
            dilation_rate=1,
            kernel_initializer=KERNEL_INITIALIZER,
            n=1,
            name="first_conv_kernel",
            skip_batch_norm=True,
        )

    atacseq_layer1 = get_layer(
        inbound_layer=atacseq_layer,
        filters=filters,
        kernel_size=input_kernel_size,
        activation="linear",
        padding=padding,
        dilation_rate=1,
        kernel_initializer=KERNEL_INITIALIZER,
        n=1,
        skip_batch_norm=True,
    )

    # Compress features
    genome_layer = get_layer(
        inbound_layer=genome_layer,
        filters=model_config["CONV_TOWER_CONFIGS_FUSION"]["num_filters"],
        kernel_size=input_kernel_size,
        activation="relu" if model_config["OVERRIDE_ACTIVATION"]==None else model_config["OVERRIDE_ACTIVATION"],
        padding=padding,
        dilation_rate=1,
        kernel_initializer=KERNEL_INITIALIZER,
        n=1,
        pre_activation=True,
        name="compress_feature_channel_in_genome",
    )
    atacseq_layer2 = get_layer(
        inbound_layer=atacseq_layer1,
        filters=model_config["CONV_TOWER_CONFIGS_FUSION"]["num_filters"],
        kernel_size=input_kernel_size,
        activation="relu" if model_config["OVERRIDE_ACTIVATION"]==None else model_config["OVERRIDE_ACTIVATION"],
        padding=padding,
        dilation_rate=1,
        kernel_initializer=KERNEL_INITIALIZER,
        n=1,
        pre_activation=True,
        name="compress_feature_channel_in_signal",
    )

    # Get the conv tower for each branch
    genome_layer = get_conv_tower(
        genome_layer,
        model_config["CONV_TOWER_CONFIGS_FUSION"]["genome"],
        model_config["DOWNSAMPLE_METHOD_CONV_TOWER"],
        base_name="GENOME_tower",
        use_residual=model_config["CONV_TOWER_CONFIGS_FUSION"]["use_residual"],
        suppress_activation=True,
        pre_activation=True,
        residual_connection_dropout_rate=model_config[
            "RESIDUAL_CONNECTION_DROPOUT_RATE"
        ]
        if model_config["SUPPRESS_DROPOUT"] == False
        else None,
        override_activation=model_config["OVERRIDE_ACTIVATION"]
    )
    atacseq_layer = get_conv_tower(
        atacseq_layer2,
        model_config["CONV_TOWER_CONFIGS_FUSION"]["atac"],
        model_config["DOWNSAMPLE_METHOD_CONV_TOWER"],
        base_name="ATAC_tower",
        use_residual=model_config["CONV_TOWER_CONFIGS_FUSION"]["use_residual"],
        suppress_activation=True,
        pre_activation=True,
        residual_connection_dropout_rate=model_config[
            "RESIDUAL_CONNECTION_DROPOUT_RATE"
        ]
        if model_config["SUPPRESS_DROPOUT"] == False
        else None,
        override_activation=model_config["OVERRIDE_ACTIVATION"]
    )

    # genome_layer and atacseq_layer now should have shape (batch, seq_len, mha_embed_dim // 2)
    # Concatenate the two and pass it through another conv layer
    layer = tf.keras.layers.Concatenate(axis=1)([genome_layer, atacseq_layer])

    # if model_config["CONV_TOWER_CONFIGS_FUSION"]["merge"] != {}:
    #     layer = get_conv_block(
    #         layer,
    #         model_config["CONV_TOWER_CONFIGS_FUSION"]["merge"],
    #         "Intermediate_fusion_conv",
    #         use_residual=True,
    #         suppress_activation=True,
    #         pre_activation=True,
    #     )

    # get seq_len after the conv tower
    seq_len = layer.shape[1]

    if not model_config["USE_RPE"]:
        # A positional encoding layer
        layer = get_positional_encoding(
            layer, seq_len=seq_len, depth=model_config["EMBEDDING_SIZE"]
        )

        # Stack a list of encoders
        for i in range(model_config["NUM_MHA"]):
            # The weights are in the shape of (batch_size, num_head, seq_len, seq_len)
            layer, att_weights = get_multihead_attention_custom(
                layer,
                model_config["KEY_DIMS"],
                model_config["NUM_HEADS"],
                seq_len,
                base_name=f"Encoder_{i}",
            )
            layer = get_feed_forward_nn(
                layer, model_config["D_FF"], base_name=f"Encoder_{i}"
            )

    else:
        new_rpe_attention_kwargs = deepcopy(model_config["WHOLE_ATTENTION_KWARGS"])
        new_rpe_attention_kwargs["initializer"] = initializers.get(
            model_config["WHOLE_ATTENTION_KWARGS"]["initializer"]
        )
        for i in range(model_config["NUM_MHA"]):
            deepmind_transformer_block = TransformerBlock(
                channels=model_config["EMBEDDING_SIZE"],
                dropout_rate=model_config["DM_DROPOUT_RATE"],
                attention_kwargs=new_rpe_attention_kwargs,
                name=f"Transformer_block_{i}",
            )
            outputs = deepmind_transformer_block(layer)
            # logging.error(f"Length of transformer block output: {len(outputs)}")
            layer = outputs[0]

    # Final postprocessing

    # use only sequence side
    _offset = layer.shape[1] // 2
    _channel = layer.shape[-1]

    if model_config["FULL_TRANSFORMER_OUTPUT"] == False:
        layer = tf.keras.layers.Lambda(
            lambda x: x[:, :_offset, :], name="Extract_sequence_region_INFO"
        )(layer)
    else:
        layer = tf.keras.layers.Conv1D(
            filters=_channel,
            kernel_size=2,
            padding="valid",
            dilation_rate=_offset,
            name="combine_seq_and_signal",
        )(layer)

    _prediction_head_config = {
        "activation": "relu",
        "kernel": 10,
        "num_filters": 64,
        "num_layer": 1,
        "padding": "same",
        "stride": 1,
    }
    model_config["prediction_head_config"] = _prediction_head_config

    layer = get_conv_block(
        layer,
        _prediction_head_config,
        base_name="Pre_prediction_head",
        dropout_rate=model_config["PREDICTION_HEAD_DROPOUT_RATE"],
        suppress_activation=True,
        use_residual=False,
        pre_activation=True,
        regularization=model_config["REGULARIZATION"],
        l1=model_config["ELASTIC_L1"],
        l2=model_config["ELASTIC_L2"],
        override_activation=model_config["OVERRIDE_ACTIVATION"]
    )

    # Outputs
    layer_dilation_rate = dilation_rate[0]
    # if dense_b:
    #     output_layer = get_layer(
    #         inbound_layer=layer,
    #         filters=output_filters,
    #         kernel_size=output_kernel_size,
    #         activation=input_activation,
    #         padding=padding,
    #         dilation_rate=layer_dilation_rate,
    #         kernel_initializer=KERNEL_INITIALIZER,
    #         skip_batch_norm=True,
    #         n=1,
    #     )
    # else:
    #     output_layer = get_layer(
    #         inbound_layer=layer,
    #         filters=_prediction_head_config["num_filters"],
    #         kernel_size=output_kernel_size,
    #         activation='linear',
    #         padding=padding,
    #         dilation_rate=layer_dilation_rate,
    #         kernel_initializer=KERNEL_INITIALIZER,
    #         skip_batch_norm=True,
    #         n=1,
    #         focal_initializing=model_config["FOCAL_LOSS"],
    #     ) # N, 256, 1

    atacseq_layer1_compressed=tf.keras.layers.Conv1D(filters=32,kernel_size=4,strides=4, padding='valid')(atacseq_layer1) # N, 1024, 16
    atacseq_layer1_compressed=SwiGlu(units=_prediction_head_config["num_filters"])(atacseq_layer1_compressed)
    output_layer_weighted=tf.keras.layers.Multiply()([layer,atacseq_layer1_compressed])
    output_layer=tf.keras.layers.Conv1D(filters=output_filters,
                                        kernel_size=output_kernel_size,
                                        activation=output_activation,
                                        padding=padding,
                                        dilation_rate=layer_dilation_rate,
                                        kernel_initializer=KERNEL_INITIALIZER,
                                        )(output_layer_weighted)

    # Downsampling from 1024 to 32 (change this) for a dynamic change
    seq_len = output_layer.shape[1]  # should be 256 now
    # strides = ((seq_len - model_config["POOL_SIZE_BEFORE_FLATTEN"] + 1) // output_length) + 1

    output_layer = tf.keras.layers.MaxPooling1D(
        pool_size=model_config["POOL_SIZE_BEFORE_FLATTEN"],
        strides=model_config["POOL_SIZE_BEFORE_FLATTEN"],
        padding="valid",
    )(output_layer)

    # Depending on the output activation functions, model outputs need to be scaled appropriately
    output_layer = Flatten()(output_layer)

    if dense_b:
        output_layer = Dense(
            output_length,
            activation=output_activation,
            kernel_initializer="glorot_uniform",
        )(output_layer)

    logging.debug("Added outputs layer: " + "\n - " + str(output_layer))

    # Model
    model = Model(inputs=[_input], outputs=output_layer)

    # lr schedule
    if model_config["COSINEDECAYRESTARTS"]:
        lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=model_config["INITIAL_LEARNING_RATE"]
            * DEFAULT_COSINEDECAYRESTARTS_INITIAL_LR_MULTIPLIER,
            first_decay_steps=model_config["COSINEDECAYRESTARTS_FIRST_DECAY_STEPS"],
            t_mul=DEFAULT_COSINEDECAYRESTARTS_T_MUL,
            m_mul=DEFAULT_COSINEDECAYRESTARTS_M_MUL,
            alpha=DEFAULT_COSINEDECAYRESTARTS_ALPHA,
        )
    elif model_config["COSINEDECAY"]:
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=model_config["INITIAL_LEARNING_RATE"],
            decay_steps=model_config["COSINEDECAYDECAYSTEPS"],
            alpha=model_config["COSINEDECAYALPHA"],
            name=None,
            warmup_target=model_config["INITIAL_LEARNING_RATE"]
            * DEFAULT_COSINEDECAY_WARMUP_TARGET_MULTIPLIER,
            warmup_steps=DEFAULT_COSINEDECAY_WARMUP_STEPS,
        )
    else:
        lr_schedule = (
            model_config["INITIAL_LEARNING_RATE"]
            if model_config["OPTIMIZER"] != "Lion"
            else model_config["INITIAL_LEARNING_RATE"] / 3.0
        )

    if (
        "USING_BASENJI_KERNEL" in model_config.keys()
        and model_config["USING_BASENJI_KERNEL"]
    ):
        try:
            import pickle

            pickled_weight_file = model_config["BASENJI_KERNEL_PICKLED_FILE"]
            with open(pickled_weight_file, "rb") as f:
                _weights = pickle.load(f)
            for layer in model.layers:
                if layer.name == "injected_first_conv_kernel":
                    layer.set_weights(_weights)
                    layer.trainable = model_config["BASENJI_KERNEL_TRAINABLE"]
        except:
            logging.error("Failed to set BASENJI kernel weights.")
            raise ValueError("Failed to set BASENJI kernel weights.")
    if (
        "USING_ENFORMER_KERNEL" in model_config.keys()
        and model_config["USING_ENFORMER_KERNEL"]
    ):
        try:
            import pickle

            pickled_weight_file = model_config["ENFORMER_KERNEL_PICKLED_FILE"]
            with open(pickled_weight_file, "rb") as f:
                _weights = pickle.load(f)
            for layer in model.layers:
                if layer.name == "injected_first_conv_kernel":
                    layer.set_weights(_weights)
                    layer.trainable = model_config["ENFORMER_KERNEL_TRAINABLE"]
        except:
            logging.error("Failed to set ENFORMER kernel weights.")
            raise ValueError("Failed to set ENFORMER kernel weights.")

    if model_config["OPTIMIZER"] == "Adam":
        optimizer = Adam(
            learning_rate=lr_schedule,
            beta_1=adam_beta_1,
            beta_2=adam_beta_2,
            weight_decay=adam_decay,
        )
    elif model_config["OPTIMIZER"] == "AdamW":
        optimizer = AdamW(
            learning_rate=lr_schedule,
            beta_1=adam_beta_1,
            beta_2=adam_beta_2,
            weight_decay=adam_decay,
        )
    elif model_config["OPTIMIZER"] == "Lion":
        optimizer = Lion(
            learning_rate=lr_schedule,
            beta_1=adam_beta_1,
            beta_2=0.99,
            # weight_decay=adam_decay,
        )

    if model_config["FOCAL_LOSS"] == False:
        model.compile(
            optimizer,
            loss=loss_function_class(
                flanking_truncation_size=model_config["LOSS_FLANKING_TRUNCATION_SIZE"],
            ),
            weighted_metrics=[
                MeanMetricWrapper(
                    dice_coef_class(
                        flanking_truncation_size=model_config[
                            "LOSS_FLANKING_TRUNCATION_SIZE"
                        ],
                        unknown_coef= model_config['dice_unknown_coef']
                    ),
                    name="dice_coef",
                )
            ],
        )
    else:
        model.compile(
            optimizer,
            loss=loss_function_focal_class(
                alpha=model_config["FOCAL_LOSS_ALPHA"],
                gamma=model_config["FOCAL_LOSS_GAMMA"],
                apply_class_balancing=model_config["FOCAL_LOSS_APPLY_ALPHA"],
                flanking_truncation_size=model_config["LOSS_FLANKING_TRUNCATION_SIZE"],
            ),
            weighted_metrics=[
                MeanMetricWrapper(
                    dice_coef_class(
                        flanking_truncation_size=model_config[
                            "LOSS_FLANKING_TRUNCATION_SIZE"
                        ],
                        unknown_coef= model_config['dice_unknown_coef']
                    ),
                    name="dice_coef",
                )
            ],
        )

    logging.debug("Model compiled")

    if weights is not None and weights != "":
        logging.error(f"The weights: {weights}")
        model.load_weights(weights)
        logging.debug("Weights loaded")

    return model
