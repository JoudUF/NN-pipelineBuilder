# =====================================================
# Layer Configuration Metadata
# =====================================================
# This module defines what parameters each layer type
# requires from the user, their types, defaults, and
# display labels. The GUI reads this to dynamically
# build parameter forms.
# =====================================================

LAYER_CATEGORIES = {
    "Input & Embedding": [
        "embedding",
        "linear",
    ],
    "Sequence Packing": [
        "pack_sequence",
        "unpack_sequence",
    ],
    "Core Recurrent": [
        "lstm",
        "gru",
        "rnn",
    ],
    "1D Convolution": [
        "conv1d",
    ],
    "Pooling": [
        "maxpool1d",
        "avgpool1d",
        "adaptive_avgpool1d",
        "adaptive_maxpool1d",
        "mean_pool",        
        "max_pool_time",    
    ],
    "Attention": [
        "multihead_attention",
    ],
    "Normalization": [
        "layernorm",
        "batchnorm1d",
    ],
    "Regularization": [
        "dropout",
    ],
    "Reshape Operations": [
        "permute",
        "flatten",
        "squeeze",
        "unsqueeze",
        "slice_last",
        "split",            
        "reshape",          
    ],
    "Combine Operations": [
        "residual_add",
        "cat",
        "bmm",              
        "stack",            
    ],
    "Recurrent State Extraction": [
        "select_hidden",
        "select_cell",
    ],
    "Direction Operations": [     
        "sum_directions",
        "split_directions",
    ],
    "Activations": [
        "relu",
        "leaky_relu",
        "gelu",
        "tanh",
        "sigmoid",
        "softmax",
        "log_softmax",
    ],
}

# Display names for the GUI
LAYER_DISPLAY_NAMES = {
    "embedding":            "nn.Embedding",
    "linear":               "nn.Linear",
    "pack_sequence":        "pack_padded_sequence",
    "unpack_sequence":      "pad_packed_sequence",
    "lstm":                 "nn.LSTM",
    "gru":                  "nn.GRU",
    "rnn":                  "nn.RNN",
    "conv1d":               "nn.Conv1d",
    "maxpool1d":            "nn.MaxPool1d",
    "avgpool1d":            "nn.AvgPool1d",
    "adaptive_avgpool1d":   "nn.AdaptiveAvgPool1d",
    "adaptive_maxpool1d":   "nn.AdaptiveMaxPool1d",
    "multihead_attention":  "nn.MultiheadAttention",
    "layernorm":            "nn.LayerNorm",
    "batchnorm1d":          "nn.BatchNorm1d",
    "dropout":              "nn.Dropout",
    "permute":              "Permute / Transpose",
    "flatten":              "nn.Flatten",
    "squeeze":              "Squeeze",
    "unsqueeze":            "Unsqueeze",
    "slice_last":           "Slice Last Timestep",
    "residual_add":         "Residual Add",
    "cat":                  "torch.cat (Concatenate)",
    "relu":                 "nn.ReLU",
    "leaky_relu":           "nn.LeakyReLU",
    "gelu":                 "nn.GELU",
    "tanh":                 "nn.Tanh",
    "sigmoid":              "nn.Sigmoid",
    "softmax":              "nn.Softmax",
    "log_softmax":          "nn.LogSoftmax",
    "sum_directions":       "Sum Directions (fwd+bwd)",
    "split_directions":     "Split Directions",
    "select_hidden":        "Select Hidden State (h_n)",
    "select_cell":          "Select Cell State (c_n)",
    "mean_pool":            "Mean Pool (over time)",
    "max_pool_time":        "Max Pool (over time)",
    "bmm":                  "torch.bmm (Batch Matmul)",
    "stack":                "torch.stack",
    "split":                "torch.split / chunk",
    "reshape":              "torch.reshape / view",
}

# =====================================================
# Parameter definitions per layer type
# Each entry is a list of dicts describing one parameter:
#   - "name":    the kwarg name used in the Fact
#   - "label":   display label for the GUI
#   - "type":    "int", "float", "bool", "choice"
#   - "default": default value (None means required)
#   - "options": list of options for "choice" type
# =====================================================

LAYER_PARAMS = {
    # --- Input & Embedding ---
    "embedding": [
        {"name": "vocab_size",     "label": "Vocabulary Size",   "type": "int",   "default": None},
        {"name": "embedding_dim",  "label": "Embedding Dim",     "type": "int",   "default": None},
    ],
    "linear": [
        {"name": "out_features",   "label": "Output Features",   "type": "int",   "default": None},
    ],

    # --- Sequence Packing (no user params) ---
    "pack_sequence":   [],
    "unpack_sequence": [],

    # --- Core Recurrent ---
    "lstm": [
        {"name": "hidden_size",    "label": "Hidden Size",       "type": "int",   "default": None},
        {"name": "num_layers",     "label": "Number of Layers",  "type": "int",   "default": 1},
        {"name": "bidirectional",  "label": "Bidirectional",     "type": "bool",  "default": False},
        {"name": "dropout",        "label": "Dropout (between layers)", "type": "float", "default": 0.0},
    ],
    "gru": [
        {"name": "hidden_size",    "label": "Hidden Size",       "type": "int",   "default": None},
        {"name": "num_layers",     "label": "Number of Layers",  "type": "int",   "default": 1},
        {"name": "bidirectional",  "label": "Bidirectional",     "type": "bool",  "default": False},
        {"name": "dropout",        "label": "Dropout (between layers)", "type": "float", "default": 0.0},
    ],
    "rnn": [
        {"name": "hidden_size",    "label": "Hidden Size",       "type": "int",   "default": None},
        {"name": "num_layers",     "label": "Number of Layers",  "type": "int",   "default": 1},
        {"name": "bidirectional",  "label": "Bidirectional",     "type": "bool",  "default": False},
        {"name": "dropout",        "label": "Dropout (between layers)", "type": "float", "default": 0.0},
    ],

    # --- 1D Convolution ---
    "conv1d": [
        {"name": "out_channels",   "label": "Output Channels",   "type": "int",   "default": None},
        {"name": "kernel_size",    "label": "Kernel Size",       "type": "int",   "default": None},
        {"name": "stride",         "label": "Stride",            "type": "int",   "default": 1},
        {"name": "padding",        "label": "Padding",           "type": "int",   "default": 0},
    ],

    # --- Pooling ---
    "maxpool1d": [
        {"name": "kernel_size",    "label": "Kernel Size",       "type": "int",   "default": None},
        {"name": "stride",         "label": "Stride",            "type": "int",   "default": None},
    ],
    "avgpool1d": [
        {"name": "kernel_size",    "label": "Kernel Size",       "type": "int",   "default": None},
        {"name": "stride",         "label": "Stride",            "type": "int",   "default": None},
    ],
    "adaptive_avgpool1d": [
        {"name": "output_size",    "label": "Output Size",       "type": "int",   "default": 1},
    ],
    "adaptive_maxpool1d": [
        {"name": "output_size",    "label": "Output Size",       "type": "int",   "default": 1},
    ],

    # --- Attention ---
    "multihead_attention": [
        {"name": "num_heads",      "label": "Number of Heads",   "type": "int",   "default": None},
    ],

    # --- Normalization (no user params — auto from prev shape) ---
    "layernorm":   [],
    "batchnorm1d": [],

    # --- Regularization ---
    "dropout": [
        {"name": "p",             "label": "Drop Probability",   "type": "float", "default": 0.5},
    ],

    # --- Reshape Operations ---
    "permute": [
        {"name": "target_format", "label": "Target Format",      "type": "choice", "default": None,
         "options": ["seq_batch_feat", "batch_chan_seq"]},
    ],
    "flatten":     [],
    "squeeze": [
        {"name": "dim",           "label": "Dimension",          "type": "int",   "default": None},
    ],
    "unsqueeze": [
        {"name": "dim",           "label": "Dimension",          "type": "int",   "default": None},
    ],
    "slice_last":  [],

    # --- Combine Operations ---
    "residual_add": [
        {"name": "source_layer_index", "label": "Source Layer Index", "type": "int", "default": None},
    ],
    "cat": [
        {"name": "source_layer_index", "label": "Source Layer Index", "type": "int", "default": None},
        {"name": "concat_dim",         "label": "Concat Dimension",  "type": "int", "default": -1},
    ],
    "select_hidden": [
        {"name": "source_layer_index", "label": "Source Layer Index (LSTM/GRU)", "type": "int", "default": None},
    ],
    "select_cell": [
        {"name": "source_layer_index", "label": "Source Layer Index (LSTM)", "type": "int", "default": None},
    ],
    "bmm": [
        {"name": "source_layer_index", "label": "Source Layer Index (2nd tensor)", "type": "int", "default": None},
    ],
    "stack": [
        {"name": "source_layer_index", "label": "Source Layer Index", "type": "int", "default": None},
        {"name": "dim",                "label": "Stack Dimension",    "type": "int", "default": 0},
    ],
    "split": [
        {"name": "split_size_or_sections", "label": "Split Size", "type": "int", "default": None},
        {"name": "dim",                    "label": "Dimension",  "type": "int", "default": 0},
    ],
    "reshape": [
        {"name": "shape", "label": "Shape (e.g. -1, 128)", "type": "str", "default": None},
    ],

    # --- Activations ---
    "relu":        [],
    "leaky_relu": [
        {"name": "negative_slope", "label": "Negative Slope",    "type": "float", "default": 0.01},
    ],
    "gelu":        [],
    "tanh":        [],
    "sigmoid":     [],
    "softmax": [
        {"name": "dim",           "label": "Dimension",          "type": "int",   "default": -1},
    ],
    "log_softmax": [
        {"name": "dim",           "label": "Dimension",          "type": "int",   "default": -1},
    ],
}

# =====================================================
# Activation layer types — these use ActivationParam
# fact instead of a dedicated param fact.
# =====================================================

ACTIVATION_TYPES = {
    "relu", "leaky_relu", "gelu", "tanh",
    "sigmoid", "softmax", "log_softmax",
}

# =====================================================
# Layers that need no custom param fact at all
# (only a LayerNode is declared)
# =====================================================

NO_PARAM_LAYERS = {
    "pack_sequence", "unpack_sequence",
    "layernorm", "batchnorm1d",
    "flatten", "slice_last",
    "sum_directions", "split_directions",
    "mean_pool", "max_pool_time",
}
