from experta import Fact


# =====================================================
# GLOBAL PIPELINE CONFIGURATION
# =====================================================

class PipelineConfig(Fact):
    """
    Declared once when the user starts the LSTM builder.

    Fields:
        batch_size          : int   (e.g. 32)
        seq_length          : int   (e.g. 50)
        input_type          : str   ("text" | "continuous")
        vocab_size          : int   (only when input_type == "text")
        input_feature_size  : int   (only when input_type == "continuous")
    """
    pass


# =====================================================
# LAYER NODE — one per layer in the pipeline
# =====================================================

class LayerNode(Fact):
    """
    Represents a single layer in the model pipeline.

    Fields:
        index       : int   (0, 1, 2, ...)
        layer_type  : str   ("embedding", "lstm", "gru", "rnn", "linear",
                             "conv1d", "maxpool1d", "avgpool1d",
                             "adaptive_avgpool1d", "adaptive_maxpool1d",
                             "multihead_attention", "layernorm", "batchnorm1d",
                             "dropout", "permute", "flatten", "squeeze",
                             "unsqueeze", "slice_last", "pack_sequence",
                             "unpack_sequence", "residual_add", "cat",
                             "activation")
    """
    pass


# =====================================================
# PER-LAYER CUSTOM PARAMETER FACTS
# Each fact class stores user-provided parameters
# for a specific layer type, tied by layer_index.
# =====================================================

class EmbeddingParam(Fact):
    """
    Fields:
        layer_index     : int
        vocab_size      : int
        embedding_dim   : int
    """
    pass


class LSTMParam(Fact):
    """
    Fields:
        layer_index     : int
        hidden_size     : int
        num_layers      : int   (default 1)
        bidirectional   : bool  (default False)
        dropout         : float (default 0.0, only used when num_layers > 1)
    """
    pass


class GRUParam(Fact):
    """
    Fields:
        layer_index     : int
        hidden_size     : int
        num_layers      : int   (default 1)
        bidirectional   : bool  (default False)
        dropout         : float (default 0.0)
    """
    pass


class RNNParam(Fact):
    """
    Fields:
        layer_index     : int
        hidden_size     : int
        num_layers      : int   (default 1)
        bidirectional   : bool  (default False)
        dropout         : float (default 0.0)
    """
    pass


class LinearParam(Fact):
    """
    Fields:
        layer_index     : int
        out_features    : int
    """
    pass


class Conv1dParam(Fact):
    """
    Fields:
        layer_index     : int
        out_channels    : int
        kernel_size     : int
        stride          : int   (default 1)
        padding         : int   (default 0)
    """
    pass


class Pool1dParam(Fact):
    """
    Fields:
        layer_index     : int
        kernel_size     : int
        stride          : int   (default = kernel_size)
    """
    pass


class AdaptivePool1dParam(Fact):
    """
    Fields:
        layer_index     : int
        output_size     : int   (typically 1)
    """
    pass


class AttentionParam(Fact):
    """
    Fields:
        layer_index     : int
        num_heads       : int
    """
    pass


class DropoutParam(Fact):
    """
    Fields:
        layer_index     : int
        p               : float (default 0.5)
    """
    pass


class ActivationParam(Fact):
    """
    Fields:
        layer_index         : int
        activation_type     : str   ("relu", "leaky_relu", "gelu", "tanh",
                                     "sigmoid", "softmax", "log_softmax")
        dim                 : int   (for softmax/log_softmax, default -1)
        negative_slope      : float (for leaky_relu, default 0.01)
    """
    pass


class PermuteParam(Fact):
    """
    Fields:
        layer_index     : int
        target_format   : str   ("seq_batch_feat" | "batch_chan_seq")
    """
    pass


class SqueezeParam(Fact):
    """
    Fields:
        layer_index     : int
        dim             : int
    """
    pass


class UnsqueezeParam(Fact):
    """
    Fields:
        layer_index     : int
        dim             : int
    """
    pass


class CatParam(Fact):
    """
    Fields:
        layer_index         : int
        source_layer_index  : int
        concat_dim          : int
    """
    pass


class ResidualParam(Fact):
    """
    Fields:
        layer_index         : int
        source_layer_index  : int
    """
    pass

class SelectHiddenParam(Fact):
    """
    Fields:
        layer_index         : int
        source_layer_index  : int
    """
    pass

class SelectCellParam(Fact):
    """
    Fields:
        layer_index         : int
        source_layer_index  : int
    """
    pass

class BmmParam(Fact):
    """
    Fields:
        layer_index         : int
        source_layer_index  : int
    """
    pass

class StackParam(Fact):
    """
    Fields:
        layer_index         : int
        source_layer_index  : int
        dim                 : int
    """
    pass

class SplitParam(Fact):
    """
    Fields:
        layer_index             : int
        split_size_or_sections  : int
        dim                     : int
    """
    pass

class ReshapeParam(Fact):
    """
    Fields:
        layer_index         : int
        shape               : str
    """
    pass


# =====================================================
# COMPUTED SHAPE FACTS
# Declared by shape calculation rules.
# =====================================================

class LayerOutputShape(Fact):
    """
    The computed output tensor shape after a layer executes.

    Fields:
        layer_index : int
        dims        : int   (2 or 3)
        d0          : int   (first dimension size)
        d1          : int   (second dimension size)
        d2          : int   (third dimension size, or -1 if dims==2)
        format      : str   ("seq_batch_feat" | "batch_feat" |
                             "batch_chan_seq" | "packed")
    """
    pass


class LayerExpectedInput(Fact):
    """
    The expected input shape for a layer (copied from previous
    layer's LayerOutputShape, or from PipelineConfig for index 0).

    Fields:
        layer_index : int
        dims        : int
        d0          : int
        d1          : int
        d2          : int
        format      : str
    """
    pass


# =====================================================
# DIAGNOSTIC FACTS — fired by pruning rules
# =====================================================

class Mismatch(Fact):
    """
    Represents a detected incompatibility between consecutive layers.

    Fields:
        layer_index     : int   (the layer that caused the issue)
        prev_index      : int   (the preceding layer)
        mismatch_type   : str   ("dim_3d_to_2d", "dim_2d_to_3d", "format",
                                 "feature_size", "packed_protocol",
                                 "embedding_position", "redundancy")
        description     : str   (human-readable explanation)
        severity        : str   ("error" | "warning")
    """
    pass


class FixRecommendation(Fact):
    """
    A suggested fix for a detected mismatch.

    Fields:
        layer_index         : int   (which layer the fix is for)
        fix_description     : str   (what to do)
        insert_layer_type   : str   (suggested layer to insert, or "adjust"
                                     if the fix is a parameter change)
    """
    pass


# =====================================================
# CODE GENERATION FACTS
# =====================================================

class LayerCode(Fact):
    """
    The generated PyTorch code snippet for a single layer.

    Fields:
        layer_index     : int
        init_code       : str   (code for __init__, e.g. 'self.lstm = nn.LSTM(...)')
        forward_code    : str   (code for forward(), e.g. 'x, (h, c) = self.lstm(x)')
    """
    pass
class AutoUnpackCode(Fact):
    """
    Stores injected code when a layer auto-unpacks a PackedSequence.
    Fields:
        layer_index     : int
        code            : str
    """
    pass
