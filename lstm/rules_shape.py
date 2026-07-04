from experta import *

from lstm.facts import (
    PipelineConfig, LayerNode, LayerOutputShape,
    EmbeddingParam, LSTMParam, GRUParam, RNNParam, LinearParam,
    Conv1dParam, Pool1dParam, AdaptivePool1dParam, AttentionParam,
    DropoutParam, ActivationParam, PermuteParam, SqueezeParam,
    UnsqueezeParam, CatParam, ResidualParam,
)


class ShapeRules(KnowledgeEngine):
    """
    Shape Calculation Engine (salience=10).

    Each rule matches a LayerNode + its parameter fact + the previous
    layer's LayerOutputShape, then declares the new LayerOutputShape.

    For the first layer (index=0), shape is derived from PipelineConfig.
    """

    # =====================================================
    # SC_EMBED: nn.Embedding
    # Input:  [seq_length, batch_size]  (integers)
    # Output: [seq_length, batch_size, embedding_dim]
    # =====================================================
    @Rule(
        LayerNode(index=MATCH.i, layer_type="embedding"),
        EmbeddingParam(layer_index=MATCH.i, embedding_dim=MATCH.edim),
        PipelineConfig(batch_size=MATCH.b, seq_length=MATCH.s),
        NOT(LayerOutputShape(layer_index=MATCH.i)),
        salience=10
    )
    def sc_embedding(self, i, edim, b, s):
        self.declare(LayerOutputShape(
            layer_index=i,
            dims=3,
            d0=s,
            d1=b,
            d2=edim,
            format="seq_batch_feat"
        ))

    # =====================================================
    # SC_LINEAR_INITIAL: nn.Linear as first layer for
    # continuous input (index=0, no previous shape)
    # Input:  [seq, batch, input_feature_size]
    # Output: [seq, batch, out_features]
    # =====================================================
    @Rule(
        LayerNode(index=0, layer_type="linear"),
        LinearParam(layer_index=0, out_features=MATCH.out_f),
        PipelineConfig(batch_size=MATCH.b, seq_length=MATCH.s,
                        input_type="continuous", input_feature_size=MATCH.in_f),
        NOT(LayerOutputShape(layer_index=0)),
        salience=10
    )
    def sc_linear_initial(self, out_f, b, s, in_f):
        self.declare(LayerOutputShape(
            layer_index=0,
            dims=3,
            d0=s,
            d1=b,
            d2=out_f,
            format="seq_batch_feat"
        ))

    # =====================================================
    # SC_LSTM_INITIAL: LSTM as first layer for continuous input
    # Input:  [seq, batch, input_feature_size]
    # Output: [seq, batch, hidden * dirs]
    # =====================================================
    @Rule(
        LayerNode(index=0, layer_type="lstm"),
        LSTMParam(layer_index=0, hidden_size=MATCH.h,
                   bidirectional=MATCH.bidir),
        PipelineConfig(batch_size=MATCH.b, seq_length=MATCH.s,
                        input_type="continuous"),
        NOT(LayerOutputShape(layer_index=0)),
        salience=10
    )
    def sc_lstm_initial(self, h, bidir, b, s):
        num_dirs = 2 if bidir else 1
        self.declare(LayerOutputShape(
            layer_index=0,
            dims=3,
            d0=s,
            d1=b,
            d2=h * num_dirs,
            format="seq_batch_feat"
        ))

    # =====================================================
    # SC_GRU_INITIAL: GRU as first layer for continuous input
    # =====================================================
    @Rule(
        LayerNode(index=0, layer_type="gru"),
        GRUParam(layer_index=0, hidden_size=MATCH.h,
                  bidirectional=MATCH.bidir),
        PipelineConfig(batch_size=MATCH.b, seq_length=MATCH.s,
                        input_type="continuous"),
        NOT(LayerOutputShape(layer_index=0)),
        salience=10
    )
    def sc_gru_initial(self, h, bidir, b, s):
        num_dirs = 2 if bidir else 1
        self.declare(LayerOutputShape(
            layer_index=0,
            dims=3,
            d0=s,
            d1=b,
            d2=h * num_dirs,
            format="seq_batch_feat"
        ))

    # =====================================================
    # SC_RNN_INITIAL: RNN as first layer for continuous input
    # =====================================================
    @Rule(
        LayerNode(index=0, layer_type="rnn"),
        RNNParam(layer_index=0, hidden_size=MATCH.h,
                  bidirectional=MATCH.bidir),
        PipelineConfig(batch_size=MATCH.b, seq_length=MATCH.s,
                        input_type="continuous"),
        NOT(LayerOutputShape(layer_index=0)),
        salience=10
    )
    def sc_rnn_initial(self, h, bidir, b, s):
        num_dirs = 2 if bidir else 1
        self.declare(LayerOutputShape(
            layer_index=0,
            dims=3,
            d0=s,
            d1=b,
            d2=h * num_dirs,
            format="seq_batch_feat"
        ))

    # =====================================================
    # SC_LINEAR_3D: nn.Linear applied to 3D tensor
    # (per-timestep projection)
    # Input:  [d0, d1, in_features]
    # Output: [d0, d1, out_features]
    # =====================================================
    @Rule(
        LayerNode(index=MATCH.i, layer_type="linear"),
        LinearParam(layer_index=MATCH.i, out_features=MATCH.out_f),
        LayerOutputShape(layer_index=MATCH.prev_i, dims=3,
                          d0=MATCH.d0, d1=MATCH.d1, d2=MATCH.d2,
                          format=MATCH.fmt),
        TEST(lambda i, prev_i: i == prev_i + 1),
        NOT(LayerOutputShape(layer_index=MATCH.i)),
        salience=10
    )
    def sc_linear_3d(self, i, out_f, d0, d1, fmt, **kwargs):
        self.declare(LayerOutputShape(
            layer_index=i,
            dims=3,
            d0=d0,
            d1=d1,
            d2=out_f,
            format=fmt
        ))

    # =====================================================
    # SC_LINEAR_2D: nn.Linear applied to 2D tensor
    # Input:  [batch, in_features]
    # Output: [batch, out_features]
    # =====================================================
    @Rule(
        LayerNode(index=MATCH.i, layer_type="linear"),
        LinearParam(layer_index=MATCH.i, out_features=MATCH.out_f),
        LayerOutputShape(layer_index=MATCH.prev_i, dims=2,
                          d0=MATCH.d0, d1=MATCH.d1,
                          format=MATCH.fmt),
        TEST(lambda i, prev_i: i == prev_i + 1),
        NOT(LayerOutputShape(layer_index=MATCH.i)),
        salience=10
    )
    def sc_linear_2d(self, i, out_f, d0, fmt, **kwargs):
        self.declare(LayerOutputShape(
            layer_index=i,
            dims=2,
            d0=d0,
            d1=out_f,
            d2=-1,
            format="batch_feat"
        ))

    # =====================================================
    # SC_LSTM: nn.LSTM
    # Input:  [seq, batch, features]
    # Output: [seq, batch, hidden_size * num_directions]
    # =====================================================
    @Rule(
        LayerNode(index=MATCH.i, layer_type="lstm"),
        LSTMParam(layer_index=MATCH.i, hidden_size=MATCH.h,
                   bidirectional=MATCH.bidir),
        LayerOutputShape(layer_index=MATCH.prev_i, dims=3,
                          d0=MATCH.d0, d1=MATCH.d1,
                          format=MATCH.fmt),
        TEST(lambda i, prev_i: i == prev_i + 1),
        NOT(LayerOutputShape(layer_index=MATCH.i)),
        salience=10
    )
    def sc_lstm(self, i, h, bidir, d0, d1, **kwargs):
        num_dirs = 2 if bidir else 1
        self.declare(LayerOutputShape(
            layer_index=i,
            dims=3,
            d0=d0,
            d1=d1,
            d2=h * num_dirs,
            format="seq_batch_feat"
        ))

    # =====================================================
    # SC_GRU: nn.GRU
    # Input:  [seq, batch, features]
    # Output: [seq, batch, hidden_size * num_directions]
    # =====================================================
    @Rule(
        LayerNode(index=MATCH.i, layer_type="gru"),
        GRUParam(layer_index=MATCH.i, hidden_size=MATCH.h,
                  bidirectional=MATCH.bidir),
        LayerOutputShape(layer_index=MATCH.prev_i, dims=3,
                          d0=MATCH.d0, d1=MATCH.d1,
                          format=MATCH.fmt),
        TEST(lambda i, prev_i: i == prev_i + 1),
        NOT(LayerOutputShape(layer_index=MATCH.i)),
        salience=10
    )
    def sc_gru(self, i, h, bidir, d0, d1, **kwargs):
        num_dirs = 2 if bidir else 1
        self.declare(LayerOutputShape(
            layer_index=i,
            dims=3,
            d0=d0,
            d1=d1,
            d2=h * num_dirs,
            format="seq_batch_feat"
        ))

    # =====================================================
    # SC_RNN: nn.RNN
    # Input:  [seq, batch, features]
    # Output: [seq, batch, hidden_size * num_directions]
    # =====================================================
    @Rule(
        LayerNode(index=MATCH.i, layer_type="rnn"),
        RNNParam(layer_index=MATCH.i, hidden_size=MATCH.h,
                  bidirectional=MATCH.bidir),
        LayerOutputShape(layer_index=MATCH.prev_i, dims=3,
                          d0=MATCH.d0, d1=MATCH.d1,
                          format=MATCH.fmt),
        TEST(lambda i, prev_i: i == prev_i + 1),
        NOT(LayerOutputShape(layer_index=MATCH.i)),
        salience=10
    )
    def sc_rnn(self, i, h, bidir, d0, d1, **kwargs):
        num_dirs = 2 if bidir else 1
        self.declare(LayerOutputShape(
            layer_index=i,
            dims=3,
            d0=d0,
            d1=d1,
            d2=h * num_dirs,
            format="seq_batch_feat"
        ))

    # =====================================================
    # SC_CONV1D: nn.Conv1d
    # Input:  [batch, channels, seq]  (batch_chan_seq format)
    # Output: [batch, out_channels, new_seq]
    # new_seq = floor((seq + 2*padding - kernel) / stride) + 1
    # =====================================================
    @Rule(
        LayerNode(index=MATCH.i, layer_type="conv1d"),
        Conv1dParam(layer_index=MATCH.i, out_channels=MATCH.out_ch,
                     kernel_size=MATCH.k, stride=MATCH.st, padding=MATCH.pad),
        LayerOutputShape(layer_index=MATCH.prev_i, dims=3,
                          d0=MATCH.d0, d1=MATCH.d1, d2=MATCH.d2,
                          format="batch_chan_seq"),
        TEST(lambda i, prev_i: i == prev_i + 1),
        NOT(LayerOutputShape(layer_index=MATCH.i)),
        salience=10
    )
    def sc_conv1d(self, i, out_ch, k, st, pad, d0, d2, **kwargs):
        new_seq = ((d2 + 2 * pad - k) // st) + 1
        self.declare(LayerOutputShape(
            layer_index=i,
            dims=3,
            d0=d0,
            d1=out_ch,
            d2=new_seq,
            format="batch_chan_seq"
        ))

    # =====================================================
    # SC_MAXPOOL1D: nn.MaxPool1d
    # Input:  [batch, channels, seq]
    # Output: [batch, channels, floor((seq - kernel) / stride) + 1]
    # =====================================================
    @Rule(
        LayerNode(index=MATCH.i, layer_type="maxpool1d"),
        Pool1dParam(layer_index=MATCH.i, kernel_size=MATCH.k, stride=MATCH.st),
        LayerOutputShape(layer_index=MATCH.prev_i, dims=3,
                          d0=MATCH.d0, d1=MATCH.d1, d2=MATCH.d2,
                          format="batch_chan_seq"),
        TEST(lambda i, prev_i: i == prev_i + 1),
        NOT(LayerOutputShape(layer_index=MATCH.i)),
        salience=10
    )
    def sc_maxpool1d(self, i, k, st, d0, d1, d2, **kwargs):
        new_seq = ((d2 - k) // st) + 1
        self.declare(LayerOutputShape(
            layer_index=i,
            dims=3,
            d0=d0,
            d1=d1,
            d2=new_seq,
            format="batch_chan_seq"
        ))

    # =====================================================
    # SC_AVGPOOL1D: nn.AvgPool1d
    # Same shape logic as MaxPool1d
    # =====================================================
    @Rule(
        LayerNode(index=MATCH.i, layer_type="avgpool1d"),
        Pool1dParam(layer_index=MATCH.i, kernel_size=MATCH.k, stride=MATCH.st),
        LayerOutputShape(layer_index=MATCH.prev_i, dims=3,
                          d0=MATCH.d0, d1=MATCH.d1, d2=MATCH.d2,
                          format="batch_chan_seq"),
        TEST(lambda i, prev_i: i == prev_i + 1),
        NOT(LayerOutputShape(layer_index=MATCH.i)),
        salience=10
    )
    def sc_avgpool1d(self, i, k, st, d0, d1, d2, **kwargs):
        new_seq = ((d2 - k) // st) + 1
        self.declare(LayerOutputShape(
            layer_index=i,
            dims=3,
            d0=d0,
            d1=d1,
            d2=new_seq,
            format="batch_chan_seq"
        ))

    # =====================================================
    # SC_ADAPTIVE_AVGPOOL1D: nn.AdaptiveAvgPool1d
    # Input:  [batch, channels, seq]
    # Output: [batch, channels, output_size]
    # =====================================================
    @Rule(
        LayerNode(index=MATCH.i, layer_type="adaptive_avgpool1d"),
        AdaptivePool1dParam(layer_index=MATCH.i, output_size=MATCH.os),
        LayerOutputShape(layer_index=MATCH.prev_i, dims=3,
                          d0=MATCH.d0, d1=MATCH.d1,
                          format="batch_chan_seq"),
        TEST(lambda i, prev_i: i == prev_i + 1),
        NOT(LayerOutputShape(layer_index=MATCH.i)),
        salience=10
    )
    def sc_adaptive_avgpool1d(self, i, os, d0, d1, **kwargs):
        self.declare(LayerOutputShape(
            layer_index=i,
            dims=3,
            d0=d0,
            d1=d1,
            d2=os,
            format="batch_chan_seq"
        ))

    # =====================================================
    # SC_ADAPTIVE_MAXPOOL1D: nn.AdaptiveMaxPool1d
    # Same shape logic as AdaptiveAvgPool1d
    # =====================================================
    @Rule(
        LayerNode(index=MATCH.i, layer_type="adaptive_maxpool1d"),
        AdaptivePool1dParam(layer_index=MATCH.i, output_size=MATCH.os),
        LayerOutputShape(layer_index=MATCH.prev_i, dims=3,
                          d0=MATCH.d0, d1=MATCH.d1,
                          format="batch_chan_seq"),
        TEST(lambda i, prev_i: i == prev_i + 1),
        NOT(LayerOutputShape(layer_index=MATCH.i)),
        salience=10
    )
    def sc_adaptive_maxpool1d(self, i, os, d0, d1, **kwargs):
        self.declare(LayerOutputShape(
            layer_index=i,
            dims=3,
            d0=d0,
            d1=d1,
            d2=os,
            format="batch_chan_seq"
        ))

    # =====================================================
    # SC_ATTENTION: nn.MultiheadAttention
    # Input:  [seq, batch, embed_dim]  (seq_batch_feat)
    # Output: [seq, batch, embed_dim]  (shape-preserving)
    # =====================================================
    @Rule(
        LayerNode(index=MATCH.i, layer_type="multihead_attention"),
        AttentionParam(layer_index=MATCH.i),
        LayerOutputShape(layer_index=MATCH.prev_i, dims=3,
                          d0=MATCH.d0, d1=MATCH.d1, d2=MATCH.d2,
                          format="seq_batch_feat"),
        TEST(lambda i, prev_i: i == prev_i + 1),
        NOT(LayerOutputShape(layer_index=MATCH.i)),
        salience=10
    )
    def sc_attention(self, i, d0, d1, d2, **kwargs):
        self.declare(LayerOutputShape(
            layer_index=i,
            dims=3,
            d0=d0,
            d1=d1,
            d2=d2,
            format="seq_batch_feat"
        ))

    # =====================================================
    # SC_LAYERNORM: nn.LayerNorm (shape-preserving)
    # Works on both 2D and 3D — we need two rules
    # =====================================================
    @Rule(
        LayerNode(index=MATCH.i, layer_type="layernorm"),
        LayerOutputShape(layer_index=MATCH.prev_i, dims=3,
                          d0=MATCH.d0, d1=MATCH.d1, d2=MATCH.d2,
                          format=MATCH.fmt),
        TEST(lambda i, prev_i: i == prev_i + 1),
        NOT(LayerOutputShape(layer_index=MATCH.i)),
        salience=10
    )
    def sc_layernorm_3d(self, i, d0, d1, d2, fmt, **kwargs):
        self.declare(LayerOutputShape(
            layer_index=i, dims=3, d0=d0, d1=d1, d2=d2, format=fmt
        ))

    @Rule(
        LayerNode(index=MATCH.i, layer_type="layernorm"),
        LayerOutputShape(layer_index=MATCH.prev_i, dims=2,
                          d0=MATCH.d0, d1=MATCH.d1,
                          format=MATCH.fmt),
        TEST(lambda i, prev_i: i == prev_i + 1),
        NOT(LayerOutputShape(layer_index=MATCH.i)),
        salience=10
    )
    def sc_layernorm_2d(self, i, d0, d1, fmt, **kwargs):
        self.declare(LayerOutputShape(
            layer_index=i, dims=2, d0=d0, d1=d1, d2=-1, format=fmt
        ))

    # =====================================================
    # SC_BATCHNORM1D: nn.BatchNorm1d (shape-preserving, 2D)
    # Input:  [batch, features]
    # Output: [batch, features]
    # =====================================================
    @Rule(
        LayerNode(index=MATCH.i, layer_type="batchnorm1d"),
        LayerOutputShape(layer_index=MATCH.prev_i, dims=2,
                          d0=MATCH.d0, d1=MATCH.d1,
                          format=MATCH.fmt),
        TEST(lambda i, prev_i: i == prev_i + 1),
        NOT(LayerOutputShape(layer_index=MATCH.i)),
        salience=10
    )
    def sc_batchnorm1d(self, i, d0, d1, fmt, **kwargs):
        self.declare(LayerOutputShape(
            layer_index=i, dims=2, d0=d0, d1=d1, d2=-1, format=fmt
        ))

    # =====================================================
    # SC_DROPOUT: nn.Dropout (shape-preserving, any dims)
    # =====================================================
    @Rule(
        LayerNode(index=MATCH.i, layer_type="dropout"),
        LayerOutputShape(layer_index=MATCH.prev_i, dims=3,
                          d0=MATCH.d0, d1=MATCH.d1, d2=MATCH.d2,
                          format=MATCH.fmt),
        TEST(lambda i, prev_i: i == prev_i + 1),
        NOT(LayerOutputShape(layer_index=MATCH.i)),
        salience=10
    )
    def sc_dropout_3d(self, i, d0, d1, d2, fmt, **kwargs):
        self.declare(LayerOutputShape(
            layer_index=i, dims=3, d0=d0, d1=d1, d2=d2, format=fmt
        ))

    @Rule(
        LayerNode(index=MATCH.i, layer_type="dropout"),
        LayerOutputShape(layer_index=MATCH.prev_i, dims=2,
                          d0=MATCH.d0, d1=MATCH.d1,
                          format=MATCH.fmt),
        TEST(lambda i, prev_i: i == prev_i + 1),
        NOT(LayerOutputShape(layer_index=MATCH.i)),
        salience=10
    )
    def sc_dropout_2d(self, i, d0, d1, fmt, **kwargs):
        self.declare(LayerOutputShape(
            layer_index=i, dims=2, d0=d0, d1=d1, d2=-1, format=fmt
        ))

    # =====================================================
    # SC_PERMUTE_SBF_BCS: Permute seq_batch_feat -> batch_chan_seq
    # [seq, batch, feat] -> [batch, feat, seq]
    # =====================================================
    @Rule(
        LayerNode(index=MATCH.i, layer_type="permute"),
        PermuteParam(layer_index=MATCH.i, target_format="batch_chan_seq"),
        LayerOutputShape(layer_index=MATCH.prev_i, dims=3,
                          d0=MATCH.d0, d1=MATCH.d1, d2=MATCH.d2,
                          format="seq_batch_feat"),
        TEST(lambda i, prev_i: i == prev_i + 1),
        NOT(LayerOutputShape(layer_index=MATCH.i)),
        salience=10
    )
    def sc_permute_sbf_to_bcs(self, i, d0, d1, d2, **kwargs):
        # [seq, batch, feat] -> [batch, feat, seq]
        self.declare(LayerOutputShape(
            layer_index=i,
            dims=3,
            d0=d1,       # batch
            d1=d2,       # feat (now channels)
            d2=d0,       # seq
            format="batch_chan_seq"
        ))

    # =====================================================
    # SC_PERMUTE_BCS_SBF: Permute batch_chan_seq -> seq_batch_feat
    # [batch, channels, seq] -> [seq, batch, channels]
    # =====================================================
    @Rule(
        LayerNode(index=MATCH.i, layer_type="permute"),
        PermuteParam(layer_index=MATCH.i, target_format="seq_batch_feat"),
        LayerOutputShape(layer_index=MATCH.prev_i, dims=3,
                          d0=MATCH.d0, d1=MATCH.d1, d2=MATCH.d2,
                          format="batch_chan_seq"),
        TEST(lambda i, prev_i: i == prev_i + 1),
        NOT(LayerOutputShape(layer_index=MATCH.i)),
        salience=10
    )
    def sc_permute_bcs_to_sbf(self, i, d0, d1, d2, **kwargs):
        # [batch, channels, seq] -> [seq, batch, channels]
        self.declare(LayerOutputShape(
            layer_index=i,
            dims=3,
            d0=d2,       # seq
            d1=d0,       # batch
            d2=d1,       # channels (now features)
            format="seq_batch_feat"
        ))

    # =====================================================
    # SC_FLATTEN: nn.Flatten (3D -> 2D)
    # Input:  [d0, d1, d2]
    # Output: [d0, d1 * d2]  (for seq_batch_feat: [seq*batch?, ...])
    #
    # For seq_batch_feat: flatten removes seq, result is [batch, seq*feat]
    # Actually Flatten(start_dim=1) on [batch, d1, d2] -> [batch, d1*d2]
    # But for [seq, batch, feat] we first need context...
    # Typically flatten is used AFTER permute to batch_chan_seq,
    # so [batch, chan, seq] -> [batch, chan*seq]
    # =====================================================
    @Rule(
        LayerNode(index=MATCH.i, layer_type="flatten"),
        LayerOutputShape(layer_index=MATCH.prev_i, dims=3,
                          d0=MATCH.d0, d1=MATCH.d1, d2=MATCH.d2,
                          format="batch_chan_seq"),
        TEST(lambda i, prev_i: i == prev_i + 1),
        NOT(LayerOutputShape(layer_index=MATCH.i)),
        salience=10
    )
    def sc_flatten_bcs(self, i, d0, d1, d2, **kwargs):
        # [batch, chan, seq] -> [batch, chan * seq]
        self.declare(LayerOutputShape(
            layer_index=i,
            dims=2,
            d0=d0,
            d1=d1 * d2,
            d2=-1,
            format="batch_feat"
        ))

    @Rule(
        LayerNode(index=MATCH.i, layer_type="flatten"),
        LayerOutputShape(layer_index=MATCH.prev_i, dims=3,
                          d0=MATCH.d0, d1=MATCH.d1, d2=MATCH.d2,
                          format="seq_batch_feat"),
        TEST(lambda i, prev_i: i == prev_i + 1),
        NOT(LayerOutputShape(layer_index=MATCH.i)),
        salience=10
    )
    def sc_flatten_sbf(self, i, d0, d1, d2, **kwargs):
        # [seq, batch, feat] — flatten after batch dim
        # Result: [batch, seq * feat] (treating batch=d1 as the batch dim)
        # Note: this is unusual, user likely needs permute first
        self.declare(LayerOutputShape(
            layer_index=i,
            dims=2,
            d0=d1,
            d1=d0 * d2,
            d2=-1,
            format="batch_feat"
        ))

    # =====================================================
    # SC_SLICE_LAST: Tensor Slicing x[-1, :, :]
    # Input:  [seq, batch, features]  (seq_batch_feat)
    # Output: [batch, features]       (batch_feat)
    # =====================================================
    @Rule(
        LayerNode(index=MATCH.i, layer_type="slice_last"),
        LayerOutputShape(layer_index=MATCH.prev_i, dims=3,
                          d0=MATCH.d0, d1=MATCH.d1, d2=MATCH.d2,
                          format="seq_batch_feat"),
        TEST(lambda i, prev_i: i == prev_i + 1),
        NOT(LayerOutputShape(layer_index=MATCH.i)),
        salience=10
    )
    def sc_slice_last(self, i, d1, d2, **kwargs):
        self.declare(LayerOutputShape(
            layer_index=i,
            dims=2,
            d0=d1,      # batch
            d1=d2,      # features
            d2=-1,
            format="batch_feat"
        ))

    # =====================================================
    # SC_SQUEEZE: Squeeze a dimension of size 1
    # Removes one dimension -> reduces dims count
    # =====================================================
    @Rule(
        LayerNode(index=MATCH.i, layer_type="squeeze"),
        SqueezeParam(layer_index=MATCH.i, dim=MATCH.dim),
        LayerOutputShape(layer_index=MATCH.prev_i, dims=3,
                          d0=MATCH.d0, d1=MATCH.d1, d2=MATCH.d2,
                          format=MATCH.fmt),
        TEST(lambda i, prev_i: i == prev_i + 1),
        NOT(LayerOutputShape(layer_index=MATCH.i)),
        salience=10
    )
    def sc_squeeze_3d(self, i, dim, d0, d1, d2, fmt, **kwargs):
        # Squeeze dim 0, 1, or 2 from 3D -> 2D
        if dim == 0 or dim == -3:
            self.declare(LayerOutputShape(
                layer_index=i, dims=2, d0=d1, d1=d2, d2=-1,
                format="batch_feat"))
        elif dim == 1 or dim == -2:
            self.declare(LayerOutputShape(
                layer_index=i, dims=2, d0=d0, d1=d2, d2=-1,
                format="batch_feat"))
        elif dim == 2 or dim == -1:
            self.declare(LayerOutputShape(
                layer_index=i, dims=2, d0=d0, d1=d1, d2=-1,
                format="batch_feat"))

    # =====================================================
    # SC_UNSQUEEZE: Unsqueeze adds a dimension of size 1
    # 2D -> 3D
    # =====================================================
    @Rule(
        LayerNode(index=MATCH.i, layer_type="unsqueeze"),
        UnsqueezeParam(layer_index=MATCH.i, dim=MATCH.dim),
        LayerOutputShape(layer_index=MATCH.prev_i, dims=2,
                          d0=MATCH.d0, d1=MATCH.d1,
                          format=MATCH.fmt),
        TEST(lambda i, prev_i: i == prev_i + 1),
        NOT(LayerOutputShape(layer_index=MATCH.i)),
        salience=10
    )
    def sc_unsqueeze_2d(self, i, dim, d0, d1, **kwargs):
        if dim == 0 or dim == -3:
            self.declare(LayerOutputShape(
                layer_index=i, dims=3, d0=1, d1=d0, d2=d1,
                format="seq_batch_feat"))
        elif dim == 1 or dim == -2:
            self.declare(LayerOutputShape(
                layer_index=i, dims=3, d0=d0, d1=1, d2=d1,
                format="seq_batch_feat"))
        elif dim == 2 or dim == -1:
            self.declare(LayerOutputShape(
                layer_index=i, dims=3, d0=d0, d1=d1, d2=1,
                format="seq_batch_feat"))

    # =====================================================
    # SC_ACTIVATION: All activations are shape-preserving
    # Works for both 2D and 3D
    # =====================================================
    @Rule(
        LayerNode(index=MATCH.i, layer_type="activation"),
        LayerOutputShape(layer_index=MATCH.prev_i, dims=3,
                          d0=MATCH.d0, d1=MATCH.d1, d2=MATCH.d2,
                          format=MATCH.fmt),
        TEST(lambda i, prev_i: i == prev_i + 1),
        NOT(LayerOutputShape(layer_index=MATCH.i)),
        salience=10
    )
    def sc_activation_3d(self, i, d0, d1, d2, fmt, **kwargs):
        self.declare(LayerOutputShape(
            layer_index=i, dims=3, d0=d0, d1=d1, d2=d2, format=fmt
        ))

    @Rule(
        LayerNode(index=MATCH.i, layer_type="activation"),
        LayerOutputShape(layer_index=MATCH.prev_i, dims=2,
                          d0=MATCH.d0, d1=MATCH.d1,
                          format=MATCH.fmt),
        TEST(lambda i, prev_i: i == prev_i + 1),
        NOT(LayerOutputShape(layer_index=MATCH.i)),
        salience=10
    )
    def sc_activation_2d(self, i, d0, d1, fmt, **kwargs):
        self.declare(LayerOutputShape(
            layer_index=i, dims=2, d0=d0, d1=d1, d2=-1, format=fmt
        ))

    # =====================================================
    # SC_PACK: pack_padded_sequence
    # Input:  [seq, batch, features]
    # Output: PackedSequence (we mark format as "packed"
    #         and preserve dims for the recurrent layer)
    # =====================================================
    @Rule(
        LayerNode(index=MATCH.i, layer_type="pack_sequence"),
        LayerOutputShape(layer_index=MATCH.prev_i, dims=3,
                          d0=MATCH.d0, d1=MATCH.d1, d2=MATCH.d2,
                          format="seq_batch_feat"),
        TEST(lambda i, prev_i: i == prev_i + 1),
        NOT(LayerOutputShape(layer_index=MATCH.i)),
        salience=10
    )
    def sc_pack(self, i, d0, d1, d2, **kwargs):
        self.declare(LayerOutputShape(
            layer_index=i,
            dims=3,
            d0=d0,
            d1=d1,
            d2=d2,
            format="packed"
        ))

    # =====================================================
    # SC_UNPACK: pad_packed_sequence
    # Input:  PackedSequence (format="packed")
    # Output: [seq, batch, features] (restores seq_batch_feat)
    # =====================================================
    @Rule(
        LayerNode(index=MATCH.i, layer_type="unpack_sequence"),
        LayerOutputShape(layer_index=MATCH.prev_i, dims=3,
                          d0=MATCH.d0, d1=MATCH.d1, d2=MATCH.d2,
                          format="packed"),
        TEST(lambda i, prev_i: i == prev_i + 1),
        NOT(LayerOutputShape(layer_index=MATCH.i)),
        salience=10
    )
    def sc_unpack(self, i, d0, d1, d2, **kwargs):
        self.declare(LayerOutputShape(
            layer_index=i,
            dims=3,
            d0=d0,
            d1=d1,
            d2=d2,
            format="seq_batch_feat"
        ))

    # =====================================================
    # SC_LSTM_PACKED: nn.LSTM with packed input
    # Same output as regular LSTM but accepts packed format
    # =====================================================
    @Rule(
        LayerNode(index=MATCH.i, layer_type="lstm"),
        LSTMParam(layer_index=MATCH.i, hidden_size=MATCH.h,
                   bidirectional=MATCH.bidir),
        LayerOutputShape(layer_index=MATCH.prev_i, dims=3,
                          d0=MATCH.d0, d1=MATCH.d1,
                          format="packed"),
        TEST(lambda i, prev_i: i == prev_i + 1),
        NOT(LayerOutputShape(layer_index=MATCH.i)),
        salience=10
    )
    def sc_lstm_packed(self, i, h, bidir, d0, d1, **kwargs):
        num_dirs = 2 if bidir else 1
        self.declare(LayerOutputShape(
            layer_index=i,
            dims=3,
            d0=d0,
            d1=d1,
            d2=h * num_dirs,
            format="packed"
        ))

    # =====================================================
    # SC_GRU_PACKED / SC_RNN_PACKED: Same pattern
    # =====================================================
    @Rule(
        LayerNode(index=MATCH.i, layer_type="gru"),
        GRUParam(layer_index=MATCH.i, hidden_size=MATCH.h,
                  bidirectional=MATCH.bidir),
        LayerOutputShape(layer_index=MATCH.prev_i, dims=3,
                          d0=MATCH.d0, d1=MATCH.d1,
                          format="packed"),
        TEST(lambda i, prev_i: i == prev_i + 1),
        NOT(LayerOutputShape(layer_index=MATCH.i)),
        salience=10
    )
    def sc_gru_packed(self, i, h, bidir, d0, d1, **kwargs):
        num_dirs = 2 if bidir else 1
        self.declare(LayerOutputShape(
            layer_index=i, dims=3, d0=d0, d1=d1,
            d2=h * num_dirs, format="packed"
        ))

    @Rule(
        LayerNode(index=MATCH.i, layer_type="rnn"),
        RNNParam(layer_index=MATCH.i, hidden_size=MATCH.h,
                  bidirectional=MATCH.bidir),
        LayerOutputShape(layer_index=MATCH.prev_i, dims=3,
                          d0=MATCH.d0, d1=MATCH.d1,
                          format="packed"),
        TEST(lambda i, prev_i: i == prev_i + 1),
        NOT(LayerOutputShape(layer_index=MATCH.i)),
        salience=10
    )
    def sc_rnn_packed(self, i, h, bidir, d0, d1, **kwargs):
        num_dirs = 2 if bidir else 1
        self.declare(LayerOutputShape(
            layer_index=i, dims=3, d0=d0, d1=d1,
            d2=h * num_dirs, format="packed"
        ))

    # =====================================================
    # SC_RESIDUAL_ADD: Residual / Skip Connection
    # Output shape = same as current input (validates
    # that source shape matches)
    # =====================================================
    @Rule(
        LayerNode(index=MATCH.i, layer_type="residual_add"),
        ResidualParam(layer_index=MATCH.i, source_layer_index=MATCH.src),
        LayerOutputShape(layer_index=MATCH.prev_i,
                          dims=MATCH.d, d0=MATCH.d0, d1=MATCH.d1,
                          d2=MATCH.d2, format=MATCH.fmt),
        TEST(lambda i, prev_i: i == prev_i + 1),
        NOT(LayerOutputShape(layer_index=MATCH.i)),
        salience=10
    )
    def sc_residual_add(self, i, d, d0, d1, d2, fmt, **kwargs):
        self.declare(LayerOutputShape(
            layer_index=i, dims=d, d0=d0, d1=d1, d2=d2, format=fmt
        ))

    # =====================================================
    # SC_CAT: torch.cat — concatenation along a dimension
    # For simplicity, concat along the feature dim (last)
    # =====================================================
    @Rule(
        LayerNode(index=MATCH.i, layer_type="cat"),
        CatParam(layer_index=MATCH.i, source_layer_index=MATCH.src,
                  concat_dim=MATCH.cdim),
        LayerOutputShape(layer_index=MATCH.prev_i, dims=2,
                          d0=MATCH.d0, d1=MATCH.d1,
                          format=MATCH.fmt),
        LayerOutputShape(layer_index=MATCH.src, dims=2,
                          d1=MATCH.src_d1),
        TEST(lambda i, prev_i: i == prev_i + 1),
        NOT(LayerOutputShape(layer_index=MATCH.i)),
        salience=10
    )
    def sc_cat_2d(self, i, d0, d1, src_d1, fmt, **kwargs):
        # Concatenate along feature dim (dim=1 for 2D)
        self.declare(LayerOutputShape(
            layer_index=i,
            dims=2,
            d0=d0,
            d1=d1 + src_d1,
            d2=-1,
            format=fmt
        ))
