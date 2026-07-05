from experta import *

from lstm.facts import (
    PipelineConfig, LayerNode, LayerOutputShape,
    EmbeddingParam, LSTMParam, GRUParam, RNNParam, LinearParam,
    Conv1dParam, Pool1dParam, AdaptivePool1dParam, AttentionParam,
    DropoutParam, ActivationParam, PermuteParam, SqueezeParam,
    UnsqueezeParam, CatParam, ResidualParam,
    LayerCode, AutoUnpackCode,
    SelectHiddenParam, SelectCellParam, BmmParam, StackParam, SplitParam, ReshapeParam,
)


class CodeRules(KnowledgeEngine):
    """
    Code Generation Engine (salience=0).

    Each rule matches a LayerNode + its parameter fact + the previous
    LayerOutputShape (to auto-calculate in_features etc.), then declares
    a LayerCode fact with init_code and forward_code strings.
    """

    # =====================================================
    # CG_AUTO_UNPACK: Auto-unpack packed sequences for non-RNNs
    # =====================================================
    @Rule(
        LayerNode(index=MATCH.i, layer_type=MATCH.ltype),
        TEST(lambda ltype: ltype not in ("lstm", "gru", "rnn", "unpack_sequence", "pack_sequence")),
        LayerOutputShape(layer_index=MATCH.prev_i, format="packed"),
        TEST(lambda i, prev_i: i == prev_i + 1),
        NOT(AutoUnpackCode(layer_index=MATCH.i)),
        salience=100
    )
    def cg_auto_unpack(self, i):
        self.declare(AutoUnpackCode(
            layer_index=i,
            code="x, _ = pad_packed_sequence(x, batch_first=False)"
        ))

    # =====================================================
    # CG_EMBED: nn.Embedding
    # =====================================================
    @Rule(
        LayerNode(index=MATCH.i, layer_type="embedding"),
        EmbeddingParam(layer_index=MATCH.i, vocab_size=MATCH.vs,
                        embedding_dim=MATCH.ed),
        NOT(LayerCode(layer_index=MATCH.i)),
        salience=0
    )
    def cg_embedding(self, i, vs, ed):
        self.declare(LayerCode(
            layer_index=i,
            init_code="self.embedding = nn.Embedding({}, {})".format(vs, ed),
            forward_code="x = self.embedding(x)"
        ))

    # =====================================================
    # CG_LINEAR_INITIAL: nn.Linear as first layer (continuous)
    # =====================================================
    @Rule(
        LayerNode(index=0, layer_type="linear"),
        LinearParam(layer_index=0, out_features=MATCH.out_f),
        PipelineConfig(input_type="continuous", input_feature_size=MATCH.in_f),
        NOT(LayerCode(layer_index=0)),
        salience=1
    )
    def cg_linear_initial(self, out_f, in_f):
        self.declare(LayerCode(
            layer_index=0,
            init_code="self.proj_0 = nn.Linear({}, {})".format(in_f, out_f),
            forward_code="x = self.proj_0(x)"
        ))

    # =====================================================
    # CG_LSTM_INITIAL: LSTM as first layer (continuous)
    # =====================================================
    @Rule(
        LayerNode(index=0, layer_type="lstm"),
        LSTMParam(layer_index=0, hidden_size=MATCH.h,
                   num_layers=MATCH.nl, bidirectional=MATCH.bidir,
                   dropout=MATCH.dp),
        PipelineConfig(input_type="continuous", input_feature_size=MATCH.in_f),
        NOT(LayerCode(layer_index=0)),
        salience=1
    )
    def cg_lstm_initial(self, h, nl, bidir, dp, in_f):
        parts = [str(in_f), str(h)]
        parts.append("num_layers={}".format(nl))
        if bidir:
            parts.append("bidirectional=True")
        if dp > 0 and nl > 1:
            parts.append("dropout={}".format(dp))
        parts.append("batch_first=False")

        self.declare(LayerCode(
            layer_index=0,
            init_code="self.lstm_0 = nn.LSTM({})".format(", ".join(parts)),
            forward_code="x, (h_0, c_0) = self.lstm_0(x)"
        ))

    # =====================================================
    # CG_GRU_INITIAL: GRU as first layer (continuous)
    # =====================================================
    @Rule(
        LayerNode(index=0, layer_type="gru"),
        GRUParam(layer_index=0, hidden_size=MATCH.h,
                  num_layers=MATCH.nl, bidirectional=MATCH.bidir,
                  dropout=MATCH.dp),
        PipelineConfig(input_type="continuous", input_feature_size=MATCH.in_f),
        NOT(LayerCode(layer_index=0)),
        salience=1
    )
    def cg_gru_initial(self, h, nl, bidir, dp, in_f):
        parts = [str(in_f), str(h)]
        parts.append("num_layers={}".format(nl))
        if bidir:
            parts.append("bidirectional=True")
        if dp > 0 and nl > 1:
            parts.append("dropout={}".format(dp))
        parts.append("batch_first=False")

        self.declare(LayerCode(
            layer_index=0,
            init_code="self.gru_0 = nn.GRU({})".format(", ".join(parts)),
            forward_code="x, h_0 = self.gru_0(x)"
        ))

    # =====================================================
    # CG_RNN_INITIAL: RNN as first layer (continuous)
    # =====================================================
    @Rule(
        LayerNode(index=0, layer_type="rnn"),
        RNNParam(layer_index=0, hidden_size=MATCH.h,
                  num_layers=MATCH.nl, bidirectional=MATCH.bidir,
                  dropout=MATCH.dp),
        PipelineConfig(input_type="continuous", input_feature_size=MATCH.in_f),
        NOT(LayerCode(layer_index=0)),
        salience=1
    )
    def cg_rnn_initial(self, h, nl, bidir, dp, in_f):
        parts = [str(in_f), str(h)]
        parts.append("num_layers={}".format(nl))
        if bidir:
            parts.append("bidirectional=True")
        if dp > 0 and nl > 1:
            parts.append("dropout={}".format(dp))
        parts.append("batch_first=False")

        self.declare(LayerCode(
            layer_index=0,
            init_code="self.rnn_0 = nn.RNN({})".format(", ".join(parts)),
            forward_code="x, h_0 = self.rnn_0(x)"
        ))

    # =====================================================
    # CG_LSTM: nn.LSTM
    # =====================================================
    @Rule(
        LayerNode(index=MATCH.i, layer_type="lstm"),
        LSTMParam(layer_index=MATCH.i, hidden_size=MATCH.h,
                   num_layers=MATCH.nl, bidirectional=MATCH.bidir,
                   dropout=MATCH.dp),
        LayerOutputShape(layer_index=MATCH.prev_i, d2=MATCH.in_feat),
        TEST(lambda i, prev_i: i == prev_i + 1),
        NOT(LayerCode(layer_index=MATCH.i)),
        salience=0
    )
    def cg_lstm(self, i, h, nl, bidir, dp, in_feat, **kwargs):
        parts = [str(in_feat), str(h)]
        parts.append("num_layers={}".format(nl))
        if bidir:
            parts.append("bidirectional=True")
        if dp > 0 and nl > 1:
            parts.append("dropout={}".format(dp))
        parts.append("batch_first=False")

        self.declare(LayerCode(
            layer_index=i,
            init_code="self.lstm_{} = nn.LSTM({})".format(i, ", ".join(parts)),
            forward_code="x, (h_{0}, c_{0}) = self.lstm_{0}(x)".format(i)
        ))

    # =====================================================
    # CG_GRU: nn.GRU
    # =====================================================
    @Rule(
        LayerNode(index=MATCH.i, layer_type="gru"),
        GRUParam(layer_index=MATCH.i, hidden_size=MATCH.h,
                  num_layers=MATCH.nl, bidirectional=MATCH.bidir,
                  dropout=MATCH.dp),
        LayerOutputShape(layer_index=MATCH.prev_i, d2=MATCH.in_feat),
        TEST(lambda i, prev_i: i == prev_i + 1),
        NOT(LayerCode(layer_index=MATCH.i)),
        salience=0
    )
    def cg_gru(self, i, h, nl, bidir, dp, in_feat, **kwargs):
        parts = [str(in_feat), str(h)]
        parts.append("num_layers={}".format(nl))
        if bidir:
            parts.append("bidirectional=True")
        if dp > 0 and nl > 1:
            parts.append("dropout={}".format(dp))
        parts.append("batch_first=False")

        self.declare(LayerCode(
            layer_index=i,
            init_code="self.gru_{} = nn.GRU({})".format(i, ", ".join(parts)),
            forward_code="x, h_{0} = self.gru_{0}(x)".format(i)
        ))

    # =====================================================
    # CG_RNN: nn.RNN
    # =====================================================
    @Rule(
        LayerNode(index=MATCH.i, layer_type="rnn"),
        RNNParam(layer_index=MATCH.i, hidden_size=MATCH.h,
                  num_layers=MATCH.nl, bidirectional=MATCH.bidir,
                  dropout=MATCH.dp),
        LayerOutputShape(layer_index=MATCH.prev_i, d2=MATCH.in_feat),
        TEST(lambda i, prev_i: i == prev_i + 1),
        NOT(LayerCode(layer_index=MATCH.i)),
        salience=0
    )
    def cg_rnn(self, i, h, nl, bidir, dp, in_feat, **kwargs):
        parts = [str(in_feat), str(h)]
        parts.append("num_layers={}".format(nl))
        if bidir:
            parts.append("bidirectional=True")
        if dp > 0 and nl > 1:
            parts.append("dropout={}".format(dp))
        parts.append("batch_first=False")

        self.declare(LayerCode(
            layer_index=i,
            init_code="self.rnn_{} = nn.RNN({})".format(i, ", ".join(parts)),
            forward_code="x, h_{0} = self.rnn_{0}(x)".format(i)
        ))

    # =====================================================
    # CG_LINEAR_2D: nn.Linear (classifier head, 2D input)
    # =====================================================
    @Rule(
        LayerNode(index=MATCH.i, layer_type="linear"),
        LinearParam(layer_index=MATCH.i, out_features=MATCH.out_f),
        LayerOutputShape(layer_index=MATCH.prev_i, dims=2, d1=MATCH.in_f),
        TEST(lambda i, prev_i: i == prev_i + 1),
        NOT(LayerCode(layer_index=MATCH.i)),
        salience=0
    )
    def cg_linear_2d(self, i, out_f, in_f, **kwargs):
        self.declare(LayerCode(
            layer_index=i,
            init_code="self.fc_{} = nn.Linear({}, {})".format(i, in_f, out_f),
            forward_code="x = self.fc_{}(x)".format(i)
        ))

    # =====================================================
    # CG_LINEAR_3D: nn.Linear (projection, 3D input)
    # =====================================================
    @Rule(
        LayerNode(index=MATCH.i, layer_type="linear"),
        LinearParam(layer_index=MATCH.i, out_features=MATCH.out_f),
        LayerOutputShape(layer_index=MATCH.prev_i, dims=3, d2=MATCH.in_f),
        TEST(lambda i, prev_i: i == prev_i + 1),
        NOT(LayerCode(layer_index=MATCH.i)),
        salience=0
    )
    def cg_linear_3d(self, i, out_f, in_f, **kwargs):
        self.declare(LayerCode(
            layer_index=i,
            init_code="self.proj_{} = nn.Linear({}, {})".format(i, in_f, out_f),
            forward_code="x = self.proj_{}(x)".format(i)
        ))

    # =====================================================
    # CG_CONV1D: nn.Conv1d
    # =====================================================
    @Rule(
        LayerNode(index=MATCH.i, layer_type="conv1d"),
        Conv1dParam(layer_index=MATCH.i, out_channels=MATCH.oc,
                     kernel_size=MATCH.k, stride=MATCH.st, padding=MATCH.pad),
        LayerOutputShape(layer_index=MATCH.prev_i, dims=3, d1=MATCH.in_ch,
                          format="batch_chan_seq"),
        TEST(lambda i, prev_i: i == prev_i + 1),
        NOT(LayerCode(layer_index=MATCH.i)),
        salience=0
    )
    def cg_conv1d(self, i, oc, k, st, pad, in_ch, **kwargs):
        parts = [str(in_ch), str(oc), "kernel_size={}".format(k)]
        if st != 1:
            parts.append("stride={}".format(st))
        if pad != 0:
            parts.append("padding={}".format(pad))

        self.declare(LayerCode(
            layer_index=i,
            init_code="self.conv_{} = nn.Conv1d({})".format(i, ", ".join(parts)),
            forward_code="x = self.conv_{}(x)".format(i)
        ))

    # =====================================================
    # CG_MAXPOOL1D: nn.MaxPool1d
    # =====================================================
    @Rule(
        LayerNode(index=MATCH.i, layer_type="maxpool1d"),
        Pool1dParam(layer_index=MATCH.i, kernel_size=MATCH.k, stride=MATCH.st),
        NOT(LayerCode(layer_index=MATCH.i)),
        salience=0
    )
    def cg_maxpool1d(self, i, k, st):
        stride_str = ", stride={}".format(st) if st != k else ""
        self.declare(LayerCode(
            layer_index=i,
            init_code="self.maxpool_{} = nn.MaxPool1d({}{})".format(i, k, stride_str),
            forward_code="x = self.maxpool_{}(x)".format(i)
        ))

    # =====================================================
    # CG_AVGPOOL1D: nn.AvgPool1d
    # =====================================================
    @Rule(
        LayerNode(index=MATCH.i, layer_type="avgpool1d"),
        Pool1dParam(layer_index=MATCH.i, kernel_size=MATCH.k, stride=MATCH.st),
        NOT(LayerCode(layer_index=MATCH.i)),
        salience=0
    )
    def cg_avgpool1d(self, i, k, st):
        stride_str = ", stride={}".format(st) if st != k else ""
        self.declare(LayerCode(
            layer_index=i,
            init_code="self.avgpool_{} = nn.AvgPool1d({}{})".format(i, k, stride_str),
            forward_code="x = self.avgpool_{}(x)".format(i)
        ))

    # =====================================================
    # CG_ADAPTIVE_AVGPOOL1D: nn.AdaptiveAvgPool1d
    # =====================================================
    @Rule(
        LayerNode(index=MATCH.i, layer_type="adaptive_avgpool1d"),
        AdaptivePool1dParam(layer_index=MATCH.i, output_size=MATCH.os),
        NOT(LayerCode(layer_index=MATCH.i)),
        salience=0
    )
    def cg_adaptive_avgpool1d(self, i, os):
        self.declare(LayerCode(
            layer_index=i,
            init_code="self.adapt_avgpool_{} = nn.AdaptiveAvgPool1d({})".format(i, os),
            forward_code="x = self.adapt_avgpool_{}(x)".format(i)
        ))

    # =====================================================
    # CG_ADAPTIVE_MAXPOOL1D: nn.AdaptiveMaxPool1d
    # =====================================================
    @Rule(
        LayerNode(index=MATCH.i, layer_type="adaptive_maxpool1d"),
        AdaptivePool1dParam(layer_index=MATCH.i, output_size=MATCH.os),
        NOT(LayerCode(layer_index=MATCH.i)),
        salience=0
    )
    def cg_adaptive_maxpool1d(self, i, os):
        self.declare(LayerCode(
            layer_index=i,
            init_code="self.adapt_maxpool_{} = nn.AdaptiveMaxPool1d({})".format(i, os),
            forward_code="x = self.adapt_maxpool_{}(x)".format(i)
        ))

    # =====================================================
    # CG_ATTENTION: nn.MultiheadAttention
    # =====================================================
    @Rule(
        LayerNode(index=MATCH.i, layer_type="multihead_attention"),
        AttentionParam(layer_index=MATCH.i, num_heads=MATCH.nh),
        LayerOutputShape(layer_index=MATCH.prev_i, dims=3, d2=MATCH.edim),
        TEST(lambda i, prev_i: i == prev_i + 1),
        NOT(LayerCode(layer_index=MATCH.i)),
        salience=0
    )
    def cg_attention(self, i, nh, edim, **kwargs):
        self.declare(LayerCode(
            layer_index=i,
            init_code="self.attn_{} = nn.MultiheadAttention({}, {})".format(i, edim, nh),
            forward_code="x, _ = self.attn_{}(x, x, x)".format(i)
        ))

    # =====================================================
    # CG_LAYERNORM: nn.LayerNorm
    # =====================================================
    @Rule(
        LayerNode(index=MATCH.i, layer_type="layernorm"),
        LayerOutputShape(layer_index=MATCH.prev_i, d2=MATCH.feat),
        TEST(lambda i, prev_i: i == prev_i + 1),
        TEST(lambda feat: feat > 0),
        NOT(LayerCode(layer_index=MATCH.i)),
        salience=0
    )
    def cg_layernorm_3d(self, i, feat, **kwargs):
        self.declare(LayerCode(
            layer_index=i,
            init_code="self.ln_{} = nn.LayerNorm({})".format(i, feat),
            forward_code="x = self.ln_{}(x)".format(i)
        ))

    @Rule(
        LayerNode(index=MATCH.i, layer_type="layernorm"),
        LayerOutputShape(layer_index=MATCH.prev_i, dims=2, d1=MATCH.feat),
        TEST(lambda i, prev_i: i == prev_i + 1),
        NOT(LayerCode(layer_index=MATCH.i)),
        salience=0
    )
    def cg_layernorm_2d(self, i, feat, **kwargs):
        self.declare(LayerCode(
            layer_index=i,
            init_code="self.ln_{} = nn.LayerNorm({})".format(i, feat),
            forward_code="x = self.ln_{}(x)".format(i)
        ))

    # =====================================================
    # CG_BATCHNORM1D: nn.BatchNorm1d
    # =====================================================
    @Rule(
        LayerNode(index=MATCH.i, layer_type="batchnorm1d"),
        LayerOutputShape(layer_index=MATCH.prev_i, dims=2, d1=MATCH.feat),
        TEST(lambda i, prev_i: i == prev_i + 1),
        NOT(LayerCode(layer_index=MATCH.i)),
        salience=0
    )
    def cg_batchnorm1d(self, i, feat, **kwargs):
        self.declare(LayerCode(
            layer_index=i,
            init_code="self.bn_{} = nn.BatchNorm1d({})".format(i, feat),
            forward_code="x = self.bn_{}(x)".format(i)
        ))

    # =====================================================
    # CG_DROPOUT: nn.Dropout
    # =====================================================
    @Rule(
        LayerNode(index=MATCH.i, layer_type="dropout"),
        DropoutParam(layer_index=MATCH.i, p=MATCH.p),
        NOT(LayerCode(layer_index=MATCH.i)),
        salience=0
    )
    def cg_dropout(self, i, p):
        self.declare(LayerCode(
            layer_index=i,
            init_code="self.dropout_{} = nn.Dropout(p={})".format(i, p),
            forward_code="x = self.dropout_{}(x)".format(i)
        ))

    # =====================================================
    # CG_PERMUTE_SBF_BCS: Permute seq_batch_feat -> batch_chan_seq
    # =====================================================
    @Rule(
        LayerNode(index=MATCH.i, layer_type="permute"),
        PermuteParam(layer_index=MATCH.i, target_format="batch_chan_seq"),
        NOT(LayerCode(layer_index=MATCH.i)),
        salience=0
    )
    def cg_permute_sbf_to_bcs(self, i):
        self.declare(LayerCode(
            layer_index=i,
            init_code="",
            forward_code="x = x.permute(1, 2, 0)  # [Seq,Batch,Feat] -> [Batch,Feat,Seq]"
        ))

    # =====================================================
    # CG_PERMUTE_BCS_SBF: Permute batch_chan_seq -> seq_batch_feat
    # =====================================================
    @Rule(
        LayerNode(index=MATCH.i, layer_type="permute"),
        PermuteParam(layer_index=MATCH.i, target_format="seq_batch_feat"),
        NOT(LayerCode(layer_index=MATCH.i)),
        salience=0
    )
    def cg_permute_bcs_to_sbf(self, i):
        self.declare(LayerCode(
            layer_index=i,
            init_code="",
            forward_code="x = x.permute(2, 0, 1)  # [Batch,Chan,Seq] -> [Seq,Batch,Chan]"
        ))

    # =====================================================
    # CG_FLATTEN: nn.Flatten
    # =====================================================
    @Rule(
        LayerNode(index=MATCH.i, layer_type="flatten"),
        NOT(LayerCode(layer_index=MATCH.i)),
        salience=0
    )
    def cg_flatten(self, i):
        self.declare(LayerCode(
            layer_index=i,
            init_code="self.flatten_{} = nn.Flatten(start_dim=1)".format(i),
            forward_code="x = self.flatten_{}(x)".format(i)
        ))

    # =====================================================
    # CG_SLICE_LAST: x[-1, :, :]
    # =====================================================
    @Rule(
        LayerNode(index=MATCH.i, layer_type="slice_last"),
        NOT(LayerCode(layer_index=MATCH.i)),
        salience=0
    )
    def cg_slice_last(self, i):
        self.declare(LayerCode(
            layer_index=i,
            init_code="",
            forward_code="x = x[-1, :, :]  # Take last timestep: [Seq,Batch,H] -> [Batch,H]"
        ))

    # =====================================================
    # CG_SQUEEZE: Squeeze
    # =====================================================
    @Rule(
        LayerNode(index=MATCH.i, layer_type="squeeze"),
        SqueezeParam(layer_index=MATCH.i, dim=MATCH.d),
        NOT(LayerCode(layer_index=MATCH.i)),
        salience=0
    )
    def cg_squeeze(self, i, d):
        self.declare(LayerCode(
            layer_index=i,
            init_code="",
            forward_code="x = x.squeeze(dim={})".format(d)
        ))

    # =====================================================
    # CG_UNSQUEEZE: Unsqueeze
    # =====================================================
    @Rule(
        LayerNode(index=MATCH.i, layer_type="unsqueeze"),
        UnsqueezeParam(layer_index=MATCH.i, dim=MATCH.d),
        NOT(LayerCode(layer_index=MATCH.i)),
        salience=0
    )
    def cg_unsqueeze(self, i, d):
        self.declare(LayerCode(
            layer_index=i,
            init_code="",
            forward_code="x = x.unsqueeze(dim={})".format(d)
        ))

    # =====================================================
    # CG_PACK: pack_padded_sequence
    # =====================================================
    @Rule(
        LayerNode(index=MATCH.i, layer_type="pack_sequence"),
        NOT(LayerCode(layer_index=MATCH.i)),
        salience=0
    )
    def cg_pack(self, i):
        self.declare(LayerCode(
            layer_index=i,
            init_code="",
            forward_code="x = pack_padded_sequence(x, lengths, enforce_sorted=False)"
        ))

    # =====================================================
    # CG_UNPACK: pad_packed_sequence
    # =====================================================
    @Rule(
        LayerNode(index=MATCH.i, layer_type="unpack_sequence"),
        NOT(LayerCode(layer_index=MATCH.i)),
        salience=0
    )
    def cg_unpack(self, i):
        self.declare(LayerCode(
            layer_index=i,
            init_code="",
            forward_code="x, _ = pad_packed_sequence(x)"
        ))

    # =====================================================
    # CG_RESIDUAL_ADD: Residual / Skip Connection
    # =====================================================
    @Rule(
        LayerNode(index=MATCH.i, layer_type="residual_add"),
        ResidualParam(layer_index=MATCH.i, source_layer_index=MATCH.src),
        NOT(LayerCode(layer_index=MATCH.i)),
        salience=0
    )
    def cg_residual_add(self, i, src):
        self.declare(LayerCode(
            layer_index=i,
            init_code="",
            forward_code="x = x + residual_{}  # Residual from layer #{}".format(src, src)
        ))

    # =====================================================
    # CG_CAT: torch.cat
    # =====================================================
    @Rule(
        LayerNode(index=MATCH.i, layer_type="cat"),
        CatParam(layer_index=MATCH.i, source_layer_index=MATCH.src,
                  concat_dim=MATCH.cdim),
        NOT(LayerCode(layer_index=MATCH.i)),
        salience=0
    )
    def cg_cat(self, i, src, cdim):
        self.declare(LayerCode(
            layer_index=i,
            init_code="",
            forward_code="x = torch.cat([x, out_{}], dim={})  "
                         "# Concat with layer #{}".format(src, cdim, src)
        ))

    # =====================================================
    # CG_ACTIVATION: All activation functions
    # Each activation type gets its own rule
    # =====================================================

    # ReLU
    @Rule(
        LayerNode(index=MATCH.i, layer_type="activation"),
        ActivationParam(layer_index=MATCH.i, activation_type="relu"),
        NOT(LayerCode(layer_index=MATCH.i)),
        salience=0
    )
    def cg_relu(self, i):
        self.declare(LayerCode(
            layer_index=i,
            init_code="self.relu_{} = nn.ReLU()".format(i),
            forward_code="x = self.relu_{}(x)".format(i)
        ))

    # LeakyReLU
    @Rule(
        LayerNode(index=MATCH.i, layer_type="activation"),
        ActivationParam(layer_index=MATCH.i, activation_type="leaky_relu",
                         negative_slope=MATCH.ns),
        NOT(LayerCode(layer_index=MATCH.i)),
        salience=0
    )
    def cg_leaky_relu(self, i, ns):
        self.declare(LayerCode(
            layer_index=i,
            init_code="self.leaky_relu_{} = nn.LeakyReLU("
                      "negative_slope={})".format(i, ns),
            forward_code="x = self.leaky_relu_{}(x)".format(i)
        ))

    # GELU
    @Rule(
        LayerNode(index=MATCH.i, layer_type="activation"),
        ActivationParam(layer_index=MATCH.i, activation_type="gelu"),
        NOT(LayerCode(layer_index=MATCH.i)),
        salience=0
    )
    def cg_gelu(self, i):
        self.declare(LayerCode(
            layer_index=i,
            init_code="self.gelu_{} = nn.GELU()".format(i),
            forward_code="x = self.gelu_{}(x)".format(i)
        ))

    # Tanh
    @Rule(
        LayerNode(index=MATCH.i, layer_type="activation"),
        ActivationParam(layer_index=MATCH.i, activation_type="tanh"),
        NOT(LayerCode(layer_index=MATCH.i)),
        salience=0
    )
    def cg_tanh(self, i):
        self.declare(LayerCode(
            layer_index=i,
            init_code="self.tanh_{} = nn.Tanh()".format(i),
            forward_code="x = self.tanh_{}(x)".format(i)
        ))

    # Sigmoid
    @Rule(
        LayerNode(index=MATCH.i, layer_type="activation"),
        ActivationParam(layer_index=MATCH.i, activation_type="sigmoid"),
        NOT(LayerCode(layer_index=MATCH.i)),
        salience=0
    )
    def cg_sigmoid(self, i):
        self.declare(LayerCode(
            layer_index=i,
            init_code="self.sigmoid_{} = nn.Sigmoid()".format(i),
            forward_code="x = self.sigmoid_{}(x)".format(i)
        ))

    # Softmax
    @Rule(
        LayerNode(index=MATCH.i, layer_type="activation"),
        ActivationParam(layer_index=MATCH.i, activation_type="softmax",
                         dim=MATCH.d),
        NOT(LayerCode(layer_index=MATCH.i)),
        salience=0
    )
    def cg_softmax(self, i, d):
        self.declare(LayerCode(
            layer_index=i,
            init_code="self.softmax_{} = nn.Softmax(dim=-1)".format(i),
            forward_code="x = self.softmax_{}(x)".format(i)
        ))

    # LogSoftmax
    @Rule(
        LayerNode(index=MATCH.i, layer_type="activation"),
        ActivationParam(layer_index=MATCH.i, activation_type="log_softmax",
                         dim=MATCH.d),
        NOT(LayerCode(layer_index=MATCH.i)),
        salience=0
    )
    def cg_log_softmax(self, i, d):
        self.declare(LayerCode(
            layer_index=i,
            init_code="self.log_softmax_{} = nn.LogSoftmax(dim=-1)".format(i),
            forward_code="x = self.log_softmax_{}(x)".format(i)
        ))

    # =====================================================
    # NEW LAYERS
    # =====================================================

    # CG_SUM_DIRECTIONS: Sum Directions (fwd+bwd)
    @Rule(
        LayerNode(index=MATCH.i, layer_type="sum_directions"),
        LayerOutputShape(layer_index=MATCH.prev_i, dims=3, d2=MATCH.feat),
        TEST(lambda i, prev_i: i == prev_i + 1),
        NOT(LayerCode(layer_index=MATCH.i)),
        salience=0
    )
    def cg_sum_directions(self, i, feat):
        half = feat // 2
        self.declare(LayerCode(
            layer_index=i,
            init_code="",
            forward_code="x = x[:, :, :{} ] + x[:, :, {} :]  # Sum directions".format(half, half)
        ))

    # CG_SPLIT_DIRECTIONS: Split Directions
    @Rule(
        LayerNode(index=MATCH.i, layer_type="split_directions"),
        LayerOutputShape(layer_index=MATCH.prev_i, dims=3, d0=MATCH.seq, d1=MATCH.batch, d2=MATCH.feat),
        NOT(LayerCode(layer_index=MATCH.i)),
        salience=0
    )
    def cg_split_directions(self, i, seq, batch, feat):
        half = feat // 2
        self.declare(LayerCode(
            layer_index=i,
            init_code="",
            forward_code="x = x.view({}, {}, 2, {})  # Split directions".format(seq, batch, half)
        ))

    # CG_SELECT_HIDDEN: Select Hidden State (h_n)
    @Rule(
        LayerNode(index=MATCH.i, layer_type="select_hidden"),
        SelectHiddenParam(layer_index=MATCH.i, source_layer_index=MATCH.src),
        NOT(LayerCode(layer_index=MATCH.i)),
        salience=0
    )
    def cg_select_hidden(self, i, src):
        self.declare(LayerCode(
            layer_index=i,
            init_code="",
            forward_code="x = h_{}[-1]  # Select hidden state from layer {}".format(src, src)
        ))

    # CG_SELECT_CELL: Select Cell State (c_n)
    @Rule(
        LayerNode(index=MATCH.i, layer_type="select_cell"),
        SelectCellParam(layer_index=MATCH.i, source_layer_index=MATCH.src),
        NOT(LayerCode(layer_index=MATCH.i)),
        salience=0
    )
    def cg_select_cell(self, i, src):
        self.declare(LayerCode(
            layer_index=i,
            init_code="",
            forward_code="x = c_{}[-1]  # Select cell state from layer {}".format(src, src)
        ))

    # CG_MEAN_POOL: Mean Pool (over time)
    @Rule(
        LayerNode(index=MATCH.i, layer_type="mean_pool"),
        NOT(LayerCode(layer_index=MATCH.i)),
        salience=0
    )
    def cg_mean_pool(self, i):
        self.declare(LayerCode(
            layer_index=i,
            init_code="",
            forward_code="x = x.mean(dim=0)  # Mean pool over sequence"
        ))

    # CG_MAX_POOL_TIME: Max Pool (over time)
    @Rule(
        LayerNode(index=MATCH.i, layer_type="max_pool_time"),
        NOT(LayerCode(layer_index=MATCH.i)),
        salience=0
    )
    def cg_max_pool_time(self, i):
        self.declare(LayerCode(
            layer_index=i,
            init_code="",
            forward_code="x, _ = x.max(dim=0)  # Max pool over sequence"
        ))

    # CG_BMM: Batch Matmul
    @Rule(
        LayerNode(index=MATCH.i, layer_type="bmm"),
        BmmParam(layer_index=MATCH.i, source_layer_index=MATCH.src),
        NOT(LayerCode(layer_index=MATCH.i)),
        salience=0
    )
    def cg_bmm(self, i, src):
        self.declare(LayerCode(
            layer_index=i,
            init_code="",
            forward_code="x = torch.bmm(x, out_{})  # Batch matmul with layer {}".format(src, src)
        ))

    # CG_STACK: torch.stack
    @Rule(
        LayerNode(index=MATCH.i, layer_type="stack"),
        StackParam(layer_index=MATCH.i, source_layer_index=MATCH.src, dim=MATCH.d),
        NOT(LayerCode(layer_index=MATCH.i)),
        salience=0
    )
    def cg_stack(self, i, src, d):
        self.declare(LayerCode(
            layer_index=i,
            init_code="",
            forward_code="x = torch.stack([x, out_{}], dim={})  # Stack with layer {}".format(src, d, src)
        ))

    # CG_SPLIT: torch.split
    @Rule(
        LayerNode(index=MATCH.i, layer_type="split"),
        SplitParam(layer_index=MATCH.i, split_size_or_sections=MATCH.sz, dim=MATCH.d),
        NOT(LayerCode(layer_index=MATCH.i)),
        salience=0
    )
    def cg_split(self, i, sz, d):
        self.declare(LayerCode(
            layer_index=i,
            init_code="",
            forward_code="x = torch.split(x, {}, dim={})[0]  # Split and take first chunk".format(sz, d)
        ))

    # CG_RESHAPE: torch.reshape / view
    @Rule(
        LayerNode(index=MATCH.i, layer_type="reshape"),
        ReshapeParam(layer_index=MATCH.i, shape=MATCH.shape_str),
        NOT(LayerCode(layer_index=MATCH.i)),
        salience=0
    )
    def cg_reshape(self, i, shape_str):
        self.declare(LayerCode(
            layer_index=i,
            init_code="",
            forward_code="x = x.view({})  # Reshape".format(shape_str)
        ))
