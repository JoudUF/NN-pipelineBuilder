from experta import *

from lstm.facts import (
    PipelineConfig, LayerNode, LayerOutputShape,
    EmbeddingParam, LSTMParam, GRUParam, RNNParam, LinearParam,
    Conv1dParam, Pool1dParam, AdaptivePool1dParam, AttentionParam,
    DropoutParam, ActivationParam, PermuteParam,
    SelectHiddenParam, SelectCellParam, BmmParam, StackParam,
    SplitParam, ReshapeParam,                                
    Mismatch, FixRecommendation,
)


class PruningRules(KnowledgeEngine):
    """
    Pruning / Validation Engine.

    These rules fire at the HIGHEST salience to detect invalid
    layer sequences before shape calculation even runs.

    Categories:
        PR1 (salience=90): Dimensionality mismatch 3D -> 2D
        PR2 (salience=90): Dimensionality mismatch 2D -> 3D
        PR3 (salience=80): Axis format mismatch
        PR4 (salience=60): Feature size mismatch
        PR5 (salience=100): PackedSequence protocol violations
        PR6 (salience=100): Embedding position violations
        PR7 (salience=70): Structural redundancy / logical nonsense
    """

    # ==========================================================
    # PR5: PACKED SEQUENCE PROTOCOL VIOLATIONS (salience=100)
    # ==========================================================

    # PR5_PACK_NOT_RECURRENT: pack_sequence not followed by recurrent
    @Rule(
        LayerNode(index=MATCH.i, layer_type="pack_sequence"),
        LayerNode(index=MATCH.j, layer_type=MATCH.next_type),
        TEST(lambda i, j: j == i + 1),
        TEST(lambda next_type: next_type not in ("lstm", "gru", "rnn")),
        NOT(Mismatch(layer_index=MATCH.j, mismatch_type="packed_protocol")),
        salience=100
    )
    def pr5_pack_not_recurrent(self, j, next_type):
        self.declare(Mismatch(
            layer_index=j,
            prev_index=j - 1,
            mismatch_type="packed_protocol",
            description="pack_padded_sequence must be followed by a recurrent layer "
                        "(LSTM, GRU, or RNN), but found '{}' instead.".format(next_type),
            severity="error"
        ))
        self.declare(FixRecommendation(
            layer_index=j,
            fix_description="Move pack_padded_sequence to right before an LSTM/GRU/RNN layer, "
                            "or remove it.",
            insert_layer_type="reorder"
        ))

    # PR5_PACK_ALREADY_PACKED: pack_sequence receiving packed sequence
    @Rule(
        LayerNode(index=MATCH.i, layer_type="pack_sequence"),
        LayerOutputShape(layer_index=MATCH.prev_i, format="packed"),
        TEST(lambda i, prev_i: i == prev_i + 1),
        NOT(Mismatch(layer_index=MATCH.i, mismatch_type="packed_protocol")),
        salience=100
    )
    def pr5_pack_already_packed(self, i):
        self.declare(Mismatch(
            layer_index=i,
            prev_index=i - 1,
            mismatch_type="packed_protocol",
            description="Cannot pack an already packed sequence. "
                        "Double pack_padded_sequence detected.",
            severity="error"
        ))
        self.declare(FixRecommendation(
            layer_index=i,
            fix_description="Remove the duplicate pack_padded_sequence.",
            insert_layer_type="remove"
        ))

    # PR5_UNPACK_NOT_AFTER_RECURRENT: unpack not preceded by recurrent
    @Rule(
        LayerNode(index=MATCH.i, layer_type=MATCH.prev_type),
        LayerNode(index=MATCH.j, layer_type="unpack_sequence"),
        TEST(lambda i, j: j == i + 1),
        TEST(lambda prev_type: prev_type not in ("lstm", "gru", "rnn")),
        NOT(Mismatch(layer_index=MATCH.j, mismatch_type="packed_protocol")),
        salience=100
    )
    def pr5_unpack_not_after_recurrent(self, j, prev_type):
        self.declare(Mismatch(
            layer_index=j,
            prev_index=j - 1,
            mismatch_type="packed_protocol",
            description="pad_packed_sequence must come after a recurrent layer "
                        "(LSTM, GRU, or RNN), but found '{}' before it.".format(prev_type),
            severity="error"
        ))
        self.declare(FixRecommendation(
            layer_index=j,
            fix_description="Move pad_packed_sequence to right after an LSTM/GRU/RNN layer, "
                            "or remove it.",
            insert_layer_type="reorder"
        ))

    # PR5_DOUBLE_UNPACK: unpack_sequence -> unpack_sequence
    @Rule(
        LayerNode(index=MATCH.i, layer_type="unpack_sequence"),
        LayerNode(index=MATCH.j, layer_type="unpack_sequence"),
        TEST(lambda i, j: j == i + 1),
        NOT(Mismatch(layer_index=MATCH.j, mismatch_type="packed_protocol")),
        salience=100
    )
    def pr5_double_unpack(self, j):
        self.declare(Mismatch(
            layer_index=j,
            prev_index=j - 1,
            mismatch_type="packed_protocol",
            description="Cannot unpack an already unpacked sequence. "
                        "Double pad_packed_sequence detected.",
            severity="error"
        ))
        self.declare(FixRecommendation(
            layer_index=j,
            fix_description="Remove the duplicate pad_packed_sequence.",
            insert_layer_type="remove"
        ))

    # ==========================================================
    # PR6: EMBEDDING POSITION VIOLATIONS (salience=100)
    # ==========================================================

    # PR6_EMBED_NOT_FIRST: Embedding at index > 0
    @Rule(
        LayerNode(index=MATCH.i, layer_type="embedding"),
        TEST(lambda i: i > 0),
        NOT(Mismatch(layer_index=MATCH.i, mismatch_type="embedding_position")),
        salience=100
    )
    def pr6_embed_not_first(self, i):
        self.declare(Mismatch(
            layer_index=i,
            prev_index=i - 1,
            mismatch_type="embedding_position",
            description="nn.Embedding must be the FIRST layer (index 0). "
                        "It requires integer tensor input, but all other layers "
                        "output float tensors.",
            severity="error"
        ))
        self.declare(FixRecommendation(
            layer_index=i,
            fix_description="Move nn.Embedding to the beginning of the pipeline (index 0).",
            insert_layer_type="reorder"
        ))

    # PR6_DOUBLE_EMBED: Embedding -> Embedding
    @Rule(
        LayerNode(index=MATCH.i, layer_type="embedding"),
        LayerNode(index=MATCH.j, layer_type="embedding"),
        TEST(lambda i, j: j == i + 1),
        NOT(Mismatch(layer_index=MATCH.j, mismatch_type="embedding_position")),
        salience=100
    )
    def pr6_double_embed(self, j):
        self.declare(Mismatch(
            layer_index=j,
            prev_index=j - 1,
            mismatch_type="embedding_position",
            description="Double nn.Embedding detected. Embedding outputs float tensors, "
                        "but Embedding requires integer input.",
            severity="error"
        ))
        self.declare(FixRecommendation(
            layer_index=j,
            fix_description="Remove the duplicate nn.Embedding layer.",
            insert_layer_type="remove"
        ))

    # ==========================================================
    # PR1: DIMENSIONALITY MISMATCH 3D -> 2D (salience=90)
    # Recurrent/3D layers -> layers that need 2D input
    # ==========================================================

    # PR1_RECURRENT_TO_BATCHNORM: 3D -> BatchNorm1d (needs 2D)
    @Rule(
        LayerNode(index=MATCH.i, layer_type=MATCH.prev_type),
        LayerNode(index=MATCH.j, layer_type="batchnorm1d"),
        TEST(lambda i, j: j == i + 1),
        TEST(lambda prev_type: prev_type in ("lstm", "gru", "rnn",
                                              "multihead_attention",
                                              "unpack_sequence")),
        NOT(Mismatch(layer_index=MATCH.j, mismatch_type="dim_3d_to_2d")),
        salience=90
    )
    def pr1_recurrent_to_batchnorm(self, j, prev_type):
        self.declare(Mismatch(
            layer_index=j,
            prev_index=j - 1,
            mismatch_type="dim_3d_to_2d",
            description="'{}' outputs a 3D tensor [Seq, Batch, Hidden], but "
                        "nn.BatchNorm1d expects a 2D tensor [Batch, Features]. "
                        "The sequence dimension must be removed first.".format(prev_type),
            severity="error"
        ))
        self.declare(FixRecommendation(
            layer_index=j,
            fix_description="Insert a 'Slice Last Timestep' or 'nn.Flatten' layer "
                            "before nn.BatchNorm1d to reduce 3D -> 2D.",
            insert_layer_type="slice_last"
        ))

    # ==========================================================
    # PR2: DIMENSIONALITY MISMATCH 2D -> 3D (salience=90)
    # Flat 2D output -> layers that need 3D sequence input
    # ==========================================================

    # PR2_FLAT_TO_RECURRENT: 2D -> LSTM/GRU/RNN
    @Rule(
        LayerOutputShape(layer_index=MATCH.prev_i, dims=2),
        LayerNode(index=MATCH.i, layer_type=MATCH.rnn_type),
        TEST(lambda i, prev_i: i == prev_i + 1),
        TEST(lambda rnn_type: rnn_type in ("lstm", "gru", "rnn")),
        NOT(Mismatch(layer_index=MATCH.i, mismatch_type="dim_2d_to_3d")),
        salience=90
    )
    def pr2_flat_to_recurrent(self, i, rnn_type):
        self.declare(Mismatch(
            layer_index=i,
            prev_index=i - 1,
            mismatch_type="dim_2d_to_3d",
            description="Previous layer outputs a 2D tensor [Batch, Features], but "
                        "'{}' requires a 3D sequential tensor [Seq, Batch, Features]. "
                        "Cannot feed flat data into a recurrent layer.".format(rnn_type),
            severity="error"
        ))
        self.declare(FixRecommendation(
            layer_index=i,
            fix_description="This is a structural error. Recurrent layers cannot come "
                            "after dimensionality reduction. Reorder your pipeline so that "
                            "LSTM/GRU/RNN comes before SliceLast/Flatten/Pooling.",
            insert_layer_type="reorder"
        ))

    # PR2_FLAT_TO_CONV: 2D -> Conv1d
    @Rule(
        LayerOutputShape(layer_index=MATCH.prev_i, dims=2),
        LayerNode(index=MATCH.i, layer_type="conv1d"),
        TEST(lambda i, prev_i: i == prev_i + 1),
        NOT(Mismatch(layer_index=MATCH.i, mismatch_type="dim_2d_to_3d")),
        salience=90
    )
    def pr2_flat_to_conv(self, i):
        self.declare(Mismatch(
            layer_index=i,
            prev_index=i - 1,
            mismatch_type="dim_2d_to_3d",
            description="Previous layer outputs a 2D tensor [Batch, Features], but "
                        "nn.Conv1d requires a 3D tensor [Batch, Channels, Seq].",
            severity="error"
        ))
        self.declare(FixRecommendation(
            layer_index=i,
            fix_description="nn.Conv1d cannot operate on flat 2D data. "
                            "Place it before dimensionality reduction.",
            insert_layer_type="reorder"
        ))

    # PR2_FLAT_TO_ATTENTION: 2D -> MultiheadAttention
    @Rule(
        LayerOutputShape(layer_index=MATCH.prev_i, dims=2),
        LayerNode(index=MATCH.i, layer_type="multihead_attention"),
        TEST(lambda i, prev_i: i == prev_i + 1),
        NOT(Mismatch(layer_index=MATCH.i, mismatch_type="dim_2d_to_3d")),
        salience=90
    )
    def pr2_flat_to_attention(self, i):
        self.declare(Mismatch(
            layer_index=i,
            prev_index=i - 1,
            mismatch_type="dim_2d_to_3d",
            description="Previous layer outputs a 2D tensor [Batch, Features], but "
                        "nn.MultiheadAttention requires 3D [Seq, Batch, Embed_Dim].",
            severity="error"
        ))
        self.declare(FixRecommendation(
            layer_index=i,
            fix_description="MultiheadAttention needs sequential 3D data. "
                            "Place it before dimensionality reduction.",
            insert_layer_type="reorder"
        ))

    # ==========================================================
    # PR3: AXIS FORMAT MISMATCH (salience=80)
    # Both 3D, but wrong axis arrangement
    # ==========================================================

    # PR3_SBF_TO_CONV: seq_batch_feat -> Conv1d (needs batch_chan_seq)
    @Rule(
        LayerOutputShape(layer_index=MATCH.prev_i, dims=3,
                          format="seq_batch_feat"),
        LayerNode(index=MATCH.i, layer_type="conv1d"),
        TEST(lambda i, prev_i: i == prev_i + 1),
        NOT(Mismatch(layer_index=MATCH.i, mismatch_type="format")),
        salience=80
    )
    def pr3_sbf_to_conv(self, i):
        self.declare(Mismatch(
            layer_index=i,
            prev_index=i - 1,
            mismatch_type="format",
            description="Previous layer outputs [Seq, Batch, Features] format, but "
                        "nn.Conv1d expects [Batch, Channels, Seq] format. "
                        "A Permute/Transpose operation is needed.",
            severity="error"
        ))
        self.declare(FixRecommendation(
            layer_index=i,
            fix_description="Insert a 'Permute' layer with target_format='batch_chan_seq' "
                            "before nn.Conv1d.",
            insert_layer_type="permute"
        ))

    # PR3_SBF_TO_POOL: seq_batch_feat -> MaxPool1d/AvgPool1d
    @Rule(
        LayerOutputShape(layer_index=MATCH.prev_i, dims=3,
                          format="seq_batch_feat"),
        LayerNode(index=MATCH.i, layer_type=MATCH.pool_type),
        TEST(lambda i, prev_i: i == prev_i + 1),
        TEST(lambda pool_type: pool_type in ("maxpool1d", "avgpool1d")),
        NOT(Mismatch(layer_index=MATCH.i, mismatch_type="format")),
        salience=80
    )
    def pr3_sbf_to_pool(self, i, pool_type):
        self.declare(Mismatch(
            layer_index=i,
            prev_index=i - 1,
            mismatch_type="format",
            description="Previous layer outputs [Seq, Batch, Features] format, but "
                        "'{}' expects [Batch, Channels, Seq] format.".format(pool_type),
            severity="error"
        ))
        self.declare(FixRecommendation(
            layer_index=i,
            fix_description="Insert a 'Permute' layer with target_format='batch_chan_seq' "
                            "before the pooling layer.",
            insert_layer_type="permute"
        ))

    # PR3_SBF_TO_ADAPTIVE: seq_batch_feat -> AdaptivePool
    @Rule(
        LayerOutputShape(layer_index=MATCH.prev_i, dims=3,
                          format="seq_batch_feat"),
        LayerNode(index=MATCH.i, layer_type=MATCH.pool_type),
        TEST(lambda i, prev_i: i == prev_i + 1),
        TEST(lambda pool_type: pool_type in ("adaptive_avgpool1d", "adaptive_maxpool1d")),
        NOT(Mismatch(layer_index=MATCH.i, mismatch_type="format")),
        salience=80
    )
    def pr3_sbf_to_adaptive(self, i, pool_type):
        self.declare(Mismatch(
            layer_index=i,
            prev_index=i - 1,
            mismatch_type="format",
            description="Previous layer outputs [Seq, Batch, Features] format, but "
                        "'{}' expects [Batch, Channels, Seq] format.".format(pool_type),
            severity="error"
        ))
        self.declare(FixRecommendation(
            layer_index=i,
            fix_description="Insert a 'Permute' layer with target_format='batch_chan_seq' "
                            "before the adaptive pooling layer.",
            insert_layer_type="permute"
        ))

    # PR3_BCS_TO_RECURRENT: batch_chan_seq -> LSTM/GRU/RNN (needs seq_batch_feat)
    @Rule(
        LayerOutputShape(layer_index=MATCH.prev_i, dims=3,
                          format="batch_chan_seq"),
        LayerNode(index=MATCH.i, layer_type=MATCH.rnn_type),
        TEST(lambda i, prev_i: i == prev_i + 1),
        TEST(lambda rnn_type: rnn_type in ("lstm", "gru", "rnn")),
        NOT(Mismatch(layer_index=MATCH.i, mismatch_type="format")),
        salience=80
    )
    def pr3_bcs_to_recurrent(self, i, rnn_type):
        self.declare(Mismatch(
            layer_index=i,
            prev_index=i - 1,
            mismatch_type="format",
            description="Previous layer outputs [Batch, Channels, Seq] format, but "
                        "'{}' expects [Seq, Batch, Features] format.".format(rnn_type),
            severity="error"
        ))
        self.declare(FixRecommendation(
            layer_index=i,
            fix_description="Insert a 'Permute' layer with target_format='seq_batch_feat' "
                            "before the recurrent layer.",
            insert_layer_type="permute"
        ))

    # PR3_BCS_TO_ATTENTION: batch_chan_seq -> MultiheadAttention
    @Rule(
        LayerOutputShape(layer_index=MATCH.prev_i, dims=3,
                          format="batch_chan_seq"),
        LayerNode(index=MATCH.i, layer_type="multihead_attention"),
        TEST(lambda i, prev_i: i == prev_i + 1),
        NOT(Mismatch(layer_index=MATCH.i, mismatch_type="format")),
        salience=80
    )
    def pr3_bcs_to_attention(self, i):
        self.declare(Mismatch(
            layer_index=i,
            prev_index=i - 1,
            mismatch_type="format",
            description="Previous layer outputs [Batch, Channels, Seq] format, but "
                        "nn.MultiheadAttention expects [Seq, Batch, Embed_Dim] format.",
            severity="error"
        ))
        self.declare(FixRecommendation(
            layer_index=i,
            fix_description="Insert a 'Permute' layer with target_format='seq_batch_feat' "
                            "before MultiheadAttention.",
            insert_layer_type="permute"
        ))

    # ==========================================================
    # PR7: STRUCTURAL REDUNDANCY / LOGICAL NONSENSE (salience=70)
    # ==========================================================

    # PR7_DOUBLE_ACTIVATION: Activation -> Activation
    @Rule(
        LayerNode(index=MATCH.i, layer_type="activation"),
        LayerNode(index=MATCH.j, layer_type="activation"),
        TEST(lambda i, j: j == i + 1),
        NOT(Mismatch(layer_index=MATCH.j, mismatch_type="redundancy")),
        salience=70
    )
    def pr7_double_activation(self, j):
        self.declare(Mismatch(
            layer_index=j,
            prev_index=j - 1,
            mismatch_type="redundancy",
            description="Two consecutive activation functions with no Linear layer "
                        "between them. The first activation's output is immediately "
                        "distorted by the second, destroying learned features.",
            severity="warning"
        ))
        self.declare(FixRecommendation(
            layer_index=j,
            fix_description="Remove one activation, or insert an nn.Linear layer "
                            "between them.",
            insert_layer_type="linear"
        ))

    # PR7_DOUBLE_DROPOUT: Dropout -> Dropout
    @Rule(
        LayerNode(index=MATCH.i, layer_type="dropout"),
        LayerNode(index=MATCH.j, layer_type="dropout"),
        TEST(lambda i, j: j == i + 1),
        NOT(Mismatch(layer_index=MATCH.j, mismatch_type="redundancy")),
        salience=70
    )
    def pr7_double_dropout(self, j):
        self.declare(Mismatch(
            layer_index=j,
            prev_index=j - 1,
            mismatch_type="redundancy",
            description="Stacking two nn.Dropout layers compounds the drop rate "
                        "in an uncontrolled way. This is almost certainly an error.",
            severity="warning"
        ))
        self.declare(FixRecommendation(
            layer_index=j,
            fix_description="Remove one nn.Dropout and adjust the remaining one's rate.",
            insert_layer_type="remove"
        ))

    # PR7_DOUBLE_FLATTEN: Flatten -> Flatten
    @Rule(
        LayerNode(index=MATCH.i, layer_type="flatten"),
        LayerNode(index=MATCH.j, layer_type="flatten"),
        TEST(lambda i, j: j == i + 1),
        NOT(Mismatch(layer_index=MATCH.j, mismatch_type="redundancy")),
        salience=70
    )
    def pr7_double_flatten(self, j):
        self.declare(Mismatch(
            layer_index=j,
            prev_index=j - 1,
            mismatch_type="redundancy",
            description="The tensor is already flat (2D) after the first nn.Flatten. "
                        "Second Flatten has no effect.",
            severity="warning"
        ))
        self.declare(FixRecommendation(
            layer_index=j,
            fix_description="Remove the duplicate nn.Flatten.",
            insert_layer_type="remove"
        ))

    # PR7_DOUBLE_SLICE: SliceLast -> SliceLast
    @Rule(
        LayerNode(index=MATCH.i, layer_type="slice_last"),
        LayerNode(index=MATCH.j, layer_type="slice_last"),
        TEST(lambda i, j: j == i + 1),
        NOT(Mismatch(layer_index=MATCH.j, mismatch_type="redundancy")),
        salience=70
    )
    def pr7_double_slice(self, j):
        self.declare(Mismatch(
            layer_index=j,
            prev_index=j - 1,
            mismatch_type="redundancy",
            description="The tensor is already 2D [Batch, Features] after the first "
                        "Slice Last. Cannot slice the sequence dimension again.",
            severity="warning"
        ))
        self.declare(FixRecommendation(
            layer_index=j,
            fix_description="Remove the duplicate Slice Last Timestep.",
            insert_layer_type="remove"
        ))

    # PR7_SLICE_THEN_FLATTEN: SliceLast -> Flatten (already 2D)
    @Rule(
        LayerNode(index=MATCH.i, layer_type="slice_last"),
        LayerNode(index=MATCH.j, layer_type="flatten"),
        TEST(lambda i, j: j == i + 1),
        NOT(Mismatch(layer_index=MATCH.j, mismatch_type="redundancy")),
        salience=70
    )
    def pr7_slice_then_flatten(self, j):
        self.declare(Mismatch(
            layer_index=j,
            prev_index=j - 1,
            mismatch_type="redundancy",
            description="Slice Last already reduces the tensor to 2D [Batch, Features]. "
                        "nn.Flatten after it is redundant.",
            severity="warning"
        ))
        self.declare(FixRecommendation(
            layer_index=j,
            fix_description="Remove nn.Flatten — the tensor is already 2D.",
            insert_layer_type="remove"
        ))

    # PR7_COMPETING_NORM: BatchNorm1d -> LayerNorm or reverse
    @Rule(
        LayerNode(index=MATCH.i, layer_type=MATCH.norm1),
        LayerNode(index=MATCH.j, layer_type=MATCH.norm2),
        TEST(lambda i, j: j == i + 1),
        TEST(lambda norm1, norm2: {norm1, norm2} == {"batchnorm1d", "layernorm"}),
        NOT(Mismatch(layer_index=MATCH.j, mismatch_type="redundancy")),
        salience=70
    )
    def pr7_competing_norm(self, j, norm1, norm2):
        self.declare(Mismatch(
            layer_index=j,
            prev_index=j - 1,
            mismatch_type="redundancy",
            description="Consecutive normalization layers ({} -> {}) are competing. "
                        "They normalize the same data differently, which is "
                        "counterproductive.".format(norm1, norm2),
            severity="warning"
        ))
        self.declare(FixRecommendation(
            layer_index=j,
            fix_description="Keep only one normalization layer. For recurrent pipelines, "
                            "nn.LayerNorm is recommended over nn.BatchNorm1d.",
            insert_layer_type="remove"
        ))

    # PR7_ACTIVATION_BEFORE_LINEAR: Softmax/LogSoftmax -> Linear
    @Rule(
        LayerNode(index=MATCH.i, layer_type="activation"),
        ActivationParam(layer_index=MATCH.i, activation_type=MATCH.atype),
        LayerNode(index=MATCH.j, layer_type="linear"),
        TEST(lambda i, j: j == i + 1),
        TEST(lambda atype: atype in ("softmax", "log_softmax")),
        NOT(Mismatch(layer_index=MATCH.j, mismatch_type="redundancy")),
        salience=70
    )
    def pr7_activation_before_linear(self, j, atype):
        self.declare(Mismatch(
            layer_index=j,
            prev_index=j - 1,
            mismatch_type="redundancy",
            description="'{}' is applied BEFORE nn.Linear. This means the probabilities "
                        "are fed into another linear transform, destroying the "
                        "probability distribution. The activation should come AFTER "
                        "the final Linear layer.".format(atype),
            severity="error"
        ))
        self.declare(FixRecommendation(
            layer_index=j,
            fix_description="Reorder: nn.Linear should come BEFORE {} "
                            "(not after).".format(atype),
            insert_layer_type="reorder"
        ))

    # ==========================================================
    # PR4: FEATURE SIZE MISMATCH (salience=60)
    # Correct dims & format, but feature sizes don't align
    # ==========================================================

    # PR4_ATTENTION_HEADS: embed_dim % num_heads != 0
    @Rule(
        LayerNode(index=MATCH.i, layer_type="multihead_attention"),
        AttentionParam(layer_index=MATCH.i, num_heads=MATCH.nh),
        LayerOutputShape(layer_index=MATCH.prev_i, dims=3, d2=MATCH.edim,
                          format="seq_batch_feat"),
        TEST(lambda i, prev_i: i == prev_i + 1),
        TEST(lambda edim, nh: edim % nh != 0),
        NOT(Mismatch(layer_index=MATCH.i, mismatch_type="feature_size")),
        salience=60
    )
    def pr4_attention_heads(self, i, nh, edim):
        self.declare(Mismatch(
            layer_index=i,
            prev_index=i - 1,
            mismatch_type="feature_size",
            description="MultiheadAttention requires embed_dim ({}) to be divisible "
                        "by num_heads ({}), but {} % {} = {}.".format(
                            edim, nh, edim, nh, edim % nh),
            severity="error"
        ))
        self.declare(FixRecommendation(
            layer_index=i,
            fix_description="Adjust num_heads to a divisor of {} "
                            "(e.g., 1, 2, 4, 8, ...).".format(edim),
            insert_layer_type="adjust"
        ))

    # ==========================================================
    # PR8: NEW LAYER MISMATCHES
    # ==========================================================

    # PR8_SELECT_HIDDEN_NOT_RNN: select_hidden must target an RNN
    @Rule(
        LayerNode(index=MATCH.i, layer_type="select_hidden"),
        SelectHiddenParam(layer_index=MATCH.i, source_layer_index=MATCH.src),
        LayerNode(index=MATCH.src, layer_type=MATCH.src_type),
        TEST(lambda src_type: src_type not in ("lstm", "gru", "rnn")),
        NOT(Mismatch(layer_index=MATCH.i, mismatch_type="invalid_source")),
        salience=90
    )
    def pr8_select_hidden_not_rnn(self, i, src, src_type):
        self.declare(Mismatch(
            layer_index=i,
            prev_index=i - 1,
            mismatch_type="invalid_source",
            description="Select Hidden State requires a recurrent layer as source, "
                        "but layer {} is '{}'.".format(src, src_type),
            severity="error"
        ))
        self.declare(FixRecommendation(
            layer_index=i,
            fix_description="Change source_layer_index to point to an LSTM/GRU/RNN.",
            insert_layer_type="adjust"
        ))

    # PR8_SELECT_CELL_NOT_LSTM: select_cell must target an LSTM
    @Rule(
        LayerNode(index=MATCH.i, layer_type="select_cell"),
        SelectCellParam(layer_index=MATCH.i, source_layer_index=MATCH.src),
        LayerNode(index=MATCH.src, layer_type=MATCH.src_type),
        TEST(lambda src_type: src_type != "lstm"),
        NOT(Mismatch(layer_index=MATCH.i, mismatch_type="invalid_source")),
        salience=90
    )
    def pr8_select_cell_not_lstm(self, i, src, src_type):
        self.declare(Mismatch(
            layer_index=i,
            prev_index=i - 1,
            mismatch_type="invalid_source",
            description="Select Cell State requires an LSTM layer as source, "
                        "but layer {} is '{}'.".format(src, src_type),
            severity="error"
        ))
        self.declare(FixRecommendation(
            layer_index=i,
            fix_description="Change source_layer_index to point to an LSTM.",
            insert_layer_type="adjust"
        ))

    # PR8_POOL_NOT_3D: mean_pool/max_pool_time needs 3D
    @Rule(
        LayerNode(index=MATCH.i, layer_type=MATCH.pool_type),
        LayerOutputShape(layer_index=MATCH.prev_i, dims=MATCH.dims),
        TEST(lambda i, prev_i: i == prev_i + 1),
        TEST(lambda pool_type: pool_type in ("mean_pool", "max_pool_time")),
        TEST(lambda dims: dims != 3),
        NOT(Mismatch(layer_index=MATCH.i, mismatch_type="dim_mismatch")),
        salience=90
    )
    def pr8_pool_not_3d(self, i, pool_type, dims):
        self.declare(Mismatch(
            layer_index=i,
            prev_index=i - 1,
            mismatch_type="dim_mismatch",
            description="'{}' requires a 3D sequential tensor, but previous "
                        "layer output is {}D.".format(pool_type, dims),
            severity="error"
        ))
        self.declare(FixRecommendation(
            layer_index=i,
            fix_description="Ensure the layer before pooling outputs a 3D sequence.",
            insert_layer_type="reorder"
        ))

