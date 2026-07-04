from experta import *
from facts import *

class ArchitectureRules(KnowledgeEngine):

    # DL-BIN-CLASS-001 (Standard Vector Data)
    @Rule(
        Problem(task_type="binary_classification"),
        Dataset(input_modality="tabular")
    )
    def bin_class_mlp(self):
        self.declare(ScoreEvidence(
            arch="MLP", 
            points=100, 
            reason="Goodfellow Ch6: Fixed-size vectors without spatial topology.", 
            source_rule="DL-BIN-CLASS-001"
        ))

    # DL-BIN-CLASS-002 (Grid-Structured Data)
    @Rule(
        Problem(task_type="binary_classification"),
        Dataset(input_modality="image")
    )
    def bin_class_cnn(self):
        self.declare(ScoreEvidence(
            arch="CNN", 
            points=100, 
            reason="Goodfellow Ch9: Grid-like topology requires translation invariance and parameter sharing.", 
            source_rule="DL-BIN-CLASS-002"
        ))

    # DL-BIN-CLASS-003 (Sequential Data)
    @Rule(
        Problem(task_type="binary_classification"),
        Dataset(input_modality="text") 
    )
    def bin_class_rnn(self):
        self.declare(ScoreEvidence(
            arch="LSTM/GRU", 
            points=100, 
            reason="Goodfellow Ch10: Sequential data with long-term dependencies.", 
            source_rule="DL-BIN-CLASS-003"
        ))
        
        
    # =====================================================
    # DL-MULTI-CLASS-001 (Standard Vector Data / Tabular)
    # =====================================================
    @Rule(
        Problem(task_type="multiclass_classification"),
        Dataset(input_modality="tabular")
    )
    def multi_class_mlp(self):
        self.declare(ScoreEvidence(
            arch="MLP", 
            points=100, 
            reason="Goodfellow Ch6: Tabular data for multiclass requires a deep feedforward network to compress features.", 
            source_rule="DL-MULTI-CLASS-001"
        ))

    # =====================================================
    # DL-MULTI-CLASS-002 (Grid-Structured Data / Image)
    # =====================================================
    @Rule(
        Problem(task_type="multiclass_classification"),
        Dataset(input_modality="image")
    )
    def multi_class_cnn(self):
        self.declare(ScoreEvidence(
            arch="CNN", 
            points=100, 
            reason="Goodfellow Ch9: Grid topologies require translation equivariance and sparse connectivity.", 
            source_rule="DL-MULTI-CLASS-002"
        ))

    # =====================================================
    # DL-MULTI-CLASS-003 (Sequential Data / Text)
    # =====================================================
    @Rule(
        Problem(task_type="multiclass_classification"),
        Dataset(input_modality="text")
    )
    def multi_class_rnn(self):
        self.declare(ScoreEvidence(
            arch="LSTM/GRU", 
            points=100, 
            reason="Goodfellow Ch10: Variable sequence lengths with long-term dependencies require recurrent gated logic.", 
            source_rule="DL-MULTI-CLASS-003"
        ))
        
    # =====================================================
    # DL-MDN-001 (Multimodal Regression / MDN)
    # =====================================================
    @Rule(
        Problem(task_type="multimodal_regression"),
        Dataset(input_modality="tabular")
    )
    def multimodal_mlp(self):
        self.declare(ScoreEvidence(
            arch="MLP", 
            points=100, 
            reason="Goodfellow Ch6: Tabular multimodal data requires a Mixture Density Network backbone.", 
            source_rule="DL-MDN-001"
        ))

    # =====================================================
    # DL-REG-001 (Standard Regression)
    # =====================================================
    @Rule(
        Problem(task_type="regression"),
        Dataset(input_modality="tabular")
    )
    def regression_mlp(self):
        self.declare(ScoreEvidence(
            arch="MLP", 
            points=100, 
            reason="Goodfellow Ch6: Tabular continuous data fits standard feedforward architectures.", 
            source_rule="DL-REG-001"
        ))
        
    # =====================================================
    # DL-REG-IMAGE-001: Image Regression (CNN Backbone)
    # Ref: Goodfellow Ch9 & Ch6 (Sec 6.2.2.1)
    # =====================================================
    @Rule(
        Problem(task_type="regression"),
        Dataset(input_modality="image")
    )
    def image_regression_cnn(self):
        self.declare(ScoreEvidence(
            arch="CNN", 
            points=100, 
            reason="Goodfellow Ch9: Continuous numerical prediction from visual grid topologies requires a CNN backbone.", 
            source_rule="DL-REG-IMAGE-001"
        ))
        
    # =====================================================
    # DL-TRANS-001 (Translation / Sequence to Sequence)
    # =====================================================
    @Rule(Problem(task_type="translation"), Dataset(input_modality="text"))
    def translation_seq2seq(self):
        self.declare(ScoreEvidence(
            arch="Encoder-Decoder RNN", points=100, 
            reason="Goodfellow Ch10: Mapping sequences of different lengths requires an Encoder-Decoder architecture.", source_rule="DL-TRANS-001"
        ))

    # =====================================================
    # DL-GEN-001 (Image Generation / GANs)
    # =====================================================
    @Rule(Problem(task_type="generation"), Dataset(input_modality="image"))
    def generation_gan(self):
        self.declare(ScoreEvidence(
            arch="GAN (DCGAN)", points=100, 
            reason="Goodfellow Ch20: High-dimensional image generation requires an adversarial setup (Generator vs Discriminator).", source_rule="DL-GEN-001"
        ))

    # =====================================================
    # DL-TS-001 (Time Series Forecasting)
    # =====================================================
    @Rule(Problem(task_type="time_series_forecasting"), Dataset(input_modality="tabular"))
    def timeseries_lstm(self):
        self.declare(ScoreEvidence(
            arch="LSTM/GRU", points=100, 
            reason="Goodfellow Ch10: Capturing temporal dynamics in historical numerical data requires gated recurrent units.", source_rule="DL-TS-001"
        ))