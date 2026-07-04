from experta import *
from facts import *

class OutputRules(KnowledgeEngine):

    # =====================================================
    # 1. BINARY CLASSIFICATION OUTPUT
    # =====================================================
    @Rule(Problem(task_type="binary_classification"), SelectedArchitecture(name=W()))
    def binary_output_base(self):
        self.declare(Recommendation(category="OutputNeurons", value="Exactly 1 output unit representing the probability of class 1.", source_rule="Goodfellow Ch6 (Sec 6.2.2.2)"))
        self.declare(Recommendation(category="OutputActivation", value="Logistic Sigmoid function to parameterize a Bernoulli distribution.", source_rule="Goodfellow Ch6 (Sec 6.2.2.2)"))

    @Rule(Problem(task_type="binary_classification"), NOT(Derived(severe_class_imbalance=True)))
    def binary_loss_normal(self):
        self.declare(Recommendation(category="LossFunction", value="Negative Log-Likelihood (Binary Cross-Entropy) to avoid gradient saturation.", source_rule="Goodfellow Ch6 (Sec 6.2.2.2)"))

    # =====================================================
    # 2. MULTICLASS CLASSIFICATION OUTPUT 
    # =====================================================
    @Rule(Problem(task_type="multiclass_classification"), SelectedArchitecture(name=W()))
    def multiclass_output_base(self):
        self.declare(Recommendation(category="OutputNeurons", value="Exactly K output units (Where K is the total number of target classes).", source_rule="Goodfellow_Ch6_Sec6.2.2.3"))
        self.declare(Recommendation(category="OutputActivation", value="Softmax function (To ensure the output represents a valid Multinoulli distribution summing to 1).", source_rule="Goodfellow_Ch6_Sec6.2.2.3"))

    @Rule(Problem(task_type="multiclass_classification"), NOT(Derived(severe_class_imbalance=True)))
    def multiclass_loss_normal(self):
        self.declare(Recommendation(category="LossFunction", value="Categorical Cross-Entropy / Negative Log-Likelihood.", source_rule="Goodfellow_Ch6_Sec6.2.2.3"))

    # =====================================================
    # 3. MULTIMODAL REGRESSION (MDN)
    # =====================================================
    @Rule(Problem(task_type="multimodal_regression"), SelectedArchitecture(name=W()))
    def mdn_output_configuration(self):
        self.declare(Recommendation(category="OutputNeurons", value="c(1+2d) neurons [c: mixing coeffs, c*d: means, c*d: variances]", source_rule="Goodfellow_Sec6.2.2.4"))
        self.declare(Recommendation(category="OutputActivation", value="Split: Softmax (Coeffs) + Linear (Means) + Softplus/Exp (Variances)", source_rule="Goodfellow_Sec6.2.2.4"))
        self.declare(Recommendation(category="LossFunction", value="Negative Log-Likelihood of Gaussian Mixture Model (GMM)", source_rule="Goodfellow_Sec6.2.2.4"))

    # =====================================================
    # 4. STANDARD REGRESSION
    # =====================================================
    @Rule(Problem(task_type="regression"), SelectedArchitecture(name=W()))
    def regression_output_base(self):
        self.declare(Recommendation(category="OutputNeurons", value="Exactly D output units (where D is target dimensionality)", source_rule="Goodfellow_Sec6.2.2.1"))
        self.declare(Recommendation(category="OutputActivation", value="Linear (Identity) function (Prevents saturation)", source_rule="Goodfellow_Sec6.2.2.1"))

    @Rule(Problem(task_type="regression"), NOT(Derived(severe_class_imbalance=True)))
    def regression_loss_normal(self):
        self.declare(Recommendation(category="LossFunction", value="Mean Squared Error (MSE) (Derived from Gaussian NLL)", source_rule="Goodfellow_Sec6.2.2.1"))

    # =====================================================
    # 5. TIME SERIES FORECASTING OUTPUT
    # =====================================================
    @Rule(Problem(task_type="time_series_forecasting"), SelectedArchitecture(name="LSTM/GRU"))
    def timeseries_output_base(self):
        self.declare(Recommendation(category="OutputNeurons", value="D x T units (Target Dimensions x Future Steps)", source_rule="Goodfellow_Sec10.2"))
        self.declare(Recommendation(category="OutputActivation", value="Linear (Identity) function", source_rule="Goodfellow_Sec6.2.2"))

    @Rule(Problem(task_type="time_series_forecasting"), NOT(Derived(severe_class_imbalance=True)))
    def timeseries_loss_normal(self):
        self.declare(Recommendation(category="LossFunction", value="Mean Squared Error (MSE)", source_rule="Goodfellow_Sec6.2.2"))

    # =====================================================
    # 6. TRANSLATION OUTPUT
    # =====================================================
    @Rule(Problem(task_type="translation"), SelectedArchitecture(name="Encoder-Decoder RNN"))
    def translation_output(self):
        self.declare(Recommendation(category="OutputNeurons", value="Exactly |V| units (Target Language Vocabulary Size)", source_rule="Goodfellow_Sec10.4"))
        self.declare(Recommendation(category="OutputActivation", value="Softmax function over the vocabulary", source_rule="Goodfellow_Sec10.4"))
        self.declare(Recommendation(category="LossFunction", value="Categorical Cross-Entropy (Negative Log-Likelihood of sequence)", source_rule="Goodfellow_Sec10.4"))

    # =====================================================
    # 7. GENERATION (GAN) OUTPUT
    # =====================================================
    @Rule(Problem(task_type="generation"), SelectedArchitecture(name="GAN (DCGAN)"))
    def gan_output(self):
        self.declare(Recommendation(category="OutputNeurons", value="H x W x C units (Target image dimensions)", source_rule="Goodfellow_Sec20.10.4"))
        self.declare(Recommendation(category="OutputActivation", value="Tanh function (Outputs scaled to [-1, 1])", source_rule="Goodfellow_Sec20.10.4"))
        self.declare(Recommendation(category="LossFunction", value="Minimax Objective (Binary Cross-Entropy for Adversarial Training)", source_rule="Goodfellow_Sec20.10.4"))


    # =====================================================
    # 8. CONSTRAINT: SEVERE CLASS IMBALANCE / RARE EVENTS (LOSS OVERRIDE)
    # =====================================================
    
    # 1. For Classification Tasks (Focal Loss)
    @Rule(
        Derived(severe_class_imbalance=True),
        Problem(task_type=MATCH.t),
        TEST(lambda t: t in ["binary_classification", "multiclass_classification"])
    )
    def handle_imbalance_loss_class(self):
        self.declare(Recommendation(
            category="LossFunction", 
            value="⚠️ URGENT: Use Weighted Cross-Entropy or Focal Loss. Standard loss ignores the minority class.", 
            source_rule="ML_Best_Practices_Imbalance"
        ))

    # 2. For Regression & Time Series Tasks (Asymmetric Loss)
    @Rule(
        Derived(severe_class_imbalance=True),
        Problem(task_type=MATCH.t),
        TEST(lambda t: t in ["regression", "time_series_forecasting"])
    )
    def handle_imbalance_loss_reg(self):
        self.declare(Recommendation(
            category="LossFunction", 
            value="⚠️ URGENT: Use Weighted MSE or Asymmetric Loss (e.g., Log-Cosh) to penalize errors on rare extreme values (Anomalies) more heavily.", 
            source_rule="ML_Best_Practices_Rare_Events"
        ))