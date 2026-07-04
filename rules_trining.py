from experta import *
from facts import *

class TrainingRules(KnowledgeEngine):

    # =====================================================
    # 1. BINARY CLASSIFICATION TRAINING RULES
    # =====================================================
    @Rule(Problem(task_type="binary_classification"), SelectedArchitecture(name="MLP"))
    def mlp_training_binary(self):
        self.declare(Recommendation(category="InputShape", value="Fixed-size 1D vector (n features).", source_rule="DL-BIN-CLASS-001"))
        self.declare(Recommendation(category="HiddenLayers", value="Deep architecture with multiple hidden layers.", source_rule="DL-BIN-CLASS-001"))

    @Rule(Problem(task_type="binary_classification"), SelectedArchitecture(name="CNN"))
    def cnn_training_binary(self):
        self.declare(Recommendation(category="InputShape", value="Multi-dimensional tensor matching grid dimensions (H x W x Channels).", source_rule="DL-BIN-CLASS-002"))
        self.declare(Recommendation(category="HiddenLayers", value="Alternating Convolutional and Pooling layers, terminating with flattening and FC layers.", source_rule="DL-BIN-CLASS-002"))

    @Rule(Problem(task_type="binary_classification"), SelectedArchitecture(name="LSTM/GRU"))
    def rnn_training_binary(self):
        self.declare(Recommendation(category="InputShape", value="Sequence of feature vectors x(1)...x(t).", source_rule="DL-BIN-CLASS-003"))
        self.declare(Recommendation(category="HiddenLayers", value="Deep recurrent state hierarchy, potentially with skip connections.", source_rule="DL-BIN-CLASS-003"))

    # =====================================================
    # 2. MULTICLASS CLASSIFICATION TRAINING RULES
    # =====================================================
    @Rule(Problem(task_type="multiclass_classification"), SelectedArchitecture(name="MLP"))
    def mlp_training_multiclass(self):
        self.declare(Recommendation(category="InputShape", value="Fixed-size 1D vector.", source_rule="DL-MULTI-CLASS-001"))
        self.declare(Recommendation(category="HiddenLayers", value="2 to 4 hidden layers in a 'bottleneck/funnel' topology.", source_rule="DL-MULTI-CLASS-001"))

    @Rule(Problem(task_type="multiclass_classification"), SelectedArchitecture(name="CNN"))
    def cnn_training_multiclass(self):
        self.declare(Recommendation(category="InputShape", value="3D Tensor (Height x Width x Channels).", source_rule="DL-MULTI-CLASS-002"))
        self.declare(Recommendation(category="HiddenLayers", value="Alternating Convolutional & Max Pooling layers, ending with Global Average Pooling.", source_rule="DL-MULTI-CLASS-002"))

    @Rule(Problem(task_type="multiclass_classification"), SelectedArchitecture(name="LSTM/GRU"))
    def rnn_training_multiclass(self):
        self.declare(Recommendation(category="InputShape", value="Sequence of feature vectors.", source_rule="DL-MULTI-CLASS-003"))
        self.declare(Recommendation(category="HiddenLayers", value="Stacked/Deep RNNs (2 to 3 layers).", source_rule="DL-MULTI-CLASS-003"))

    # =====================================================
    # 3. REGRESSION & MDN TRAINING RULES
    # =====================================================
    @Rule(Problem(task_type="multimodal_regression"))
    def train_mdn_network(self):
        self.declare(Recommendation(category="InputShape", value="Fixed-size 1D vector.", source_rule="DL-MDN-001"))
        self.declare(Recommendation(category="HiddenLayers", value="Deep architecture (3 to 5 layers) to capture highly non-linear mapping.", source_rule="DL-MDN-001"))
        # MDN needs specific optimization logic, so we keep it here
        self.declare(Recommendation(category="Optimizer", value="Adam Optimizer with low learning rate (e.g., 1e-4) for stability", source_rule="DL-MDN-001"))
        self.declare(Recommendation(category="Regularization", value="MANDATORY: Gradient Norm Clipping & Variance Clamping", source_rule="DL-MDN-001"))

    @Rule(Problem(task_type="regression"), SelectedArchitecture(name="MLP"))
    def train_regression_network(self):
        self.declare(Recommendation(category="InputShape", value="Fixed-size 1D vector.", source_rule="DL-REG-001"))
        self.declare(Recommendation(category="HiddenLayers", value="2 to 3 hidden layers configured in a funnel topology.", source_rule="DL-REG-001"))

    @Rule(Problem(task_type="regression"), SelectedArchitecture(name="CNN"))
    def train_image_regression(self):
        self.declare(Recommendation(category="InputShape", value="3D Tensor matching grid layout (Height x Width x Channels).", source_rule="DL-REG-IMAGE-001"))
        self.declare(Recommendation(category="HiddenLayers", value="Alternating Convolutional and Pooling layers, terminating with Global Average Pooling.", source_rule="DL-REG-IMAGE-001"))

    # =====================================================
    # 4. UNIVERSAL REGULARIZATION RULES (Chapter 7)
    # =====================================================
    @Rule(SelectedArchitecture(name=W()))
    def universal_early_stopping(self):
        self.declare(Recommendation(category="Regularization", value="Early Stopping (Monitor Validation Loss; halt training when it stagnates to prevent overfitting).", source_rule="Goodfellow_Sec7.8"))

    @Rule(Dataset(input_modality="image"), Dataset(size="small"))
    def dataset_augmentation_images(self):
        self.declare(Recommendation(category="Regularization", value="Dataset Augmentation (Apply random rotations, flips, and crops to artificially expand the dataset and improve generalization).", source_rule="Goodfellow_Sec7.4"))

    @Rule(Dataset(size="small"))
    def severe_overfitting_prevention(self):
        self.declare(Recommendation(category="Regularization", value="Apply Dropout (p=0.5 on hidden layers) as an inexpensive ensemble bagging method.", source_rule="Goodfellow_Sec7.12"))
        self.declare(Recommendation(category="Regularization", value="Apply L2 Parameter Norm Penalty (Weight Decay) to force weights closer to the origin.", source_rule="Goodfellow_Sec7.1"))
        
    # =====================================================
    # 5. OPTIMIZATION ALGORITHMS (Chapter 8)
    # =====================================================
    @Rule(SelectedArchitecture(name="CNN"))
    def optimizer_sgd_momentum(self):
        self.declare(Recommendation(
            category="Optimizer", 
            value="SGD with Momentum (α ≈ 0.9) and a linearly decaying learning rate schedule. Dampens oscillations in high-curvature loss ravines.", 
            source_rule="Goodfellow_Sec8.3.2"
        ))

    @Rule(SelectedArchitecture(name="LSTM/GRU"))
    def optimizer_rmsprop(self):
        self.declare(Recommendation(
            category="Optimizer", 
            value="RMSProp (Decay rate ρ = 0.9) or Adam. Uses an exponentially weighted moving average to handle non-stationary recurrent objectives.", 
            source_rule="Goodfellow_Sec8.5.2"
        ))

    @Rule(SelectedArchitecture(name="MLP"), Problem(task_type=~L("multimodal_regression")))
    def optimizer_adam(self):
        self.declare(Recommendation(
            category="Optimizer", 
            value="Adam Optimizer (lr=0.001, β1=0.9, β2=0.999). Combines Momentum and RMSProp with crucial bias correction for robust general performance.", 
            source_rule="Goodfellow_Sec8.5.3"
        ))
        
        
    # =====================================================
    # 6. PARAMETER INITIALIZATION STRATEGIES (Chapter 8.4)
    # =====================================================
    
    # DL-INIT-001: He Initialization for ReLU (CNN & MLP)
    @Rule(
        SelectedArchitecture(name=MATCH.n),
        TEST(lambda n: n in ["MLP", "CNN"])
    )
    def init_he_relu(self):
        self.declare(Recommendation(
            category="Regularization", 
            value="Weight Initialization: Use 'He Normal' (variance 2/fan_in) for weights to compensate for ReLU's zeroed negative half. Set biases to a small positive constant (e.g., 0.1) to avoid Dead ReLUs.", 
            source_rule="Goodfellow_Sec8.4"
        ))

    # DL-INIT-002: Glorot/Xavier Initialization for Tanh/Sigmoid (LSTM/GRU)
    @Rule(
        SelectedArchitecture(name="LSTM/GRU")
    )
    def init_glorot_rnn(self):
        self.declare(Recommendation(
            category="Regularization", 
            value="Weight Initialization: Use 'Glorot/Xavier Uniform' for weights to maintain gradient variance across recurrent timesteps (crucial for Tanh/Sigmoid). Set biases to 0.", 
            source_rule="Goodfellow_Sec8.4"
        ))
        
        
        
    # =====================================================
    # TRANSLATION TRAINING RULES
    # =====================================================
    @Rule(Problem(task_type="translation"), SelectedArchitecture(name="Encoder-Decoder RNN"))
    def train_translation(self):
        self.declare(Recommendation(category="InputShape", value="Variable-length sequence of word embeddings.", source_rule="DL-TRANS-001"))
        self.declare(Recommendation(category="HiddenLayers", value="Dual networks: Encoder (compresses input to Context vector) and Decoder.", source_rule="DL-TRANS-001"))
        # أبقينا فقط التقنيات الخاصة جداً بالترجمة
        self.declare(Recommendation(category="Regularization", value="Use Beam Search during inference to find the most likely sequence.", source_rule="DL-TRANS-001"))

    # =====================================================
    # GENERATION (GAN) TRAINING RULES
    # =====================================================
    @Rule(Problem(task_type="generation"), SelectedArchitecture(name="GAN (DCGAN)"))
    def train_gan(self):
        self.declare(Recommendation(category="InputShape", value="Dense latent vector 'z' sampled from Gaussian/Uniform prior.", source_rule="DL-GEN-001"))
        self.declare(Recommendation(category="HiddenLayers", value="Fractional-Strided Convolutions for Generator, Strided Convolutions for Discriminator. No Pooling.", source_rule="DL-GEN-001"))
        # أبقينا الـ Optimizer هنا لأن GAN يحتاج إعدادات شاذة جداً (beta1=0.5)
        self.declare(Recommendation(category="Optimizer", value="Adam with momentum beta1 = 0.5 (Crucial for GAN stability to avoid mode collapse).", source_rule="DL-GEN-001"))
        self.declare(Recommendation(category="Regularization", value="Batch Normalization in G and D, One-sided Label Smoothing for D.", source_rule="DL-GEN-001"))

    # =====================================================
    # TIME SERIES FORECASTING TRAINING RULES
    # =====================================================
    @Rule(Problem(task_type="time_series_forecasting"), SelectedArchitecture(name="LSTM/GRU"))
    def train_timeseries(self):
        self.declare(Recommendation(category="InputShape", value="Sliding window tensor: (Batch x Timesteps x Features).", source_rule="DL-TS-001"))
        self.declare(Recommendation(category="HiddenLayers", value="1 to 3 recurrent layers to capture temporal periodicities.", source_rule="DL-TS-001"))
        # تم حذف الـ Optimizer والـ Regularization من هنا، 
        # لأن قواعد الفصل السابع والثامن (Universal Rules) ستلتقط الـ LSTM/GRU وتتولى المهمة باحترافية!
    # =====================================================
    # CONSTRAINT: SEVERE CLASS IMBALANCE / RARE EVENTS (TRAINING)
    # =====================================================
    
    # 1. For Classification Tasks (SMOTE)
    @Rule(
        Derived(severe_class_imbalance=True),
        Problem(task_type=MATCH.t),
        TEST(lambda t: t in ["binary_classification", "multiclass_classification"])
    )
    def handle_imbalance_training_class(self):
        self.declare(Recommendation(
            category="Regularization", 
            value="⚠️ URGENT: Apply Over-sampling (e.g., SMOTE) or Class-Aware Data Augmentation before training to balance batches.", 
            source_rule="ML_Best_Practices_Imbalance"
        ))

    # 2. For Regression & Time Series Tasks (SMOGN)
    @Rule(
        Derived(severe_class_imbalance=True),
        Problem(task_type=MATCH.t),
        TEST(lambda t: t in ["regression", "time_series_forecasting", "multimodal_regression"])
    )
    def handle_imbalance_training_reg(self):
        self.declare(Recommendation(
            category="Regularization", 
            value="⚠️ URGENT: Apply Target-Based Resampling (e.g., SMOGN algorithm) to oversample rare extreme values, or use Anomaly Injection.", 
            source_rule="ML_Best_Practices_Rare_Events"
        ))
        
    # =====================================================
    # CONSTRAINT: EDGE DEPLOYMENT & STRICT LATENCY
    # =====================================================
    @Rule(OR(Derived(strict_latency=True), Constraint(deployment_device="edge")))
    def handle_latency_and_edge(self):
        self.declare(Recommendation(
            category="Diagnostics", 
            value="⚡ DEPLOYMENT CONSTRAINT DETECTED (Edge / Low Latency):\n"
                  "   ➤ ACTION 1: Apply Post-Training Quantization (convert FP32 weights to INT8 or FP16) to reduce model size and increase inference speed.\n"
                  "   ➤ ACTION 2: Consider Knowledge Distillation (train a small, fast 'student' network to mimic a large, accurate 'teacher' network).\n"
                  "   ➤ ACTION 3: If using CNNs, replace standard convolutions with Depthwise Separable Convolutions (e.g., MobileNet architecture).", 
            source_rule="DL_Deployment_Optimization"
        ))