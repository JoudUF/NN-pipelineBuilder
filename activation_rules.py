from experta import *
from facts import *

class ActivationRules(KnowledgeEngine):

    # Hidden Activation for MLP and CNN (Works for ALL tasks)
    @Rule(
        Problem(task_type=W()),
        SelectedArchitecture(name=MATCH.n),
        TEST(lambda n: n in ["MLP", "CNN"])
    )
    def relu_activation(self):
        self.declare(Recommendation(
            category="HiddenActivation", 
            value="ReLU (Rectified Linear Units) to maintain large and predictable gradients.", 
            source_rule="Goodfellow Ch6 (Sec 6.3)"
        ))

    # Hidden Activation for Sequence Models (Works for ALL tasks)
    @Rule(
        Problem(task_type=W()),
        SelectedArchitecture(name="LSTM/GRU")
    )
    def rnn_activation(self):
        self.declare(Recommendation(
            category="HiddenActivation", 
            value="Tanh for state units, modulated by Sigmoid gating units (input/output/forget).", 
            source_rule="Goodfellow Ch10"
        ))