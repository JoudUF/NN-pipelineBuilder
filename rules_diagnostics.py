from experta import *
from facts import *

class DiagnosticRules(KnowledgeEngine):

    # =====================================================
    # DL-DIAG-001: UNDERFITTING (High Training Error)
    # Ref: Goodfellow Ch11 (Sec 11.4.1)
    # =====================================================
    @Rule(ModelStatus(training_error="High"))
    def diagnose_underfitting(self):
        self.declare(Recommendation(
            category="Diagnostics", 
            value="🚨 UNDERFITTING DETECTED: The model lacks the capacity to solve the task.\n"
                  "   ➤ ACTION 1: Increase Model Capacity (Add more hidden layers or more units per layer).\n"
                  "   ➤ ACTION 2: Decrease Regularization (Reduce Dropout rates or L2 Weight Decay).\n"
                  "   ➤ ACTION 3: Check Optimization (Ensure learning rate is not too small, or switch to Adam).", 
            source_rule="Goodfellow_Sec11.4.1"
        ))

    # =====================================================
    # DL-DIAG-002: OVERFITTING (Low Train Error, High Test Error)
    # Ref: Goodfellow Ch11 (Sec 11.4.2)
    # =====================================================
    @Rule(ModelStatus(training_error="Low", test_error="High"))
    def diagnose_overfitting(self):
        self.declare(Recommendation(
            category="Diagnostics", 
            value="⚠️ OVERFITTING DETECTED: The model memorized the training data but fails to generalize.\n"
                  "   ➤ ACTION 1: Gather more data or heavily apply Data Augmentation.\n"
                  "   ➤ ACTION 2: Increase Regularization (Increase Dropout, use higher L2 penalty).\n"
                  "   ➤ ACTION 3: Decrease Model Capacity (Remove layers or reduce neurons).", 
            source_rule="Goodfellow_Sec11.4.2"
        ))

    # =====================================================
    # DL-DIAG-003: OPTIMAL FIT (Low Train Error, Low Test Error)
    # Ref: Goodfellow Ch11
    # =====================================================
    @Rule(ModelStatus(training_error="Low", test_error="Low"))
    def diagnose_optimal(self):
        self.declare(Recommendation(
            category="Diagnostics", 
            value="✅ OPTIMAL FIT: The model is fitting and generalizing beautifully.\n"
                  "   ➤ ACTION: You are ready for deployment. You may perform minor hyperparameter grid-search for extra % gains.", 
            source_rule="Goodfellow_Ch11_Success"
        ))