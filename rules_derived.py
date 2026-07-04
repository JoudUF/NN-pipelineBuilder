from experta import *

from facts import *


class DerivedRules(KnowledgeEngine):

    # =====================================================
    # DR_01 : High Overfitting Risk
    # =====================================================

    @Rule(
        Dataset(examples_per_class=MATCH.n),
        TEST(lambda n: n < 5000)
    )
    def derive_high_overfitting_risk(self):
        self.declare(
            Derived(
                overfitting_risk="high"
            )
        )

    # =====================================================
    # DR_02 : Medium Overfitting Risk
    # =====================================================

    @Rule(
        Dataset(examples_per_class=MATCH.n),
        TEST(lambda n: 5000 <= n < 50000)
    )
    def derive_medium_overfitting_risk(self):
        self.declare(
            Derived(
                overfitting_risk="medium"
            )
        )

    # =====================================================
    # DR_03 : Low Overfitting Risk
    # =====================================================

    @Rule(
        Dataset(examples_per_class=MATCH.n),
        TEST(lambda n: n >= 50000)
    )
    def derive_low_overfitting_risk(self):
        self.declare(
            Derived(
                overfitting_risk="low"
            )
        )

    # =====================================================
    # DR_04 : Sequence Problem
    # =====================================================

    @Rule(
        Dataset(temporal_structure=True)
    )
    def derive_sequence_problem(self):
        self.declare(
            Derived(
                sequence_problem=True
            )
        )

    # =====================================================
    # DR_05 : Long Sequence
    # =====================================================

    @Rule(
        Dataset(sequence_length_type="long")
    )
    def derive_long_sequence(self):
        self.declare(
            Derived(
                long_sequence=True
            )
        )

    # =====================================================
    # DR_06 : Variable Length Sequence
    # =====================================================

    @Rule(
        Dataset(sequence_length_type="variable")
    )
    def derive_variable_sequence(self):
        self.declare(
            Derived(
                variable_length_sequence=True
            )
        )

    # =====================================================
    # DR_07 : Image-Like Data
    # =====================================================

    @Rule(
        Dataset(spatial_structure="2D_grid")
    )
    def derive_image_data(self):
        self.declare(
            Derived(
                image_like_data=True
            )
        )

    # =====================================================
    # DR_08 : Volumetric Data
    # =====================================================

    @Rule(
        Dataset(spatial_structure="3D_grid")
    )
    def derive_volumetric_data(self):
        self.declare(
            Derived(
                volumetric_data=True
            )
        )

    # =====================================================
    # DR_09 : Strict Latency Requirement
    # =====================================================

    @Rule(
        Constraint(
            inference_latency_budget=MATCH.latency
        ),
        TEST(lambda latency: latency < 50)
    )
    def derive_strict_latency(self):
        self.declare(
            Derived(
                strict_latency=True
            )
        )

    # =====================================================
    # DR_10 : High Resource Constraint
    # =====================================================

    @Rule(
        Constraint(
            deployment_device="edge_device_mobile"
        )
    )
    def derive_resource_constraint(self):
        self.declare(
            Derived(
                resource_constraint="high"
            )
        )

    # =====================================================
    # DR_11 : Severe Class Imbalance
    # =====================================================

    @Rule(
        Dataset(
            class_imbalance_ratio=MATCH.r
        ),
        TEST(lambda r: r < 0.1)
    )
    def derive_class_imbalance(self):
        self.declare(
            Derived(
                severe_class_imbalance=True
            )
        )

    # =====================================================
    # DR_12 : Unlabeled Dataset
    # =====================================================

    @Rule(
        Dataset(
            label_source_type="none"
        )
    )
    def derive_unlabeled_dataset(self):
        self.declare(
            Derived(
                unlabeled_dataset=True
            )
        )
    
     # =====================================================
    # DR_07 : Multimodal Problem
    # =====================================================

    @Rule(
        Feature(
            is_multimodal=True
        )
    )
    def derive_multimodal_problem(self):
        self.declare(
            Derived(
                multimodal_problem=True
            )
        )


      # =====================================================
    # DR_08 : Resource Constraint (VRAM)
    # =====================================================

    @Rule(
        Constraint(
            vram_budget=MATCH.v
        ),
        TEST(lambda v: v < 8)
    )
    def derive_resource_constraint_vram(self):
        self.declare(
            Derived(
                resource_constraint="high"
            )
        )


    # =====================================================
    # DR_09 : Strict Latency (duplicate guard removed)
    # =====================================================

    # NOTE: This rule is identical to DR_09 above (line 127).
    # Removed duplicate to avoid Python method-name collision.

     # =====================================================
    # DR_10 : Deep Network Requirement
    # =====================================================

    @Rule(
        Derived(
            large_dataset=True
        )
    )
    def derive_deep_network(self):
        self.declare(
            Derived(
                deep_network=True
            )
        )
