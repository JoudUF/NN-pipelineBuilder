from lstm.facts import (
    PipelineConfig, LayerNode, LayerOutputShape, LayerExpectedInput,
    EmbeddingParam, LSTMParam, GRUParam, RNNParam, LinearParam,
    Conv1dParam, Pool1dParam, AdaptivePool1dParam, AttentionParam,
    DropoutParam, ActivationParam, PermuteParam, SqueezeParam,
    UnsqueezeParam, CatParam, ResidualParam,
    Mismatch, FixRecommendation, LayerCode,
)
from lstm.layer_config import ACTIVATION_TYPES, NO_PARAM_LAYERS


# =====================================================
# Mapping: layer_type -> Param Fact class
# =====================================================

PARAM_FACT_MAP = {
    "embedding":          EmbeddingParam,
    "lstm":               LSTMParam,
    "gru":                GRUParam,
    "rnn":                RNNParam,
    "linear":             LinearParam,
    "conv1d":             Conv1dParam,
    "maxpool1d":          Pool1dParam,
    "avgpool1d":          Pool1dParam,
    "adaptive_avgpool1d": AdaptivePool1dParam,
    "adaptive_maxpool1d": AdaptivePool1dParam,
    "multihead_attention": AttentionParam,
    "dropout":            DropoutParam,
    "permute":            PermuteParam,
    "squeeze":            SqueezeParam,
    "unsqueeze":          UnsqueezeParam,
    "residual_add":       ResidualParam,
    "cat":                CatParam,
}


class LSTMPipelineEngine:
    """
    Orchestrates the expert system pipeline.
    Holds all declared facts (pipeline config, layer nodes, params,
    computed shapes) and runs the rule engines in sequence.

    The GUI interacts with this class to add layers, run validation,
    and retrieve results.
    """

    def __init__(self):
        self.pipeline_config = None
        self.layer_nodes = []
        self.layer_params = []
        self.layer_output_shapes = []
        self.mismatches = []
        self.fix_recommendations = []
        self.layer_codes = []

    def set_pipeline_config(self, batch_size, seq_length, input_type,
                            vocab_size=None, input_feature_size=None):
        """Store the global pipeline configuration."""
        config_args = {
            "batch_size": batch_size,
            "seq_length": seq_length,
            "input_type": input_type,
        }
        if input_type == "text" and vocab_size is not None:
            config_args["vocab_size"] = vocab_size
        if input_type == "continuous" and input_feature_size is not None:
            config_args["input_feature_size"] = input_feature_size

        self.pipeline_config = PipelineConfig(**config_args)

    def add_layer(self, layer_type, user_params):
        """
        Add a new layer to the pipeline.

        Args:
            layer_type:  str — the layer type key (e.g. "lstm", "embedding")
            user_params: dict — the user-provided parameters for that layer

        Returns:
            dict with keys:
                "index":      int — the new layer's index
                "mismatches": list of Mismatch facts (empty if valid)
                "fixes":      list of FixRecommendation facts
                "output_shape": LayerOutputShape fact or None
                "code":       LayerCode fact or None
        """
        index = len(self.layer_nodes)

        # Activation layers use "activation" as the node type;
        # the specific type (relu, sigmoid, etc.) lives in ActivationParam
        node_type = "activation" if layer_type in ACTIVATION_TYPES else layer_type
        node = LayerNode(index=index, layer_type=node_type)
        self.layer_nodes.append(node)

        # 2. Create the param fact for this layer type
        param_fact = self._create_param_fact(layer_type, index, user_params)
        if param_fact is not None:
            self.layer_params.append(param_fact)

        # 3. Run the full engine pipeline
        result = self._run_engines()

        return result

    def remove_last_layer(self):
        """Remove the last layer and re-run validation."""
        if not self.layer_nodes:
            return None

        removed_index = len(self.layer_nodes) - 1
        self.layer_nodes.pop()

        # Remove the param fact for the removed layer
        self.layer_params = [
            p for p in self.layer_params
            if p.get("layer_index") != removed_index
        ]

        # Re-run engines
        result = self._run_engines()
        return result

    def get_layer_count(self):
        """Return the number of layers in the pipeline."""
        return len(self.layer_nodes)

    def get_all_output_shapes(self):
        """Return all computed output shapes."""
        return list(self.layer_output_shapes)

    def get_all_mismatches(self):
        """Return all current mismatches."""
        return list(self.mismatches)

    def get_all_codes(self):
        """Return all generated code facts, sorted by index."""
        sorted_codes = sorted(self.layer_codes, key=lambda c: c.get("layer_index", 0))
        return sorted_codes

    def _create_param_fact(self, layer_type, index, user_params):
        """Create the appropriate param fact for the given layer type."""
        # Activations use ActivationParam
        if layer_type in ACTIVATION_TYPES:
            fact_args = {"layer_index": index, "activation_type": layer_type}
            if layer_type == "leaky_relu":
                fact_args["negative_slope"] = user_params.get("negative_slope", 0.01)
            if layer_type in ("softmax", "log_softmax"):
                fact_args["dim"] = user_params.get("dim", -1)
            return ActivationParam(**fact_args)

        # No-param layers get no param fact
        if layer_type in NO_PARAM_LAYERS:
            return None

        # All other layers use their specific param fact class
        fact_class = PARAM_FACT_MAP.get(layer_type)
        if fact_class is None:
            return None

        fact_args = {"layer_index": index}
        fact_args.update(user_params)
        return fact_class(**fact_args)

    def _run_engines(self):
        """
        Run the full rule engine pipeline:
        1. Shape calculation rules (compute output shapes)
        2. Pruning rules (detect mismatches — needs shapes for PR3/PR4)
        3. Code generation rules (generate code snippets)

        Returns a result dict.
        """
        all_facts = self._collect_all_facts()

        # --- Phase A: Run shape calculation engine FIRST ---
        # (PR3 format and PR4 size rules need computed shapes)
        self.layer_output_shapes = []
        try:
            from lstm.rules_shape import ShapeRules
            shape_engine = ShapeRules()
            shape_engine.reset()
            for f in all_facts:
                shape_engine.declare(f)
            shape_engine.run()

            for f in shape_engine.facts.values():
                if isinstance(f, LayerOutputShape):
                    self.layer_output_shapes.append(f)
        except ImportError:
            # rules_shape.py not yet created
            pass

        # --- Phase B: Run pruning engine (with shapes available) ---
        self.mismatches = []
        self.fix_recommendations = []
        try:
            from lstm.rules_pruning import PruningRules
            pruning_engine = PruningRules()
            pruning_engine.reset()
            # Feed both the original facts AND computed shapes
            pruning_facts = all_facts + list(self.layer_output_shapes)
            for f in pruning_facts:
                pruning_engine.declare(f)
            pruning_engine.run()

            for f in pruning_engine.facts.values():
                if isinstance(f, Mismatch):
                    self.mismatches.append(f)
                elif isinstance(f, FixRecommendation):
                    self.fix_recommendations.append(f)
        except ImportError:
            # rules_pruning.py not yet created
            pass

        # --- Phase C: Run code generation engine ---
        self.layer_codes = []
        try:
            from lstm.rules_code import CodeRules
            code_engine = CodeRules()
            code_engine.reset()
            # Code rules also need output shapes for auto-calculated params
            code_facts = all_facts + list(self.layer_output_shapes)
            for f in code_facts:
                code_engine.declare(f)
            code_engine.run()

            for f in code_engine.facts.values():
                if isinstance(f, LayerCode):
                    self.layer_codes.append(f)
        except ImportError:
            # rules_code.py not yet created
            pass

        # --- Build result ---
        latest_index = len(self.layer_nodes) - 1
        latest_shape = None
        for s in self.layer_output_shapes:
            if s.get("layer_index") == latest_index:
                latest_shape = s
                break

        latest_mismatches = [
            m for m in self.mismatches
            if m.get("layer_index") == latest_index
        ]
        latest_fixes = [
            f for f in self.fix_recommendations
            if f.get("layer_index") == latest_index
        ]

        return {
            "index": latest_index,
            "mismatches": latest_mismatches,
            "fixes": latest_fixes,
            "output_shape": latest_shape,
            "all_shapes": list(self.layer_output_shapes),
            "all_mismatches": list(self.mismatches),
        }

    def _collect_all_facts(self):
        """Collect all currently declared facts into a flat list."""
        facts = []

        if self.pipeline_config is not None:
            facts.append(self.pipeline_config)

        facts.extend(self.layer_nodes)
        facts.extend(self.layer_params)

        return facts

    def generate_full_code(self):
        """
        Assemble all LayerCode facts into a complete nn.Module class.

        Returns:
            str — the full PyTorch model code
        """
        sorted_codes = self.get_all_codes()
        if not sorted_codes:
            return "# No layers added yet."

        init_lines = []
        forward_lines = []

        for code_fact in sorted_codes:
            init_code = code_fact.get("init_code", "")
            forward_code = code_fact.get("forward_code", "")
            if init_code:
                init_lines.append("        " + init_code)
            if forward_code:
                forward_lines.append("        " + forward_code)

        code = "import torch\n"
        code += "import torch.nn as nn\n"
        code += "from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence\n"
        code += "\n\n"
        code += "class LSTMModel(nn.Module):\n"
        code += "    def __init__(self):\n"
        code += "        super(LSTMModel, self).__init__()\n"
        if init_lines:
            code += "\n".join(init_lines) + "\n"
        else:
            code += "        pass\n"
        code += "\n"
        code += "    def forward(self, x):\n"
        if forward_lines:
            code += "\n".join(forward_lines) + "\n"
        else:
            code += "        pass\n"
        code += "        return x\n"

        return code
