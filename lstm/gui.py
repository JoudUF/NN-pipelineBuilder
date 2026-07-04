import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText

from lstm.engine import LSTMPipelineEngine
from lstm.layer_config import (
    LAYER_CATEGORIES, LAYER_DISPLAY_NAMES, LAYER_PARAMS,
    ACTIVATION_TYPES, NO_PARAM_LAYERS,
)


# =====================================================
# Color Palette
# =====================================================
BG_DARK       = "#1a1a2e"
BG_PANEL      = "#16213e"
BG_CARD       = "#0f3460"
BG_INPUT      = "#1a1a2e"
FG_PRIMARY    = "#e0e0e0"
FG_ACCENT     = "#00d4ff"
FG_SUCCESS    = "#00e676"
FG_WARNING    = "#ffab00"
FG_ERROR      = "#ff5252"
FG_DIM        = "#7a7a9e"
BORDER_COLOR  = "#2a2a4a"
BTN_ADD       = "#00897b"
BTN_REMOVE    = "#c62828"
BTN_GENERATE  = "#1565c0"
BTN_FG        = "#ffffff"


def launch_lstm_builder():
    """
    Opens the LSTM Pipeline Builder as a new top-level window.
    Called from the main expert system GUI.
    """
    window = tk.Toplevel()
    window.title("LSTM Pipeline Builder — Expert System")
    window.geometry("960x860")
    window.configure(bg=BG_DARK)
    window.minsize(800, 700)

    engine = LSTMPipelineEngine()

    # State tracking for the GUI
    state = {
        "config_set": False,
        "param_widgets": [],
        "param_vars": {},
    }

    # ==========================================
    # STYLE CONFIGURATION
    # ==========================================
    style = ttk.Style(window)
    style.theme_use("clam")

    style.configure("Dark.TFrame", background=BG_DARK)
    style.configure("Panel.TFrame", background=BG_PANEL)
    style.configure("Card.TFrame", background=BG_CARD)

    style.configure("Dark.TLabel", background=BG_DARK, foreground=FG_PRIMARY,
                     font=("Consolas", 10))
    style.configure("Panel.TLabel", background=BG_PANEL, foreground=FG_PRIMARY,
                     font=("Consolas", 10))
    style.configure("Header.TLabel", background=BG_DARK, foreground=FG_ACCENT,
                     font=("Consolas", 13, "bold"))
    style.configure("SubHeader.TLabel", background=BG_PANEL, foreground=FG_ACCENT,
                     font=("Consolas", 11, "bold"))
    style.configure("Shape.TLabel", background=BG_CARD, foreground=FG_SUCCESS,
                     font=("Consolas", 9))
    style.configure("Dim.TLabel", background=BG_CARD, foreground=FG_DIM,
                     font=("Consolas", 9))

    style.configure("Add.TButton", font=("Consolas", 10, "bold"))
    style.configure("Remove.TButton", font=("Consolas", 10, "bold"))
    style.configure("Generate.TButton", font=("Consolas", 11, "bold"))

    # ==========================================
    # MAIN LAYOUT — 3 vertical sections
    # ==========================================
    main_frame = ttk.Frame(window, style="Dark.TFrame")
    main_frame.pack(fill="both", expand=True, padx=10, pady=5)

    # --- SECTION 1: Global Config ---
    config_frame = ttk.Frame(main_frame, style="Panel.TFrame")
    config_frame.pack(fill="x", padx=5, pady=(5, 2))

    ttk.Label(config_frame, text="⚙ GLOBAL CONFIGURATION",
              style="SubHeader.TLabel").grid(row=0, column=0, columnspan=6,
                                              padx=10, pady=(8, 5), sticky="w")

    # Row 1: Batch Size, Seq Length
    ttk.Label(config_frame, text="Batch Size:", style="Panel.TLabel").grid(
        row=1, column=0, padx=(10, 3), pady=5, sticky="w")
    var_batch = tk.StringVar(value="32")
    ent_batch = tk.Entry(config_frame, textvariable=var_batch, width=8,
                          bg=BG_INPUT, fg=FG_PRIMARY, insertbackground=FG_PRIMARY,
                          font=("Consolas", 10), relief="flat")
    ent_batch.grid(row=1, column=1, padx=3, pady=5, sticky="w")

    ttk.Label(config_frame, text="Seq Length:", style="Panel.TLabel").grid(
        row=1, column=2, padx=(15, 3), pady=5, sticky="w")
    var_seq = tk.StringVar(value="50")
    ent_seq = tk.Entry(config_frame, textvariable=var_seq, width=8,
                        bg=BG_INPUT, fg=FG_PRIMARY, insertbackground=FG_PRIMARY,
                        font=("Consolas", 10), relief="flat")
    ent_seq.grid(row=1, column=3, padx=3, pady=5, sticky="w")

    # Row 2: Input Type, Vocab / Feature Size
    ttk.Label(config_frame, text="Input Type:", style="Panel.TLabel").grid(
        row=2, column=0, padx=(10, 3), pady=5, sticky="w")
    var_input_type = tk.StringVar(value="text")
    combo_input = ttk.Combobox(config_frame, textvariable=var_input_type,
                                values=["text", "continuous"], state="readonly",
                                width=12, font=("Consolas", 10))
    combo_input.grid(row=2, column=1, padx=3, pady=5, sticky="w")

    lbl_extra = ttk.Label(config_frame, text="Vocab Size:", style="Panel.TLabel")
    lbl_extra.grid(row=2, column=2, padx=(15, 3), pady=5, sticky="w")
    var_extra = tk.StringVar(value="10000")
    ent_extra = tk.Entry(config_frame, textvariable=var_extra, width=10,
                          bg=BG_INPUT, fg=FG_PRIMARY, insertbackground=FG_PRIMARY,
                          font=("Consolas", 10), relief="flat")
    ent_extra.grid(row=2, column=3, padx=3, pady=5, sticky="w")

    def on_input_type_change(*_):
        if var_input_type.get() == "text":
            lbl_extra.config(text="Vocab Size:")
            var_extra.set("10000")
        else:
            lbl_extra.config(text="Feature Size:")
            var_extra.set("16")

    var_input_type.trace_add("write", on_input_type_change)

    # Set Config Button
    def on_set_config():
        try:
            batch = int(var_batch.get())
            seq = int(var_seq.get())
            extra = int(var_extra.get())
        except ValueError:
            messagebox.showerror("Invalid Input",
                                  "Batch Size, Seq Length, and Vocab/Feature Size must be integers.",
                                  parent=window)
            return

        itype = var_input_type.get()
        if itype == "text":
            engine.set_pipeline_config(batch, seq, itype, vocab_size=extra)
        else:
            engine.set_pipeline_config(batch, seq, itype, input_feature_size=extra)

        state["config_set"] = True
        lbl_config_status.config(text="✓ Config set", foreground=FG_SUCCESS)

    btn_config = tk.Button(config_frame, text="Set Config", command=on_set_config,
                            bg=BTN_ADD, fg=BTN_FG, font=("Consolas", 10, "bold"),
                            relief="flat", padx=12, pady=2, cursor="hand2")
    btn_config.grid(row=1, column=4, rowspan=2, padx=(20, 10), pady=5)

    lbl_config_status = ttk.Label(config_frame, text="", style="Panel.TLabel")
    lbl_config_status.grid(row=1, column=5, rowspan=2, padx=5, pady=5, sticky="w")

    # ==========================================
    # SECTION 2: Pipeline Display + Warnings
    # ==========================================
    middle_frame = ttk.Frame(main_frame, style="Dark.TFrame")
    middle_frame.pack(fill="both", expand=True, padx=5, pady=2)

    # Left: Pipeline
    pipeline_outer = ttk.Frame(middle_frame, style="Panel.TFrame")
    pipeline_outer.pack(side="left", fill="both", expand=True, padx=(0, 3))

    ttk.Label(pipeline_outer, text="📋 PIPELINE",
              style="SubHeader.TLabel").pack(padx=10, pady=(8, 3), anchor="w")

    pipeline_canvas = tk.Canvas(pipeline_outer, bg=BG_PANEL, highlightthickness=0)
    pipeline_scrollbar = ttk.Scrollbar(pipeline_outer, orient="vertical",
                                        command=pipeline_canvas.yview)
    pipeline_scroll_frame = ttk.Frame(pipeline_canvas, style="Panel.TFrame")

    pipeline_scroll_frame.bind(
        "<Configure>",
        lambda e: pipeline_canvas.configure(scrollregion=pipeline_canvas.bbox("all"))
    )
    pipeline_canvas.create_window((0, 0), window=pipeline_scroll_frame, anchor="nw")
    pipeline_canvas.configure(yscrollcommand=pipeline_scrollbar.set)

    pipeline_canvas.pack(side="left", fill="both", expand=True, padx=5, pady=5)
    pipeline_scrollbar.pack(side="right", fill="y")

    # Enable mousewheel scrolling
    def _on_mousewheel(event):
        pipeline_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    pipeline_canvas.bind_all("<MouseWheel>", _on_mousewheel)

    # Right: Warnings
    warnings_outer = ttk.Frame(middle_frame, style="Panel.TFrame")
    warnings_outer.pack(side="right", fill="both", expand=False, padx=(3, 0))
    warnings_outer.configure(width=280)
    warnings_outer.pack_propagate(False)

    ttk.Label(warnings_outer, text="⚠ WARNINGS",
              style="SubHeader.TLabel").pack(padx=10, pady=(8, 3), anchor="w")

    warnings_text = ScrolledText(warnings_outer, height=10, wrap="word",
                                  font=("Consolas", 9), bg=BG_CARD, fg=FG_WARNING,
                                  relief="flat", insertbackground=FG_WARNING,
                                  state="disabled")
    warnings_text.pack(fill="both", expand=True, padx=5, pady=5)

    # ==========================================
    # SECTION 3: Add Layer Controls
    # ==========================================
    add_frame = ttk.Frame(main_frame, style="Panel.TFrame")
    add_frame.pack(fill="x", padx=5, pady=(2, 5))

    ttk.Label(add_frame, text="➕ ADD LAYER",
              style="SubHeader.TLabel").grid(row=0, column=0, columnspan=4,
                                              padx=10, pady=(8, 5), sticky="w")

    # Layer type dropdown (categorized flat list)
    ttk.Label(add_frame, text="Layer:", style="Panel.TLabel").grid(
        row=1, column=0, padx=(10, 3), pady=5, sticky="w")

    layer_options = []
    for category, layers in LAYER_CATEGORIES.items():
        for layer_key in layers:
            display = LAYER_DISPLAY_NAMES.get(layer_key, layer_key)
            layer_options.append(display)

    var_layer_type = tk.StringVar(value="")
    combo_layer = ttk.Combobox(add_frame, textvariable=var_layer_type,
                                values=layer_options, state="readonly",
                                width=28, font=("Consolas", 10))
    combo_layer.grid(row=1, column=1, padx=3, pady=5, sticky="w")

    # Reverse lookup: display name -> layer key
    display_to_key = {v: k for k, v in LAYER_DISPLAY_NAMES.items()}

    # Dynamic parameter form
    param_form_frame = ttk.Frame(add_frame, style="Panel.TFrame")
    param_form_frame.grid(row=2, column=0, columnspan=4, padx=10, pady=5, sticky="ew")

    def clear_param_form():
        for widget in state["param_widgets"]:
            widget.destroy()
        state["param_widgets"] = []
        state["param_vars"] = {}

    def build_param_form(layer_key):
        clear_param_form()
        params = LAYER_PARAMS.get(layer_key, [])

        if not params:
            lbl = ttk.Label(param_form_frame,
                             text="  (no parameters needed)",
                             style="Panel.TLabel")
            lbl.grid(row=0, column=0, padx=5, pady=2, sticky="w")
            state["param_widgets"].append(lbl)
            return

        for i, param_def in enumerate(params):
            name = param_def["name"]
            label = param_def["label"]
            ptype = param_def["type"]
            default = param_def.get("default")

            lbl = ttk.Label(param_form_frame, text=label + ":",
                             style="Panel.TLabel")
            lbl.grid(row=i, column=0, padx=(5, 3), pady=3, sticky="w")
            state["param_widgets"].append(lbl)

            if ptype == "bool":
                var = tk.BooleanVar(value=default if default is not None else False)
                chk = tk.Checkbutton(param_form_frame, variable=var,
                                      bg=BG_PANEL, fg=FG_PRIMARY,
                                      selectcolor=BG_INPUT, activebackground=BG_PANEL,
                                      font=("Consolas", 10))
                chk.grid(row=i, column=1, padx=3, pady=3, sticky="w")
                state["param_widgets"].append(chk)
                state["param_vars"][name] = var

            elif ptype == "choice":
                options = param_def.get("options", [])
                var = tk.StringVar(value=default if default else (options[0] if options else ""))
                cmb = ttk.Combobox(param_form_frame, textvariable=var,
                                    values=options, state="readonly",
                                    width=20, font=("Consolas", 10))
                cmb.grid(row=i, column=1, padx=3, pady=3, sticky="w")
                state["param_widgets"].append(cmb)
                state["param_vars"][name] = var

            else:
                default_str = str(default) if default is not None else ""
                var = tk.StringVar(value=default_str)
                ent = tk.Entry(param_form_frame, textvariable=var, width=12,
                                bg=BG_INPUT, fg=FG_PRIMARY, insertbackground=FG_PRIMARY,
                                font=("Consolas", 10), relief="flat")
                ent.grid(row=i, column=1, padx=3, pady=3, sticky="w")
                state["param_widgets"].append(ent)
                state["param_vars"][name] = var

    def on_layer_type_change(*_):
        display_name = var_layer_type.get()
        layer_key = display_to_key.get(display_name, "")
        if layer_key:
            build_param_form(layer_key)

    var_layer_type.trace_add("write", on_layer_type_change)

    # ==========================================
    # Add / Remove / Generate Buttons
    # ==========================================
    btn_row = ttk.Frame(add_frame, style="Panel.TFrame")
    btn_row.grid(row=3, column=0, columnspan=4, padx=10, pady=(5, 8), sticky="ew")

    def on_add_layer():
        if not state["config_set"]:
            messagebox.showwarning("Config Required",
                                    "Please set the Global Configuration first.",
                                    parent=window)
            return

        display_name = var_layer_type.get()
        layer_key = display_to_key.get(display_name, "")
        if not layer_key:
            messagebox.showwarning("No Layer Selected",
                                    "Please select a layer type from the dropdown.",
                                    parent=window)
            return

        # Collect user params
        user_params = {}
        params_def = LAYER_PARAMS.get(layer_key, [])
        for param_def in params_def:
            name = param_def["name"]
            ptype = param_def["type"]
            var = state["param_vars"].get(name)
            if var is None:
                continue

            raw = var.get()

            if ptype == "int":
                if raw == "" or raw is None:
                    messagebox.showwarning("Missing Parameter",
                                            "Please provide: " + param_def["label"],
                                            parent=window)
                    return
                try:
                    user_params[name] = int(raw)
                except ValueError:
                    messagebox.showerror("Invalid Input",
                                          param_def["label"] + " must be an integer.",
                                          parent=window)
                    return
            elif ptype == "float":
                try:
                    user_params[name] = float(raw)
                except ValueError:
                    messagebox.showerror("Invalid Input",
                                          param_def["label"] + " must be a number.",
                                          parent=window)
                    return
            elif ptype == "bool":
                user_params[name] = bool(raw)
            elif ptype == "choice":
                user_params[name] = str(raw)

        # Add layer through the engine
        result = engine.add_layer(layer_key, user_params)

        # Update the pipeline display
        refresh_pipeline_display()

        # Update warnings
        refresh_warnings()

    def on_remove_last():
        if engine.get_layer_count() == 0:
            messagebox.showinfo("Empty Pipeline",
                                 "No layers to remove.",
                                 parent=window)
            return
        engine.remove_last_layer()
        refresh_pipeline_display()
        refresh_warnings()

    def on_generate_code():
        if engine.get_layer_count() == 0:
            messagebox.showinfo("Empty Pipeline",
                                 "Add some layers first.",
                                 parent=window)
            return

        code = engine.generate_full_code()
        show_code_window(code)

    btn_add = tk.Button(btn_row, text="  ➕ Add Layer  ", command=on_add_layer,
                         bg=BTN_ADD, fg=BTN_FG, font=("Consolas", 10, "bold"),
                         relief="flat", padx=10, pady=4, cursor="hand2")
    btn_add.pack(side="left", padx=(0, 8))

    btn_remove = tk.Button(btn_row, text="  🗑 Remove Last  ", command=on_remove_last,
                            bg=BTN_REMOVE, fg=BTN_FG, font=("Consolas", 10, "bold"),
                            relief="flat", padx=10, pady=4, cursor="hand2")
    btn_remove.pack(side="left", padx=(0, 8))

    btn_generate = tk.Button(btn_row, text="  📋 Generate Code  ", command=on_generate_code,
                              bg=BTN_GENERATE, fg=BTN_FG, font=("Consolas", 11, "bold"),
                              relief="flat", padx=15, pady=4, cursor="hand2")
    btn_generate.pack(side="right")

    # ==========================================
    # Refresh Functions
    # ==========================================
    def refresh_pipeline_display():
        """Rebuild the pipeline visual from current engine state."""
        for widget in pipeline_scroll_frame.winfo_children():
            widget.destroy()

        nodes = engine.layer_nodes
        shapes = engine.get_all_output_shapes()
        shape_map = {s.get("layer_index"): s for s in shapes}

        if not nodes:
            lbl = ttk.Label(pipeline_scroll_frame,
                             text="  Pipeline is empty. Add a layer to begin.",
                             style="Dim.TLabel")
            lbl.pack(padx=10, pady=20, anchor="w")
            return

        for node in nodes:
            idx = node.get("index")
            ltype = node.get("layer_type")

            # For activation layers, get the actual activation type from params
            actual_type = ltype
            if ltype == "activation":
                for p in engine.layer_params:
                    if p.get("layer_index") == idx and hasattr(p, '__class__') and p.__class__.__name__ == "ActivationParam":
                        actual_type = p.get("activation_type", "activation")
                        break

            display = LAYER_DISPLAY_NAMES.get(actual_type, actual_type)

            # Find matching param fact
            param_str = _get_param_summary(engine, idx, actual_type)

            # Card frame for this layer
            card = tk.Frame(pipeline_scroll_frame, bg=BG_CARD,
                             highlightbackground=BORDER_COLOR, highlightthickness=1)
            card.pack(fill="x", padx=8, pady=3)

            # Layer header
            header_text = "#{} {}".format(idx, display)
            if param_str:
                header_text += "({})".format(param_str)

            lbl_header = tk.Label(card, text=header_text,
                                   bg=BG_CARD, fg=FG_ACCENT,
                                   font=("Consolas", 10, "bold"),
                                   anchor="w")
            lbl_header.pack(fill="x", padx=8, pady=(5, 1))

            # Output shape
            shape = shape_map.get(idx)
            if shape is not None:
                dims = shape.get("dims", 0)
                fmt = shape.get("format", "")
                if dims == 3:
                    shape_text = "Output: [{}, {}, {}]  {}".format(
                        shape.get("d0"), shape.get("d1"), shape.get("d2"), fmt)
                else:
                    shape_text = "Output: [{}, {}]  {}".format(
                        shape.get("d0"), shape.get("d1"), fmt)

                lbl_shape = tk.Label(card, text=shape_text,
                                      bg=BG_CARD, fg=FG_SUCCESS,
                                      font=("Consolas", 9), anchor="w")
                lbl_shape.pack(fill="x", padx=8, pady=(0, 1))
            else:
                lbl_shape = tk.Label(card, text="Output: (computing...)",
                                      bg=BG_CARD, fg=FG_DIM,
                                      font=("Consolas", 9), anchor="w")
                lbl_shape.pack(fill="x", padx=8, pady=(0, 1))

            # Check for mismatches on this layer
            layer_mismatches = [m for m in engine.get_all_mismatches()
                                 if m.get("layer_index") == idx]
            if layer_mismatches:
                for m in layer_mismatches:
                    sev = m.get("severity", "error")
                    color = FG_ERROR if sev == "error" else FG_WARNING
                    icon = "❌" if sev == "error" else "⚠"
                    lbl_m = tk.Label(card,
                                      text="{} {}".format(icon, m.get("description", "")),
                                      bg=BG_CARD, fg=color,
                                      font=("Consolas", 8), anchor="w",
                                      wraplength=500, justify="left")
                    lbl_m.pack(fill="x", padx=8, pady=(0, 3))

            # Arrow connector (except for last layer)
            if idx < len(nodes) - 1:
                arrow = tk.Label(pipeline_scroll_frame, text="  ↓",
                                  bg=BG_PANEL, fg=FG_DIM,
                                  font=("Consolas", 11))
                arrow.pack(anchor="w", padx=20, pady=0)

    def refresh_warnings():
        """Update the warnings panel."""
        warnings_text.config(state="normal")
        warnings_text.delete("1.0", tk.END)

        all_mismatches = engine.get_all_mismatches()
        all_fixes = engine.fix_recommendations

        if not all_mismatches:
            warnings_text.insert(tk.END, "✓ Pipeline is valid.\nNo mismatches detected.")
            warnings_text.config(foreground=FG_SUCCESS)
        else:
            warnings_text.config(foreground=FG_WARNING)
            for m in all_mismatches:
                idx = m.get("layer_index", "?")
                sev = m.get("severity", "error")
                desc = m.get("description", "")
                mtype = m.get("mismatch_type", "")

                icon = "❌" if sev == "error" else "⚠"
                warnings_text.insert(tk.END,
                    "{} Layer #{} [{}]\n{}\n\n".format(icon, idx, mtype, desc))

            # Show fix recommendations
            for f in all_fixes:
                idx = f.get("layer_index", "?")
                fix = f.get("fix_description", "")
                insert = f.get("insert_layer_type", "")
                warnings_text.insert(tk.END,
                    "💡 Fix for #{}: {}\n   Suggested: {}\n\n".format(idx, fix, insert))

        warnings_text.config(state="disabled")

    def show_code_window(code):
        """Open a new window showing the generated PyTorch code."""
        code_win = tk.Toplevel(window)
        code_win.title("Generated PyTorch Model Code")
        code_win.geometry("700x500")
        code_win.configure(bg=BG_DARK)

        code_text = ScrolledText(code_win, wrap="none",
                                  font=("Consolas", 11), bg="#1e1e1e", fg="#d4d4d4",
                                  insertbackground="#d4d4d4", relief="flat")
        code_text.pack(fill="both", expand=True, padx=10, pady=10)
        code_text.insert(tk.END, code)

        def copy_code():
            code_win.clipboard_clear()
            code_win.clipboard_append(code)
            messagebox.showinfo("Copied", "Code copied to clipboard!", parent=code_win)

        btn_copy = tk.Button(code_win, text="📋 Copy to Clipboard", command=copy_code,
                              bg=BTN_GENERATE, fg=BTN_FG, font=("Consolas", 10, "bold"),
                              relief="flat", padx=15, pady=5, cursor="hand2")
        btn_copy.pack(pady=(0, 10))


def _get_param_summary(engine, layer_index, layer_type):
    """Build a short parameter summary string for display."""
    for param in engine.layer_params:
        if param.get("layer_index") != layer_index:
            continue

        parts = []

        # Embedding
        if layer_type == "embedding":
            parts.append(str(param.get("vocab_size", "?")))
            parts.append(str(param.get("embedding_dim", "?")))

        # Recurrent layers
        elif layer_type in ("lstm", "gru", "rnn"):
            parts.append(str(param.get("hidden_size", "?")))
            nl = param.get("num_layers", 1)
            if nl > 1:
                parts.append("layers={}".format(nl))
            if param.get("bidirectional", False):
                parts.append("bidir")

        # Linear
        elif layer_type == "linear":
            parts.append("out={}".format(param.get("out_features", "?")))

        # Conv1d
        elif layer_type == "conv1d":
            parts.append("ch={}".format(param.get("out_channels", "?")))
            parts.append("k={}".format(param.get("kernel_size", "?")))

        # Pooling
        elif layer_type in ("maxpool1d", "avgpool1d"):
            parts.append("k={}".format(param.get("kernel_size", "?")))

        # Adaptive pooling
        elif layer_type in ("adaptive_avgpool1d", "adaptive_maxpool1d"):
            parts.append("out={}".format(param.get("output_size", "?")))

        # Attention
        elif layer_type == "multihead_attention":
            parts.append("heads={}".format(param.get("num_heads", "?")))

        # Dropout
        elif layer_type == "dropout":
            parts.append("p={}".format(param.get("p", "?")))

        # Permute
        elif layer_type == "permute":
            parts.append(str(param.get("target_format", "?")))

        # Activations
        elif layer_type in ACTIVATION_TYPES:
            at = param.get("activation_type", layer_type)
            if at == "leaky_relu":
                parts.append("slope={}".format(param.get("negative_slope", 0.01)))

        return ", ".join(parts) if parts else ""

    return ""
