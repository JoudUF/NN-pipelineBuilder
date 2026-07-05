import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText
from lstm.gui import launch_lstm_builder

from facts import (
    Problem, Dataset, Feature, Constraint, Derived,
    CandidateArchitecture, SelectedArchitecture, Recommendation, ScoreEvidence, ModelStatus
)
from rules_derived import DerivedRules
from rules_architecture import ArchitectureRules
from selection_rules import SelectionRules
from activation_rules import ActivationRules
from output_rules import OutputRules
from rules_trining import TrainingRules
from rules_diagnostics import DiagnosticRules

# ==========================================
# Options Dictionaries
# ==========================================
TASK_OPTIONS = ["", "binary_classification", "multiclass_classification", "regression", "multimodal_regression", "translation", "generation", "time_series_forecasting"]
INPUT_OPTIONS = ["", "tabular", "image", "text", "audio", "video"]
SIZE_OPTIONS = ["", "Small (< 10,000)", "Medium (10k - 1M)", "Large (> 1 Million)"]
DEVICE_OPTIONS = ["", "Cloud GPU / TPU", "Edge Device (Mobile/IoT)", "Standard CPU"]
YES_NO_OPTIONS = ["", "Yes", "No"]
ERROR_OPTIONS = ["", "N/A (Not Trained Yet)", "High Error", "Low Error"]

def build_user_facts(fields):
    """Convert UI inputs into Facts for the Engine"""
    facts = []
    
    # 1. Problem Facts
    task = fields["task_type"].get().strip()
    if task: facts.append(Problem(task_type=task))
        
    safety = fields["safety_critical"].get().strip()
    if safety == "Yes": facts.append(Problem(is_human_safety_critical=True))

    # 2. Dataset Facts
    modality = fields["input_modality"].get().strip()
    if modality: facts.append(Dataset(input_modality=modality))
        
    size = fields["dataset_size"].get().strip()
    if size == "Small (< 10,000)": facts.append(Dataset(size="small"))
    elif size == "Large (> 1 Million)": facts.append(Dataset(size="large"))
        
    imbalance = fields["class_imbalance"].get().strip()
    if imbalance == "Yes": facts.append(Derived(severe_class_imbalance=True))

    # 3. Constraint Facts
    device = fields["deployment_device"].get().strip()
    if device == "Edge Device (Mobile/IoT)": facts.append(Constraint(deployment_device="edge"))
    
    latency = fields["strict_latency"].get().strip()
    if latency == "Yes": facts.append(Derived(strict_latency=True))

    # 4. Diagnostics Facts
    train_err = fields["training_error"].get().strip()
    test_err = fields["test_error"].get().strip()
    
    status_args = {}
    if train_err == "High Error": status_args["training_error"] = "High"
    elif train_err == "Low Error": status_args["training_error"] = "Low"
    
    if test_err == "High Error": status_args["test_error"] = "High"
    elif test_err == "Low Error": status_args["test_error"] = "Low"
    
    if status_args:
        facts.append(ModelStatus(**status_args))

    return facts

def run_pipeline(initial_facts):
    all_facts_in_system = list(initial_facts)

    # 1. Derived Rules
    derived_engine = DerivedRules()
    derived_engine.reset()
    for f in all_facts_in_system: derived_engine.declare(f)
    derived_engine.run()
    all_facts_in_system.extend([f for f in derived_engine.facts.values() if isinstance(f, Derived)])

    # 2. Architecture Rules
    architecture_engine = ArchitectureRules()
    architecture_engine.reset()
    for f in all_facts_in_system: architecture_engine.declare(f)
    architecture_engine.run()
    
    evidences = [f for f in architecture_engine.facts.values() if isinstance(f, ScoreEvidence)]

    # 3. Selection Rules
    selection_engine = SelectionRules()
    selection_engine.reset()
    for f in evidences: selection_engine.declare(f)
    selection_engine.run()
    
    selected = next((f for f in selection_engine.facts.values() if isinstance(f, SelectedArchitecture) and f.get("name") != "None"), None)

    # 4. Detailed Network Rules & Diagnostics
    all_recs = []
    if selected:
        all_facts_in_system.append(selected)
        for engine_class in [ActivationRules, OutputRules, TrainingRules, DiagnosticRules]:
            engine = engine_class()
            engine.reset()
            for f in all_facts_in_system: engine.declare(f)
            engine.run()
            all_recs.extend([f for f in engine.facts.values() if isinstance(f, Recommendation)])

    # ==========================================
    # Report Generation 
    # ==========================================
    report = []
    report.append("=" * 80)
    report.append(" 🧠 EXPERT SYSTEM: DEEP LEARNING CONFIGURATION & DIAGNOSTICS REPORT ")
    report.append("=" * 80 + "\n")

    if not selected:
        report.append(" ⚠️ The system could not determine an architecture. Please provide more context.\n")
        return "\n".join(report)

    recs_dict = {}
    for r in all_recs:
        cat = r.get("category", "Other")
        val = f"{r.get('value')}\n      [Ref: {r.get('source_rule')}]"
        if cat not in recs_dict: recs_dict[cat] = []
        recs_dict[cat].append(val)

    report.append("📌 1. BASE ARCHITECTURE")
    report.append(f" ➤ Architecture : {selected.get('name')}")
    report.append(f" ➤ Justification: {selected.get('reason')}")
    report.append(f"      [Ref: {selected.get('source_rule')}]\n")

    report.append("📌 2. HIDDEN LAYERS STRUCTURE & DEPTH")
    report.append(f" ➤ {', '.join(recs_dict.get('HiddenLayers', ['N/A']))}\n")

    report.append("📌 3. HIDDEN UNITS ACTIVATION FUNCTION")
    report.append(f" ➤ {', '.join(recs_dict.get('HiddenActivation', ['N/A']))}\n")

    report.append("📌 4. INPUT NEURONS / SHAPE CONSTRAINTS")
    report.append(f" ➤ {', '.join(recs_dict.get('InputShape', ['N/A']))}\n")

    report.append("📌 5. OUTPUT NEURONS COUNT LOGIC")
    report.append(f" ➤ {', '.join(recs_dict.get('OutputNeurons', ['N/A']))}\n")

    report.append("📌 6. OUTPUT ACTIVATION FUNCTION")
    report.append(f" ➤ {', '.join(recs_dict.get('OutputActivation', ['N/A']))}\n")

    report.append("📌 7. EXACT LOSS FUNCTION")
    report.append(f" ➤ {', '.join(recs_dict.get('LossFunction', ['N/A']))}\n")

    report.append("📌 8. OPTIMIZATION ALGORITHM & LEARNING RATE")
    report.append(f" ➤ {', '.join(recs_dict.get('Optimizer', ['N/A']))}\n")

    report.append("📌 9. REGULARIZATION & TRAINING TECHNIQUES")
    report.append(f" ➤ {', '.join(recs_dict.get('Regularization', ['N/A']))}\n")

    if "Diagnostics" in recs_dict:
        report.append("📌 10. DIAGNOSTICS & TROUBLESHOOTING")
        for diag in recs_dict["Diagnostics"]:
            report.append(f" ➤ {diag}\n")

    report.append("=" * 80)
    return "\n".join(report)

# ==========================================
# Advanced GUI Construction
# ==========================================
def create_combo(parent, values, default=""):
    var = tk.StringVar(value=default)
    combo = ttk.Combobox(parent, textvariable=var, values=values, state="readonly", font=("Arial", 10))
    return var, combo

def build_gui():
    root = tk.Tk()
    root.title("Deep Learning Expert System - Engineering Dashboard")
    root.geometry("900x800")
    
    BG_DARK       = "#1a1a2e"
    BG_PANEL      = "#16213e"
    BG_CARD       = "#0f3460"
    FG_PRIMARY    = "#e0e0e0"
    FG_ACCENT     = "#00d4ff"
    FG_SUCCESS    = "#00e676"

    root.configure(bg=BG_DARK)

    style = ttk.Style(root)
    style.theme_use('clam')
    
    style.configure("TFrame", background=BG_DARK)
    style.configure("TLabel", background=BG_DARK, foreground=FG_PRIMARY, font=("Consolas", 10))
    style.configure("TNotebook", background=BG_DARK, borderwidth=0)
    style.configure("TNotebook.Tab", background=BG_PANEL, foreground=FG_PRIMARY, font=("Consolas", 10, "bold"), padding=[10, 5])
    style.map("TNotebook.Tab", background=[("selected", BG_CARD)], foreground=[("selected", FG_ACCENT)])

    # 1. Notebook for Tabs
    notebook = ttk.Notebook(root)
    notebook.pack(fill="x", padx=10, pady=10)

    # Tab Frames
    tab_problem = ttk.Frame(notebook)
    tab_data = ttk.Frame(notebook)
    tab_constraints = ttk.Frame(notebook)
    tab_diagnostics = ttk.Frame(notebook)

    notebook.add(tab_problem, text=" 🎯 Problem Scope ")
    notebook.add(tab_data, text=" 📊 Data Profile ")
    notebook.add(tab_constraints, text=" ⚙️ Constraints ")
    notebook.add(tab_diagnostics, text=" 🩺 Diagnostics ")

    fields = {}

    # --- TAB 1: Problem Scope ---
    ttk.Label(tab_problem, text="Task Type:").grid(row=0, column=0, padx=15, pady=15, sticky="w")
    fields["task_type"], w = create_combo(tab_problem, TASK_OPTIONS)
    w.grid(row=0, column=1, padx=15, pady=15, sticky="ew")

    ttk.Label(tab_problem, text="Is this a safety-critical system? (e.g., Medical, Autonomous Driving):").grid(row=1, column=0, padx=15, pady=15, sticky="w")
    fields["safety_critical"], w = create_combo(tab_problem, YES_NO_OPTIONS)
    w.grid(row=1, column=1, padx=15, pady=15, sticky="ew")

    # --- TAB 2: Data Profile ---
    ttk.Label(tab_data, text="Input Modality:").grid(row=0, column=0, padx=15, pady=15, sticky="w")
    fields["input_modality"], w = create_combo(tab_data, INPUT_OPTIONS)
    w.grid(row=0, column=1, padx=15, pady=15, sticky="ew")

    ttk.Label(tab_data, text="Dataset Size:").grid(row=1, column=0, padx=15, pady=15, sticky="w")
    fields["dataset_size"], w = create_combo(tab_data, SIZE_OPTIONS)
    w.grid(row=1, column=1, padx=15, pady=15, sticky="ew")

    ttk.Label(tab_data, text="Is there a severe class imbalance?").grid(row=2, column=0, padx=15, pady=15, sticky="w")
    fields["class_imbalance"], w = create_combo(tab_data, YES_NO_OPTIONS)
    w.grid(row=2, column=1, padx=15, pady=15, sticky="ew")

    # --- TAB 3: Constraints ---
    ttk.Label(tab_constraints, text="Deployment Target:").grid(row=0, column=0, padx=15, pady=15, sticky="w")
    fields["deployment_device"], w = create_combo(tab_constraints, DEVICE_OPTIONS)
    w.grid(row=0, column=1, padx=15, pady=15, sticky="ew")

    ttk.Label(tab_constraints, text="Strict Latency Requirement? (< 50ms):").grid(row=1, column=0, padx=15, pady=15, sticky="w")
    fields["strict_latency"], w = create_combo(tab_constraints, YES_NO_OPTIONS)
    w.grid(row=1, column=1, padx=15, pady=15, sticky="ew")

    # --- TAB 4: Diagnostics ---
    ttk.Label(tab_diagnostics, text="Did you train the model? How is the TRAINING error?").grid(row=0, column=0, padx=15, pady=15, sticky="w")
    fields["training_error"], w = create_combo(tab_diagnostics, ERROR_OPTIONS, default="N/A (Not Trained Yet)")
    w.grid(row=0, column=1, padx=15, pady=15, sticky="ew")

    ttk.Label(tab_diagnostics, text="How is the TEST (Validation) error?").grid(row=1, column=0, padx=15, pady=15, sticky="w")
    fields["test_error"], w = create_combo(tab_diagnostics, ERROR_OPTIONS, default="N/A (Not Trained Yet)")
    w.grid(row=1, column=1, padx=15, pady=15, sticky="ew")

    # Configure columns for expansion
    for tab in [tab_problem, tab_data, tab_constraints, tab_diagnostics]:
        tab.columnconfigure(1, weight=1)

    # 2. Output Area
    output_box = ScrolledText(root, height=22, wrap="word", font=("Consolas", 10), bg=BG_PANEL, fg=FG_SUCCESS, insertbackground=FG_PRIMARY)
    output_box.pack(fill="both", expand=True, padx=10, pady=5)

    def on_run():
        output_box.delete("1.0", tk.END)
        user_facts = build_user_facts(fields)
        if not any(isinstance(f, Problem) for f in user_facts):
            messagebox.showwarning("Missing Input", "Please specify at least a Task Type in the 'Problem Scope' tab.")
            return
        result_report = run_pipeline(user_facts)
        output_box.insert(tk.END, result_report)

    # 3. Action Buttons
    btn_frame = ttk.Frame(root)
    btn_frame.pack(pady=15)

    run_btn = ttk.Button(btn_frame, text="🚀 Run Expert Inference Engine", command=on_run)
    run_btn.pack(side="left", padx=10, ipadx=20, ipady=5)

    lstm_btn = ttk.Button(btn_frame, text="🧠 Launch LSTM Pipeline Builder", command=launch_lstm_builder)
    lstm_btn.pack(side="left", padx=10, ipadx=20, ipady=5)

    root.mainloop()

if __name__ == "__main__":
    build_gui()