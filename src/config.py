from dataclasses import dataclass


@dataclass
class TrainConfig:
    # Model
    vocab_size: int = 8192
    n_layers: int = 6
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2048

    # Training
    max_steps: int = 10_000
    batch_size: int = 32
    seq_len: int = 512
    learning_rate: float = 3e-4
    warmup_steps: int = 500
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    # Data
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_config: str = "sample-10BT"   # subset / config name passed as second arg to load_dataset
    dataset_split: str = "train"

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    checkpoint_every: int = 1_000

    # Observability
    grad_log_every: int = 100
    weight_log_every: int = 500
    plot_every: int = 500
    grad_norm_warn_threshold: float = 10.0
    plot_dir: str = "plots"
