from __future__ import annotations

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
    max_steps: int = 20_000
    batch_size: int = 32
    seq_len: int = 512
    learning_rate: float = 1.5e-4
    warmup_steps: int = 1_500
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    # Optimizer
    adamw_betas: tuple[float, float] = (0.9, 0.999)
    adamw_eps: float = 1e-8

    # Data
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_config: str = "sample-10BT"   # subset / config name passed as second arg to load_dataset
    dataset_split: str = "train"

    # Device
    device: str = "cpu"  # "mps" for Apple Silicon, "cuda" for NVIDIA GPU
    use_compile: bool = False  # torch.compile — opt-in; adds 30-60s cold-start overhead
    use_amp: bool = False      # Automatic mixed precision — CUDA only; MPS does not support GradScaler

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    checkpoint_every: int = 1_000

    # Validation
    val_every: int = 250       # steps between val-loss evaluations; 0 disables
    val_batches: int = 20      # number of val batches to average per evaluation
    early_stopping_patience: int = 0  # val evals without improvement before stopping; 0 disables

    # Observability
    grad_log_every: int = 100
    weight_log_every: int = 500
    plot_every: int = 500
    grad_norm_warn_threshold: float = 10.0   # per-layer threshold; emits WARNING line
    grad_norm_spike_threshold: float = 3.0   # total-norm threshold; dumps all layers immediately
    plot_dir: str = "plots"
    log_file: str = "train.log"

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {self.seq_len}")
        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {self.max_steps}")
        if self.grad_clip <= 0:
            raise ValueError(f"grad_clip must be positive, got {self.grad_clip}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative, got {self.weight_decay}")
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {self.warmup_steps}")
        if self.val_batches <= 0:
            raise ValueError(f"val_batches must be positive, got {self.val_batches}")
        if self.val_every < 0:
            raise ValueError(f"val_every must be non-negative, got {self.val_every}")
        if self.early_stopping_patience < 0:
            raise ValueError(
                f"early_stopping_patience must be non-negative, got {self.early_stopping_patience}"
            )
        if self.warmup_steps >= self.max_steps:
            raise ValueError(
                f"warmup_steps ({self.warmup_steps}) must be less than "
                f"max_steps ({self.max_steps})"
            )
        if self.grad_norm_spike_threshold <= 0:
            raise ValueError(
                f"grad_norm_spike_threshold must be positive, got {self.grad_norm_spike_threshold}"
            )
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
