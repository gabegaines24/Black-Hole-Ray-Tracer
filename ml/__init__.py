# ml/ — Machine learning surrogate for Phase 3 warm-starting
#
# Usage:
#   from ml.dataset import generate_dataset, save_dataset, load_dataset
#   from ml.surrogate import init_random_weights, train, predict_batch
#   from ml.runtime_gate import RuntimeGate
#   from ml.schema import normalize_inputs, denormalize_outputs

__all__ = [
    "generate_dataset",
    "save_dataset",
    "load_dataset",
    "init_random_weights",
    "train",
    "predict_batch",
    "RuntimeGate",
    "normalize_inputs",
    "denormalize_outputs",
]
