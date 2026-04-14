from .losses import LossComputer

def __getattr__(name: str):
    if name == "Trainer":
        from .trainer import Trainer
        return Trainer
    if name == "Evaluator":
        from .eval import Evaluator
        return Evaluator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["LossComputer", "Trainer", "Evaluator"]
