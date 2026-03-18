from importlib import import_module

__all__ = [
    "FeatureTrainingArtifacts",
    "FeatureTrainer",
    "build_feature_trainer",
    "run_feature_training",
]


def __getattr__(name):
    if name in __all__:
        module = import_module(".trainer", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
