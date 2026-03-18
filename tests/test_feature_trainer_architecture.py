import ast
from pathlib import Path


def _load_trainer_module_ast():
    source = Path("saga_training/feature/trainer.py").read_text()
    return ast.parse(source)


def test_feature_trainer_constructor_uses_flat_explicit_dependencies():
    tree = _load_trainer_module_ast()
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "FeatureTrainer":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    arg_names = [arg.arg for arg in item.args.args]
                    kwonly_names = [arg.arg for arg in item.args.kwonlyargs]
                    assert arg_names == ["self"]
                    assert kwonly_names == [
                        "model",
                        "train_dataloader_factory",
                        "val_dataloader_factory",
                        "reporter",
                        "checkpoints",
                        "training_cfg",
                        "device",
                        "pipe_cfg",
                        "background_rgb",
                        "background_instance",
                        "background_semantic",
                        "dependencies",
                    ]
                    return
    raise AssertionError("FeatureTrainer.__init__ not found")


def test_feature_training_has_explicit_builder_function():
    tree = _load_trainer_module_ast()
    function_names = {
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    assert "build_feature_trainer" in function_names
    assert "run_feature_training" in function_names


def test_feature_trainer_uses_manifest_builder_instead_of_raw_artifact_paths():
    source = Path("saga_training/feature/trainer.py").read_text()
    assert "build_feature_manifest" in source
    assert "segment_masks_dir=" not in source
    assert "segment_labels_dir=" not in source
    assert "segment_label_features_path=" not in source
