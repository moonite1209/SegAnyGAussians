# Repository Guidelines

## Project Structure & Module Organization
- Core pipeline entry points live at the repository root: `train_scene.py`, `segment.py`, `train_feature.py`, `cluster.py`, `gui.py`, and `render.py`.
- Shared logic is organized into `scene/` (Gaussian/camera data models), `gaussian_renderer/` (render backends), `utils/` (I/O and helpers), and `arguments/`.
- Configuration is Hydra-based under `configs/` (`model/`, `dataset/`, `segment/`, `training/`, `clustering/`, `gui/`).
- Tests are in `tests/` (current coverage focuses on camera loading/data handling).
- External code and CUDA extensions are in `submodules/` and `third_party/`; keep local weights/checkpoints in `weights/`.

## Build, Test, and Development Commands
- `conda env create --file environment.yml && conda activate saga` — create the default environment.
- `pip install -r requirements.txt` — alternate Python 3.10/CUDA 11.8 setup used by recent local workflows.
- `python train_scene.py -s <scene_dir>` — pretrain base 3D Gaussian model.
- `python segment.py base_path=data/<scene>` — generate SAM/GroundingDINO masks and labels.
- `python train_feature.py base_path=data/<scene>` then `python cluster.py base_path=data/<scene>` — train features and assign clusters/classes.
- `python gui.py base_path=data/<scene>` — inspect results interactively.
- `python -m pytest tests -q` — run unit tests.

## Coding Style & Naming Conventions
- Follow Python conventions: 4-space indentation, `snake_case` for functions/variables/files, `PascalCase` for classes.
- Keep new config files lowercase and descriptive (example: `configs/training/my_scene.yaml`).
- Prefer explicit paths from Hydra config (`base_path`) over hard-coded absolute paths.
- Add brief comments only where tensor shapes, camera transforms, or feature normalization are non-obvious.

## Testing Guidelines
- Use `pytest` with files named `tests/test_<module>.py` and focused test functions (`test_<behavior>`).
- Prefer deterministic CPU-friendly unit tests for loaders/utils; avoid requiring full training runs in tests.
- For pipeline changes, include a smoke command in PR notes (for example, `python segment.py base_path=data/<scene>`).

## Commit & Pull Request Guidelines
- Existing history favors short, lowercase subjects (`fix`, `refactor`, `cluster`); keep commits concise and action-oriented.
- Keep one logical change per commit; separate refactors from behavior changes.
- PRs should include: purpose, impacted scripts/configs, exact reproduction commands, and sample outputs (screenshots for `gui.py` changes).
- Link related issues/data assumptions and call out any required checkpoints or local paths reviewers must prepare.

## Security & Configuration Tips
- Do not commit datasets, generated outputs, model weights, or large archives.
- Keep machine-specific paths and credentials out of tracked config; override via CLI/Hydra at runtime.
