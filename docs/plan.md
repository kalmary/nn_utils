# nn_utils Rebuild Plan

**Goal:** Convert `nn_utils` into a reproducible Python 3.12 uv project with explicit exports and tested, backward-compatible neural-network utilities.

**Role:** This is the shared dependency of `PCDSegmentation` and `TreeClassification`; its stable API is completed before either consumer is structurally changed.

**Design:** `../../../../docs/rebuild.md`

**Branch requirement:** Perform all rebuild work on `development`, including each consumer's nested checkout when it is updated. Verify branches first and request explicit approval before creating or switching them.

## Contracts to Preserve

- Existing function and class names, parameters, return types, tensor shapes, and reduction behavior.
- Model state loading/saving and JSON conversion formats.
- Metrics and loss behavior for binary, multiclass, batched, empty, and ignored-label inputs currently supported.
- `Plotter`, `classification_report`, and `EarlyStopping` behavior.
- Existing imports through `nn_utils` while consumers transition from wildcard imports to explicit names.

## Task 1: Establish the uv project

**Files:** create `.python-version`, `pyproject.toml`, `uv.lock`; update `.gitignore` and installation documentation.

- [x] Inventory imports in `src/*.py` and declare only direct packages.
- [x] Define Python 3.12, the `basic` group, and a `dev` group including `basic`, `pytest`, `matplotlib`, and `pyvista`.
- [x] Keep plotting/report dependencies in `dev`; tensor, metric, loss, callback, and model-I/O imports in `basic` do not eagerly load visualization modules.
- [x] Configure official CPU and CUDA 13.2 profiles for PyTorch 2.14/Torchvision 0.29; Torchaudio is not used.
- [x] Verify the basic environment can import non-plotting utilities.
- [x] Verify the test environment can import the complete project.
- [x] Remove old pip instructions after both consumer checkouts resolve and import against their locks.

## Task 2: Characterize public behavior

**Files:** add colocated unit tests to the files owning each function; add cross-module tests under `tests/`.

- [x] Test probability/label conversion, accuracy, class weighting, mIoU, and dataset-length behavior.
- [x] Test every loss for expected shapes, dtypes, finite values, invalid inputs, and CPU device consistency. CUDA device checks remain hardware-dependent.
- [x] Test early stopping state transitions and callback outputs.
- [x] Test JSON conversion and model save/load round trips using a small deterministic model.
- [x] Test plot/report creation in a temporary directory with a non-interactive backend.
- [x] Record the explicit names used by PCD and tree-classification consumers.

### Consumer import inventory

The current `utils` packages in both consumers re-export `nn_utils` names. The
imports below are the names used directly by their source files; preserve them
when replacing those re-exports with named imports.

| Consumer | Imported names |
| --- | --- |
| Both | `load_json`, `load_model`, `convert_str_values`, `save_to_json`, `save_model`, `Plotter`, `classification_report`, `calculate_accuracy`, `get_int_labels`, `get_probabilities`, `get_dataset_len`, `FocalLoss`, `wrap_hist` |
| PCD only | `EarlyStopping`, `compute_miou`, `compute_pos_weights_h5` |
| Tree classification only | `compute_pos_weights`, `ArcfaceFocalLoss` |

`Plotter` and `classification_report` remain explicit lazy imports: normal
inference must not load visualization dependencies.

## Task 3: Make the package boundary explicit

**Files:** modify `__init__.py`; reorganize `src/` only after tests exist.

- [x] Replace wildcard exports with an explicit compatibility export list.
- [x] Use package-relative internal imports and remove any working-directory dependence. The current source files have no cross-module imports; the package entry point uses relative imports.
- [x] Keep old import paths forwarding to the tested implementations if files are split. No source files were split, so existing paths remain the implementations.
- [x] Separate plotting/report code from tensor-only imports so `basic` stays headless.
- [x] Avoid framework abstractions; keep the existing functions and small classes.

## Task 4: Consumer acceptance

- [x] Run the PCD segmentation utility imports and focused offline-workflow tests against the shared public interface. All 44 PCD-owned cases passed; one parent-project invocation case requires the root project environment rather than the isolated PCD environment.
- [x] Run the tree-classification utility imports and focused tests against the shared public interface. All 31 cases passed.
- [x] Compare nested `nn_utils` checkout revisions before integration. The canonical checkout and both consumer checkouts point to `c2e2689` before the standalone pytest configuration update.
- [x] Request approval before any Git/submodule revision update.
- [ ] Re-run both consumer suites after their approved revision updates.

## Completion Gate

- The project resolves independently with both uv groups.
- Public imports are explicit and compatible.
- CPU tests pass; CUDA tensor/loss smoke tests pass on Linux GPU hardware.
- Both consuming submodules pass their focused suites against the same `nn_utils` interface.
