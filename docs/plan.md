# nn_utils Rebuild Plan

**Goal:** Convert `nn_utils` into a reproducible Python 3.12 uv project with explicit exports and tested, backward-compatible neural-network utilities.

**Role:** This is the shared dependency of `PCDSegmentation` and `TreeClassification`; its stable API is completed before either consumer is structurally changed.

**Design:** `../../../../docs/rebuild.md`

**Branch requirement:** Perform all rebuild work on `development`, including each consumer's nested checkout when it is updated. Verify branches first and request explicit approval before creating or switching them.

## Contracts to Preserve

- Existing function and class names, parameters, return types, tensor shapes, and reduction behavior.
- Model state loading/saving and JSON conversion formats.
- Metrics and loss behavior for binary, multiclass, batched, empty, and ignored-label inputs currently supported.
- `Plotter`, `ClassificationReport`, and `EarlyStopping` behavior.
- Existing imports through `nn_utils` while consumers transition from wildcard imports to explicit names.

## Task 1: Establish the uv project

**Files:** create `.python-version`, `pyproject.toml`, `uv.lock`; update `.gitignore` and installation documentation.

- [x] Inventory imports in `src/*.py` and declare only direct packages.
- [x] Define Python 3.12, the `basic` group, and a `test` group including `basic`, `pytest`, `matplotlib`, and `pyvista`.
- [x] Keep plotting/report dependencies in `test`; tensor, metric, loss, callback, and model-I/O imports in `basic` do not eagerly load visualization modules.
- [x] Configure official CPU and CUDA 13.2 profiles for PyTorch 2.14/Torchvision 0.29; Torchaudio is not used.
- [x] Verify the basic environment can import non-plotting utilities.
- [x] Verify the test environment can import the complete project.
- [x] Remove old pip instructions after both consumer checkouts resolve and import against their locks.

## Task 2: Characterize public behavior

**Files:** add colocated unit tests to the files owning each function; add cross-module tests under `tests/`.

- [x] Test probability/label conversion, accuracy, class weighting, mIoU, and dataset-length behavior.
- [ ] Test every loss for expected shapes, dtypes, finite values, invalid inputs, and device consistency.
- [x] Test early stopping state transitions and callback outputs.
- [x] Test JSON conversion and model save/load round trips using a small deterministic model.
- [x] Test plot/report creation in a temporary directory with a non-interactive backend.
- [ ] Record the explicit names used by PCD and tree-classification consumers.

## Task 3: Make the package boundary explicit

**Files:** modify `__init__.py`; reorganize `src/` only after tests exist.

- [x] Replace wildcard exports with an explicit compatibility export list.
- [ ] Use package-relative internal imports and remove any working-directory dependence.
- [ ] Keep old import paths forwarding to the tested implementations if files are split.
- [ ] Separate plotting/report code from tensor-only imports so `basic` stays headless.
- [ ] Avoid framework abstractions; keep the existing functions and small classes.

## Task 4: Consumer acceptance

- [ ] Run the PCD segmentation utility imports and focused model/data-loader tests against this checkout.
- [ ] Run the tree-classification utility imports and focused model/data-loader tests against this checkout.
- [ ] Compare nested `nn_utils` checkout revisions before integration.
- [ ] Request approval before any Git/submodule revision update.
- [ ] Re-run both consumer suites after their approved revision updates.

## Completion Gate

- The project resolves independently with both uv groups.
- Public imports are explicit and compatible.
- CPU tests pass; CUDA tensor/loss smoke tests pass on Linux GPU hardware.
- Both consuming submodules pass their focused suites against the same `nn_utils` interface.
