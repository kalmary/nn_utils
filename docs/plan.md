# Nested nn_utils Integration Plan

This checkout follows the canonical rebuild plan at `../../../../../utils/nn_utils/docs/plan.md` from the BRIK repository root.

All rebuild work in this nested checkout must be performed on `development`. Verify the branch first and request explicit approval before creating or switching it.

For `TreeClassification`, completion additionally requires:

- [x] the nested checkout exposes the same current `nn_utils` API as the canonical project;
- [ ] training, evaluation, model I/O, losses, metrics, callbacks, and reports import only named utilities;
- [x] both `uv sync --extra pytorch-cpu` and `uv sync --group test --extra pytorch-cpu` resolve without copying `nn_utils` source into TreeClassification;
- [ ] direct and module invocation tests use the nested package consistently;
- [ ] the tree-classification focused and full suites pass after an explicitly approved submodule revision update.

No Git or submodule update is authorized by this plan.
