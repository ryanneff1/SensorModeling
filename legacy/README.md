# Legacy workflows

This directory preserves simulation and analysis workflows that are not needed
for the current single-particle manuscript figure set. They are retained for
provenance and possible reuse, but are not part of the supported execution
path for Figures 2--5.

The active code lives in `utils/`, `protocol_scripts/`, `configs/`,
`submission_scripts/`, and `analysis_tools/` at the repository root.

## Contents

- `multiligand/`: original concentration-dependent, many-ligand simulations.
- `early_escape_assays/`: dedicated sweep scripts superseded by the generic
  manuscript escape-assay driver.
- `capture_and_fea/`: exploratory first-capture, finite-element, and LOQ work.
- `notebook_builders/`: one-off notebook generation and formatting scripts
  superseded by the current hand-maintained analysis notebooks.

Files are moved here rather than deleted so that prior calculations remain
traceable. These workflows may require older configurations or dependencies
and are not covered by the current manuscript workflow validation.
