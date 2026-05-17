---
name: hdf5_validation
description: "Guide the agent to validate HDF5 prediction bundles, initial-condition constraints, and submission traceability."
---

# HDF5 Validation

Use this skill when checking prediction artifacts or submission bundles.

Rules:
- Verify dataset existence, dtype, and shape before interpreting metrics.
- Enforce `(N, 200, 256)` prediction shape and exact first-10-step carryover when required.
- Treat NaN or Inf values as blocking failures.
- Ensure submission/code outputs remain traceable to logged write operations.
