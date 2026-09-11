# Architecture changes in v0.6

The production web architecture remains in architecture.md. The new path is:

`worker configuration -> round subprocess -> injected GraphResearchPolicy -> frozen PyTorch graph model`

The normal ResearchPolicy remains the default and fallback. PyTorch is imported only when a checkpoint is selected or the experiment CLI is used. Capture without a model needs no PyTorch. Planner inference runs inside the existing round process, subject to its cancellation and timeout; offline training runs in the CLI and never in an API request.

`ResearchState.planner_data` is persisted in the existing SQL snapshot JSON column; no database migration is required. Traces, checkpoint digest, status and predictions are included in authenticated run exports and API responses. There is no new unauthenticated upload, code execution or model promotion endpoint.

A recording snapshot is taken before exploration; outcomes are attached after verification/synthesis. Failed rounds roll back to the prior completed checkpoint through the existing worker. The default scheduler and proof verification code are unchanged. Model-loading/configuration errors fall back and are visible in the neural policy view.

See GRAPH-PLANNER.md for tensor schemas, cost/reward definitions, trust boundaries and training/deployment commands. See PLANNER-RESULTS.md for measured behavior and untested integrations.
