# v0.7 studio validation — 2026-09-10

## Verified locally

- Python: **77 passed**, including the prior 59 tests and 18 new studio/source/model/bridge checks.
- Frontend: **3 unit tests passed**; ESLint, TypeScript and Vite production build passed.
- Positive text-based PDF extraction was tested with a generated PDF, including page marker and byte hash. Blank/scanned-style PDF rejection was tested.
- Scalar exponential decay and a coupled harmonic oscillator were compared against closed-form solutions. The decay endpoint at t=4 was 0.13533528325935726, compared with exp(-2); the sampled 200/400-step difference was approximately 2.90e-11.
- Source quotes were checked against exact original spans. A mocked language-model quote absent from the source was labeled unanchored. No live OpenAI call was made.
- Unsafe expression forms (imports, attributes, comprehensions, unknown calls, unbounded powers) were rejected. Divergent trajectories were recorded as numerical failures, not theorems.
- SQLite API + actual round subprocesses completed source import, analysis and experiment stages. Tests cover restored snapshots, independent fork creation, budget limits, unverified source attachment and the research bridge's request construction.
- In-app browser: selected the mathematics example, submitted it, observed completion, opened the modal animation, scrubbed to step 125, exported WebM, forked the method and extended the horizon to observe numerical failure near t=1.
- The browser-produced WebM was inspected with ffprobe: VP9, 900 × 450, 200 decoded frames, 9.961879 seconds. The included video is silent and renders computed samples, not a generated explanation video.
- Compose configuration passed `docker compose config --quiet`. This is not an image-build or deployment test.

## Unverified or limited

- Real OpenAI paper interpretation and live MARE research execution from an arbitrary source were not exercised; no external provider key was used.
- v0.7 Docker image builds, Linux/PostgreSQL studio execution, remote CI, GPU runtimes and cross-browser video encoding were not tested.
- There is no OCR, general formula-layout recovery, automatic DNN training from papers, or proof certification of a numerical experiment.
- A small numerical refinement difference is not a rigorous error bound. The current explicit RK4 solver is not a general stiff-system solver.
- Browser animation was visually inspected at the available desktop viewport; a complete accessibility/mobile audit has not been performed.

The included authored example is not a fabricated research paper. Successful and failed experiment packets, model input and the exported animation are under `examples/studio/`. Older v0.4/v0.5/v0.6 validation documents are historical.
