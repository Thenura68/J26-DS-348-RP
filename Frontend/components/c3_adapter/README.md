# C3 dashboard page

Owner: **Jayarathna** · Branch: `Personalized-Adaptive-Layer`

This folder is yours. Edit only files here; the shared app (`Frontend/app.py`,
`Frontend/shared/`) changes only through a `shared:` pull request.

| File | Purpose |
|---|---|
| `page.py` | your page; the app calls `render()` |
| `assets/` | small static images you made for the page (diagrams, screenshots) |

Rules:
- Read results from `outputs/c3/` with `shared.data.output_dir("c3")`; never hard-code paths.
- Results and data are git-ignored, so handle missing files (the helpers already do).
- Do not commit patient data, spikes, profiles or checkpoints into `assets/`.
- Keep heavy computation in your scripts; the page should only load saved results.
