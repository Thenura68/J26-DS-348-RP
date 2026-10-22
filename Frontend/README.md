# Frontend: project dashboard

A small Streamlit app that shows the work of all four components in one place.
Pure Python: no JavaScript or build step.

## Run it

With the project environment active:

```bash
pip install -r Frontend/requirements.txt   # once, from the repository root
cd Frontend
streamlit run app.py
```

Launch it **from inside `Frontend/`**: Streamlit only reads the theme
(`.streamlit/config.toml`: dark-blue left panel, white text) from the folder it is
started in. If you start it elsewhere, the left panel shows a reminder.

It opens at http://localhost:8501. Pages read saved results from `outputs/` and
`data/interim/` (git-ignored), so run your component's scripts first; missing files show
a hint instead of an error.

## Structure

```
Frontend/
├── app.py                  # entry point: registers the pages (shared)
├── requirements.txt        # streamlit (shared)
├── .streamlit/config.toml  # theme (shared)
├── shared/                 # shared code (change only via a "shared:" PR)
│   ├── registry.py         #   the four components: titles, owners, branches
│   ├── layout.py           #   page_header, placeholder, show_image, show_table, metric_row
│   ├── data.py             #   output_dir, load_csv, load_json, load_manifest
│   └── home.py             #   overview page: pipeline, team, interface, dataset status
└── components/             # one folder per member
    ├── c1_encoding/        #   Rathnayake       (branch Encoding)
    ├── c2_global_snn/      #   Wickramaarachchi (branch Global-Model)
    ├── c3_adapter/         #   Jayarathna       (branch Personalized-Adaptive-Layer)
    └── c4_xai/             #   Edirisinghe      (branch Explainable-AI)
        ├── page.py         #     your page: the app calls render()
        ├── README.md       #     rules for your folder
        └── assets/         #     small static images (no patient data)
```

## Rules

1. **Edit only your own folder** under `components/`. Shared files change through a
   `shared:` pull request, like the rest of the project.
2. **Keep the four standard tabs** (`STANDARD_SECTIONS` in `shared/layout.py`): sub-problem
   and approach, prototype and results, user requirements met, design contribution.
3. **Load, don't compute.** Pages show results your scripts already saved in
   `outputs/<component>/`; no training or heavy processing inside the page.
4. **No hard-coded paths.** Use `shared.data.output_dir("c2")` etc. (it follows `SNN_OUTPUT_ROOT`).
5. **Never commit data**: no patient data, spikes, profiles or checkpoints in `assets/`.

## Adding to your page

```python
from shared.data import load_csv, output_dir
from shared.layout import metric_row, show_image, show_table

results = output_dir("c2")
metric_row({"Test macro-F1": "0.71", "Parameters": "320k"})
show_image(results / "training_curve.png", "Training loss")
show_table(load_csv(str(results / "summary.csv")), "Run scripts/c2_train.py first.")
```

`components/c1_encoding/page.py` is a working example.
