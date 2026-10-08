# Data (not committed)

Everything in this folder except this README is ignored by git.

## Getting MODMA
MODMA (Multi-modal Open Dataset for Mental-disorder Analysis, Lanzhou University) is released
under a usage agreement. One member requests access for the team; store the download once in
the shared Google Drive folder `J26-DS-348/data/` so everyone uses the same copy.

## Expected layout
```
data/
├── raw/
│   └── modma/
│       ├── eeg_128ch_resting/      # files exactly as downloaded
│       ├── eeg_3ch_resting/
│       └── subjects_information.xlsx   # PHQ-9 and demographics
├── interim/                        # filtered / windowed arrays (shared data pipeline output)
└── processed/                      # encoded spike trains, cached feature tensors
```

## Pointing code at the data
Code reads the data location from the `SNN_DATA_ROOT` environment variable
(see `src/snn_depression/common/paths.py`). If unset, it uses this `data/` folder.

- Local: `export SNN_DATA_ROOT=/path/to/data` (Windows PowerShell: `$env:SNN_DATA_ROOT="C:\path\to\data"`)
- Colab: set in the setup notebook to `/content/drive/MyDrive/J26-DS-348/data`
