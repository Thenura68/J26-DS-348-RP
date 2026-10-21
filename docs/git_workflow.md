# Git Workflow

Repository: https://github.com/Thenura68/J26-DS-348-RP (public — never commit data or patient files)

## Branches
```
main                            ← shared skeleton + merged, working components; update only via pull requests
├── Encoding                    ← C1  Rathnayake
├── Global-Model                ← C2  Wickramaarachchi
├── Personalized-Adaptive-Layer ← C3  Jayarathna
└── Explainable-AI              ← C4  Edirisinghe
```
For bigger pieces of work you can branch off your component branch (e.g. `c3/fallback-stdp`)
and merge back into your component branch when done.

## First time (each member)
The component branches were created before the skeleton was added to `main`, so bring the
skeleton into your branch once:
```bash
git clone https://github.com/Thenura68/J26-DS-348-RP.git
cd J26-DS-348-RP
git checkout Personalized-Adaptive-Layer     # your branch
git merge origin/main                        # pulls in the skeleton
git push
```

## Daily loop
```bash
git pull                            # on your branch
# ... work ...
git add -A
git commit -m "c3: add LIF adapter head"
git push
```
Commit message prefix = your component (`c1:`, `c2:`, `c3:`, `c4:`) or `shared:`.

## Staying up to date with main
```bash
git fetch origin
git merge origin/main               # into your branch, at least weekly
```

## Getting your work into main
Open a pull request from your branch → `main`. Fill in the template. Tests must pass and one
other member must approve. Merge when a piece is stable — not every commit.

## Recommended GitHub settings (repo owner)
- Settings → Collaborators: add all members.
- Settings → Branches → rule for `main`: require a pull request, 1 approval, and the `tests` check.

## Shared files
Anything in `src/snn_depression/common/`, `src/snn_depression/data/`,
`src/snn_depression/evaluation/`, `configs/common.yaml` or `docs/interface_contract.md` affects
everyone. Change these only in a small separate PR titled `shared: ...` with all members as reviewers.

## Never commit
Data, checkpoints, patient profiles, `.env` files, tokens. This repo is public.
`.gitignore` covers the usual cases — check `git status` before every commit.
