"""C3 Milestone 2: toy end-to-end run on SYNTHETIC data.

synthetic spike trains → frozen ToyBackbone → PersonalSNNAdapter → training loop → shared metrics.
Shows that the training machinery works; says nothing about real EEG accuracy.

Run from the repo root:
    python scripts/c3_toy_train.py [--config c3_adapter] [--seed 42] [--epochs 25]
"""

from __future__ import annotations

import argparse
import time

from snn_depression.c3_adapter import (
    PersonalSNNAdapter,
    ToyBackbone,
    make_synthetic_dataset,
    subject_train_test_split,
    train_adapter,
)
from snn_depression.common.config import load_config
from snn_depression.common.contracts import CLASS_NAMES
from snn_depression.common.seed import set_seed
from snn_depression.evaluation.profiling import count_parameters, trainable_ratio


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="c3_adapter")
    parser.add_argument("--seed", type=int, default=None, help="training seed (default: config)")
    parser.add_argument("--epochs", type=int, default=None, help="override training.epochs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_cfg = dict(cfg["training"])
    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    seed = train_cfg["seed"] if args.seed is None else args.seed
    syn = cfg["synthetic"]

    # Data and backbone come from the data seed; the adapter's initial weights from the training
    # seed (set_seed before building it), so a run is fully reproducible from the two seeds.
    data = make_synthetic_dataset(syn, seed=syn["seed"])
    train_idx, test_idx = subject_train_test_split(
        data.subject_ids, data.labels, syn["test_subjects_per_class"], seed=syn["seed"]
    )
    train, test = data.subset(train_idx), data.subset(test_idx)
    backbone = ToyBackbone(syn["n_channels"], seed=syn["seed"], **cfg["toy_backbone"])
    set_seed(seed)
    adapter = PersonalSNNAdapter.from_config(cfg)

    n_train_subj = len(train.subject_ids.unique())
    n_test_subj = len(test.subject_ids.unique())
    print("C3 Milestone 2: toy training on SYNTHETIC data (no claim about real EEG)")
    print(f"  windows: {len(train)} train / {len(test)} test   subjects: {n_train_subj} train / "
          f"{n_test_subj} test (no overlap)")
    print(f"  spike train [B, T, C_spk] = [B, {syn['n_steps']}, {syn['n_channels']}]   "
          f"features [B, T', 256] = [B, {syn['n_steps']}, 256]")
    print(f"  loss {train_cfg['loss']} (correct {train_cfg['correct_rate']}, incorrect "
          f"{train_cfg['incorrect_rate']})   epochs {train_cfg['epochs']}   lr {train_cfg['lr']}   "
          f"batch {train_cfg['batch_size']}   seed {seed}\n")

    start = time.perf_counter()
    history = train_adapter(backbone, adapter, train, test, train_cfg, seed=seed)
    elapsed = time.perf_counter() - start

    print(f"{'epoch':>5}  {'loss':>8}  {'train acc':>9}  {'test acc':>8}  {'test F1':>7}  "
          f"{'min V_th':>8}")
    print(f"{0:>5}  {history['loss'][0]:>8.3f}  {'(before training)':>27}")
    for epoch, (loss, tr, te, vth) in enumerate(
        zip(history["loss"][1:], history["train"], history["test"], history["min_threshold"]), 1
    ):
        print(f"{epoch:>5}  {loss:>8.3f}  {tr['accuracy']:>9.3f}  {te['accuracy']:>8.3f}  "
              f"{te['macro_f1']:>7.3f}  {vth:>8.3f}")

    final = history["test"][-1]
    counts = test.labels.bincount(minlength=len(CLASS_NAMES))
    majority = counts.max().item() / len(test)
    print("\nFinal test results (held-out subjects)")
    print(f"  accuracy  {final['accuracy']:.3f}   (chance {history['chance']:.3f}, "
          f"majority-class baseline {majority:.3f})")
    print(f"  macro-F1  {final['macro_f1']:.3f}")
    print("  confusion matrix (rows = true, columns = predicted)")
    print("            " + "".join(f"{name:>10}" for name in CLASS_NAMES))
    for name, row in zip(CLASS_NAMES, final["confusion_matrix"]):
        print(f"  {name:>9} " + "".join(f"{v:>10}" for v in row))
    n_silent = history["n_silent_test"]
    print(f"  silent test windows: {n_silent} of {final['n']} ({100 * n_silent / final['n']:.1f}%)")
    print(f"  lowest V_th during training: {min(history['min_threshold']):.3f} "
          f"(floor {adapter.threshold_min})")
    ratio = trainable_ratio(adapter, backbone)
    print(f"  trainable_ratio: {ratio:.3f} (adapter {count_parameters(adapter):,} / backbone "
          f"{count_parameters(backbone):,} params)")
    print("    note: the toy backbone is tiny, so this ratio is NOT the real < 5% target; that is"
          " measured against the C2 / fallback backbone.")
    print(f"  training time: {elapsed:.1f} s on CPU")


if __name__ == "__main__":
    main()
