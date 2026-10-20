from snn_depression.common.config import load_config
from snn_depression.common.contracts import CLASS_NAMES, FEATURE_DIM
from snn_depression.common.labels import PHQ9_BINS


def test_component_configs_merge_with_common():
    for name in ["c1_encoding", "c2_global_snn", "c3_adapter", "c4_xai"]:
        cfg = load_config(name)
        assert cfg["project"]["id"] == "J26-DS-348"


def test_yaml_matches_code_contract():
    cfg = load_config()
    assert cfg["contract"]["feature_dim"] == FEATURE_DIM
    assert tuple(cfg["contract"]["class_names"]) == CLASS_NAMES
    assert load_config("c3_adapter")["adapter"]["in_features"] == FEATURE_DIM


def test_yaml_phq9_bins_match_code_and_cover_0_to_27():
    bins = load_config()["labels"]["phq9_bins"]
    assert {name: tuple(bounds) for name, bounds in bins.items()} == PHQ9_BINS
    assert list(bins) == list(CLASS_NAMES)
    covered = sorted(score for low, high in PHQ9_BINS.values() for score in range(low, high + 1))
    assert covered == list(range(28))
