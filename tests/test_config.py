from snn_depression.common.config import load_config
from snn_depression.common.contracts import CLASS_NAMES, FEATURE_DIM


def test_component_configs_merge_with_common():
    for name in ["c1_encoding", "c2_global_snn", "c3_adapter", "c4_xai"]:
        cfg = load_config(name)
        assert cfg["project"]["id"] == "J26-DS-348"


def test_yaml_matches_code_contract():
    cfg = load_config()
    assert cfg["contract"]["feature_dim"] == FEATURE_DIM
    assert tuple(cfg["contract"]["class_names"]) == CLASS_NAMES
    assert load_config("c3_adapter")["adapter"]["in_features"] == FEATURE_DIM
