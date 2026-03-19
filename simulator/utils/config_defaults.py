import copy
import os
import yaml

DEFAULT_CONFIG_TEMPLATE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "configs",
    "default.yaml",
)


def _load_yaml(path: str) -> dict:
    if not path or not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _merge_with_defaults(config: dict, defaults: dict) -> dict:
    """仅为缺省字段填充默认值，不覆盖显式配置。"""
    if not isinstance(config, dict):
        config = {}
    if not isinstance(defaults, dict):
        return copy.deepcopy(config)
    merged = copy.deepcopy(config)
    for key, value in defaults.items():
        if key not in merged:
            merged[key] = copy.deepcopy(value)
            continue
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = _merge_with_defaults(merged[key], value)
    return merged


def apply_config_defaults(config: dict, template_path: str) -> dict:
    """用模板配置填充缺省项，实际 config 优先。"""
    defaults = _load_yaml(template_path)
    return _merge_with_defaults(config, defaults)
