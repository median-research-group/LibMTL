import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


def _load_config_module():
    config_path = Path(__file__).parents[1] / "LibMTL" / "config.py"
    spec = importlib.util.spec_from_file_location("libmtl_config", config_path)
    module = importlib.util.module_from_spec(spec)
    fake_torch = ModuleType("torch")
    fake_torch.cuda = SimpleNamespace(is_available=lambda: False)
    with patch.dict(sys.modules, {"numpy": ModuleType("numpy"), "torch": fake_torch}):
        spec.loader.exec_module(module)
    return module


def test_command_line_help_is_renderable():
    parser = _load_config_module().LibMTL_args
    whitespace_help_options = [
        action.option_strings
        for action in parser._actions
        if isinstance(action.help, str) and action.help.isspace()
    ]
    help_text = parser.format_help()

    assert whitespace_help_options == []
    assert "--DB_beta" in help_text
    assert "--STCH_mu" in help_text
    assert "--FORUM_phi" in help_text
