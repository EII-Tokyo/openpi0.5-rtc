import ast
import pathlib


def test_rinse_small_rl_token_config_matches_base_checkpoint_inputs():
    config_path = pathlib.Path(__file__).with_name("config.py")
    tree = ast.parse(config_path.read_text())
    call = _find_config_call(tree, "eii_rinse_11repo_cam4_fullft_rl_token_small")
    small_factory = _find_function(tree, "_make_small_rl_token_autoencoder_config")
    repack_call = _find_call(small_factory, "_aloha_real_repack_transforms")

    assert _keyword_value(call, "num_train_steps") == 10_000
    assert _keyword_value(call, "save_interval") == 2_500
    assert _keyword_value(repack_call, "include_low") is True
    assert _keyword_value(repack_call, "include_subtask") is False


def _find_config_call(tree: ast.AST, config_name: str) -> ast.Call:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not node.args:
            continue
        first_arg = node.args[0]
        if isinstance(first_arg, ast.Constant) and first_arg.value == config_name:
            return node
    raise AssertionError(f"Config call not found: {config_name}")


def _find_function(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"Function not found: {name}")


def _find_call(tree: ast.AST, name: str) -> ast.Call:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name:
            return node
    raise AssertionError(f"Call not found: {name}")


def _keyword_value(call: ast.Call, name: str):
    for keyword in call.keywords:
        if keyword.arg == name:
            return ast.literal_eval(keyword.value)
    raise AssertionError(f"Keyword not found: {name}")
