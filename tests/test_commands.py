import ast
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BOT_PY = Path(__file__).resolve().parent.parent / "tgbot" / "bot.py"


def _listed_commands() -> list[tuple[str, str]]:
    tree = ast.parse(BOT_PY.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "BOT_COMMANDS" for t in node.targets
        ):
            out = []
            for elt in node.value.elts:  # type: ignore[union-attr]
                values = {
                    kw.arg: kw.value.value
                    for kw in elt.keywords  # type: ignore[union-attr]
                    if isinstance(kw.value, ast.Constant)
                }
                out.append((values["command"], values["description"]))
            return out
    raise AssertionError("BOT_COMMANDS not found in tgbot/bot.py")


def _registered_commands() -> set[str]:
    tree = ast.parse(BOT_PY.read_text())
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for deco in node.decorator_list:
            for inner in ast.walk(deco):
                if not isinstance(inner, ast.Call) or not isinstance(inner.func, ast.Name):
                    continue
                if inner.func.id == "Command" and inner.args and isinstance(inner.args[0], ast.Constant):
                    found.add(str(inner.args[0].value))
                elif inner.func.id == "CommandStart":
                    found.add("start")
    return found


def test_menu_lists_exactly_the_registered_handlers():
    assert {command for command, _ in _listed_commands()} == _registered_commands()


def test_menu_commands_are_well_formed():
    commands = _listed_commands()
    assert len({c for c, _ in commands}) == len(commands), "duplicate command"
    for command, description in commands:
        assert re.fullmatch(r"[a-z0-9_]{1,32}", command), command
        assert 1 <= len(description) <= 256, command
        assert not description.startswith("/")


def test_every_handler_description_is_meaningful():
    for command, description in _listed_commands():
        assert description[0].isupper(), f"{command}: description should start capitalised"
