import ast
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BOT_PY = Path(__file__).resolve().parent.parent / "tgbot" / "bot.py"

# removed from the bot: no menu entry, no handler, nothing to type
REMOVED_COMMANDS = {"stats", "last", "sync"}


def _module_value(name: str) -> ast.AST:
    tree = ast.parse(BOT_PY.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return node.value
    raise AssertionError(f"{name} not found in tgbot/bot.py")


def _listed_commands() -> list[tuple[str, str]]:
    values = []
    for elt in _module_value("BOT_COMMANDS").elts:  # type: ignore[union-attr]
        data = {
            kw.arg: kw.value.value
            for kw in elt.keywords  # type: ignore[union-attr]
            if isinstance(kw.value, ast.Constant)
        }
        values.append((data["command"], data["description"]))
    return values


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
    listed = {command for command, _ in _listed_commands()}
    assert listed == _registered_commands(), "menu and handlers disagree"


def test_removed_commands_have_no_handler_and_no_menu_entry():
    registered = _registered_commands()
    listed = {command for command, _ in _listed_commands()}
    source = BOT_PY.read_text()
    for command in sorted(REMOVED_COMMANDS):
        assert command not in registered, f"/{command} still has a handler"
        assert command not in listed, f"/{command} still listed in the menu"
        assert f'Command("{command}")' not in source
        assert f"/{command} " not in source, f"/{command} still mentioned in help text"


def test_menu_commands_are_well_formed():
    commands = _listed_commands()
    assert len({c for c, _ in commands}) == len(commands), "duplicate command"
    for command, description in commands:
        assert re.fullmatch(r"[a-z0-9_]{1,32}", command), command
        assert 1 <= len(description) <= 256, command
        assert not description.startswith("/")
        assert not description[0].isascii(), f"{command}: menu description should lead with an emoji"
        first_letter = next(ch for ch in description if ch.isalpha())
        assert first_letter.isupper(), f"{command}: description should start capitalised"


def test_menu_button_labels_lead_with_an_emoji():
    source = BOT_PY.read_text()
    keyboard = source.split("def menu_keyboard", 1)[1].split("\ndef ", 1)[0]
    labels = re.findall(r'text="([^"]+)"', keyboard)
    assert labels[:4], "menu keyboard lost its buttons"
    for label in labels[:4]:
        assert not label[0].isascii(), f"{label!r} should start with an emoji"
