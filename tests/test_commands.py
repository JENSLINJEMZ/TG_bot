import ast
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BOT_PY = Path(__file__).resolve().parent.parent / "tgbot" / "bot.py"


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


def _admin_commands() -> set[str]:
    node = _module_value("ADMIN_COMMANDS")
    return {item.value for item in node.elts}  # type: ignore[union-attr]


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


def test_menu_lists_exactly_the_registered_non_admin_handlers():
    listed = {command for command, _ in _listed_commands()}
    assert listed == _registered_commands() - _admin_commands()
    assert listed <= _registered_commands()


def test_admin_commands_are_registered_but_hidden():
    source = BOT_PY.read_text()
    assert "return bool(user and config.OWNER_USER_ID" in source, "owner gate missing"
    listed = {command for command, _ in _listed_commands()}
    for command in sorted(_admin_commands()):
        assert command in _registered_commands(), f"/{command} lost its handler"
        assert command not in listed, f"/{command} leaked into the menu"
        assert f'Command("{command}")' in source
        assert f"/{command} " not in source, f"/{command} mentioned in help/labels"
        handler = source.split(f'Command("{command}")', 1)[1].split("\n@dp", 1)[0]
        assert "_require_owner" in handler, f"/{command} handler is not owner-gated"


def test_owner_gate_and_cooldown_are_defined_and_used():
    source = BOT_PY.read_text()
    assert "async def _require_owner" in source
    assert "def _cooldown_seconds" in source
    assert source.count("_cooldown_seconds(") >= 6, "cooldown should gate each heavy handler"


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