"""Apply a small, idempotent correctness patch to ComfyUI-ReActor.

ReActor's batch face swap appends only frames where a face was found to
``target_faces`` and then zips that shorter list with the full frame list.  A
single frame without a detected face therefore shifts every later frame by one,
so the wrong face is swapped (or a face is skipped) for the rest of the clip.

This patch keeps one entry per frame (``None`` when no face was found) and skips
``None`` entries while swapping.  It is safe to run more than once and creates a
``.py.bak`` backup before the first change.

Usage:
    python tools/patch_reactor.py [--comfy-dir D:\\JEMZ\\ComfyUI] [--check]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

MARKER = "# tgbot-patch: keep target_faces aligned with target_imgs"

OLD_APPEND = """                if target_face is not None:
                    target_faces.append(target_face)
                pbar.update(1)
"""

NEW_APPEND = f"""                {MARKER}
                target_faces.append(target_face)
                pbar.update(1)
"""

OLD_EMPTY = """            if len(target_faces) == 0:
                logger.status("Cannot detect any Target, skipping swapping...")
                return result_images, bbox, swapped_indexes
"""

NEW_EMPTY = """            if not any(face is not None for face in target_faces):
                logger.status("Cannot detect any Target, skipping swapping...")
                return result_images, bbox, swapped_indexes
"""

OLD_LOOP = """                    for i, (target_img, target_face_list) in enumerate(zip(results, target_faces)):
                        target_face_single, wrong_gender, target_face_index = get_face_single(target_img, target_face_list, face_index=face_num, gender_target=gender_target, order=faces_order[0])
"""

NEW_LOOP = """                    for i, (target_img, target_face_list) in enumerate(zip(results, target_faces)):
                        if target_face_list is None:
                            pbar.update(1)
                            continue
                        target_face_single, wrong_gender, target_face_index = get_face_single(target_img, target_face_list, face_index=face_num, gender_target=gender_target, order=faces_order[0])
"""


def default_comfy_dir() -> Path:
    env = os.environ.get("COMFY_DIR")
    if env:
        return Path(env)
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from tgbot import config

        return Path(config.COMFY_DIR)
    except Exception:
        return Path(r"D:\JEMZ\ComfyUI")


def reactor_file(comfy_dir: Path) -> Path:
    return (
        comfy_dir
        / "custom_nodes"
        / "ComfyUI-ReActor"
        / "scripts"
        / "reactor_swapper.py"
    )


def patch(path: Path, *, check: bool = False) -> int:
    if not path.exists():
        print(f"error: ReActor file not found: {path}")
        return 1
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        print(f"already patched: {path}")
        return 0
    missing = [
        name
        for name, snippet in (
            ("append", OLD_APPEND),
            ("empty check", OLD_EMPTY),
            ("swap loop", OLD_LOOP),
        )
        if snippet not in text
    ]
    if missing:
        print(
            "error: this ReActor version does not match the expected code "
            f"(missing: {', '.join(missing)})."
        )
        print(f"file: {path}")
        return 1
    if check:
        print(f"patch needed: {path}")
        return 2

    backup = path.with_suffix(".py.bak")
    if not backup.exists():
        backup.write_text(text, encoding="utf-8")
        print(f"backup: {backup}")

    text = text.replace(OLD_APPEND, NEW_APPEND, 1)
    text = text.replace(OLD_EMPTY, NEW_EMPTY, 1)
    text = text.replace(OLD_LOOP, NEW_LOOP, 1)
    path.write_text(text, encoding="utf-8")
    print(f"patched: {path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comfy-dir",
        type=Path,
        default=default_comfy_dir(),
        help="ComfyUI root (default: tgbot's COMFY_DIR)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="only report whether the patch is needed",
    )
    args = parser.parse_args()
    return patch(reactor_file(args.comfy_dir), check=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
