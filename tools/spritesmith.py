"""
spritesmith.py — author game artwork from the command line.

A worked example of the dependency rule this engine is built around: a tool
that produces finished sprites, importing ``visual_ai`` and the standard
library and nothing else. NumPy arrives as ``visual_ai.np``, image I/O through
``visual_ai.imaging``. Swap the engine's backend and this file does not change.

    spritesmith cast --out assets
        Render the built-in cast — five plainly-drawn birds — as front and
        side PNGs named the way games expect: ruby.png, ruby_side.png, ...

    spritesmith new dusk --body oval --colour 90,110,180 --out specs/dusk.json
        Write a spec you can hand-edit. Every field is a number or a colour;
        there is no drawing code to touch.

    spritesmith render specs/dusk.json --out assets
        Spec to PNGs.

    spritesmith import photo.png --out assets/dusk_side.png
        Cut the background out of artwork from anywhere — a render, a
        generated image, a scan — and emit a clean transparent sprite.

    spritesmith sheet assets --out preview.png
        Contact sheet of a folder, on a checkerboard so alpha is visible.

    spritesmith list
        Available body shapes and the built-in cast.

Generating artwork elsewhere? Ask for a solid, uniform backdrop — "flat vector
style, side profile, on a solid #FF00FF magenta background, no shadow" — and
``import`` will key it out losslessly. Cutting a photographic background needs
``rembg``, which is optional; run ``spritesmith list`` to see whether it is
installed here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# The engine normally lives one directory up from tools/. Adding it here means
# the tool runs from a source checkout without an install step.
_ENGINE_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
if os.path.isdir(_ENGINE_SRC) and _ENGINE_SRC not in sys.path:
    sys.path.insert(0, os.path.normpath(_ENGINE_SRC))

try:
    import visual_ai as va
except ImportError as exc:  # pragma: no cover — install/layout problem
    sys.exit(f"could not import visual_ai ({exc}).\n"
             f"Expected the engine at: {os.path.normpath(_ENGINE_SRC)}")

np = va.np  # the engine re-exports it; the tool takes it from there


# ── Shared helpers ────────────────────────────────────────────────────────────

def parse_colour(text: str) -> tuple[int, int, int]:
    """Accept ``r,g,b`` or ``#rrggbb``."""
    text = text.strip()
    if text.startswith("#"):
        digits = text[1:]
        if len(digits) != 6:
            raise argparse.ArgumentTypeError(f"expected #rrggbb, got {text!r}")
        try:
            return tuple(int(digits[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore
        except ValueError:
            raise argparse.ArgumentTypeError(f"not a hex colour: {text!r}")

    parts = text.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"expected 'r,g,b' or '#rrggbb', got {text!r}")
    try:
        values = [int(p) for p in parts]
    except ValueError:
        raise argparse.ArgumentTypeError(f"non-numeric colour: {text!r}")
    if not all(0 <= v <= 255 for v in values):
        raise argparse.ArgumentTypeError(f"colour out of range 0-255: {text!r}")
    return tuple(values)  # type: ignore


def load_spec(path: str) -> va.CreatureSpec:
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    try:
        return va.spec_from_dict(data)
    except (ValueError, TypeError) as exc:
        raise SystemExit(f"{path}: {exc}")


def write_views(spec: va.CreatureSpec, out_dir: str, size: int) -> list[str]:
    """
    Emit ``{name}.png`` and ``{name}_side.png``.

    That naming is not arbitrary — the games already resolve artwork by exactly
    these filenames, so dropping the output into a game's assets folder needs
    no code change on the game side.
    """
    written = []
    for view in va.VIEWS:
        suffix = "" if view == "front" else "_side"
        path = os.path.join(out_dir, f"{spec.name}{suffix}.png")
        va.save_png(path, va.render_creature(spec, view=view, size=size))
        written.append(path)
    return written


def checkerboard(height: int, width: int, square: int = 16) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    shade = np.where(((yy // square + xx // square) % 2) == 0, 240, 205).astype(np.uint8)
    return np.dstack([shade] * 3 + [np.full((height, width), 255, np.uint8)])


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_cast(args) -> int:
    os.makedirs(args.out, exist_ok=True)
    total = 0
    for spec in va.DEFAULT_CAST:
        for path in write_views(spec, args.out, args.size):
            print(f"  wrote {path}")
            total += 1
    print(f"\n{total} sprites from {len(va.DEFAULT_CAST)} specs -> {args.out}")
    return 0


def cmd_new(args) -> int:
    spec = va.CreatureSpec(name=args.name, body=args.body)
    changes = {}
    if args.colour:
        changes["colour"] = args.colour
    if args.belly:
        changes["belly"] = args.belly
    if args.beak:
        changes["beak"] = args.beak
    if args.scale is not None:
        changes["scale"] = args.scale
    if args.tail is not None:
        changes["tail"] = args.tail
    if changes:
        spec = spec.variant(**changes)

    out = args.out or f"{args.name}.json"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(va.spec_to_dict(spec), handle, indent=2)
        handle.write("\n")
    print(f"  wrote {out}")
    print(f"  edit it, then: spritesmith render {out} --out assets")
    return 0


def cmd_render(args) -> int:
    os.makedirs(args.out, exist_ok=True)
    for spec_path in args.specs:
        spec = load_spec(spec_path)
        for path in write_views(spec, args.out, args.size):
            print(f"  wrote {path}")
    return 0


def cmd_import(args) -> int:
    try:
        image = va.load_image(args.image)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"could not read {args.image}: {exc}")

    uniformity = va.imaging.background_uniformity(image)
    print(f"  source     {image.shape[1]}x{image.shape[0]}")
    print(f"  border     {uniformity:.0%} uniform"
          f"{'  (chroma key will work well)' if uniformity > 0.75 else ''}")

    try:
        sprite = va.clean_sprite(image, mode=args.mode, size=args.size,
                                 key=args.key, tolerance=args.tolerance)
    except RuntimeError as exc:
        raise SystemExit(str(exc))

    coverage = float((sprite[..., 3] > 8).mean())
    if coverage > 0.97:
        print("  warning    almost nothing was cut — is the background really "
              "uniform?")
    elif coverage < 0.02:
        print("  warning    almost everything was cut — try a larger "
              "--tolerance")

    va.save_png(args.out, sprite)
    print(f"  wrote      {args.out}  ({sprite.shape[1]}x{sprite.shape[0]}, "
          f"{coverage:.0%} opaque)")
    return 0


def cmd_sheet(args) -> int:
    names = sorted(f for f in os.listdir(args.directory) if f.endswith(".png"))
    if not names:
        raise SystemExit(f"no PNGs in {args.directory}")

    cell = args.size
    columns = max(1, args.columns)
    rows = (len(names) + columns - 1) // columns
    sheet = np.zeros((rows * cell, columns * cell, 4), dtype=np.uint8)

    for index, name in enumerate(names):
        try:
            tile = va.pad_to(va.load_image(os.path.join(args.directory, name)), cell)
        except (ValueError, RuntimeError) as exc:
            print(f"  skipped {name}: {exc}")
            continue
        r, c = divmod(index, columns)
        sheet[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = tile

    height, width = sheet.shape[:2]
    flat = va.imaging.composite_over(sheet, checkerboard(height, width))
    va.save_png(args.out, flat)
    print(f"  wrote {args.out}  ({len(names)} sprites, {columns}x{rows})")
    return 0


def cmd_list(_args) -> int:
    print("\n  body shapes")
    for shape in va.BODY_SHAPES:
        print(f"    {shape}")

    print("\n  built-in cast")
    for spec in va.DEFAULT_CAST:
        r, g, b = spec.colour
        print(f"    {spec.name.ljust(8)} {spec.body.ljust(6)} "
              f"#{r:02x}{g:02x}{b:02x}  scale {spec.scale:.2f}")

    print("\n  background removal")
    print("    chroma key   always available — best for flat art on a solid "
          "backdrop")
    status = "installed" if va.REMBG_AVAILABLE else 'not installed'
    print(f"    rembg        {status} — for backgrounds you could not control")
    if not va.REMBG_AVAILABLE:
        print('                 install with: pip install "rembg[cpu]"')
    print()
    return 0


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spritesmith",
        description="Author game sprites through the visual_ai engine.",
    )
    sub = parser.add_subparsers(dest="command")

    p_cast = sub.add_parser("cast", help="render the built-in cast")
    p_cast.add_argument("--out", default="assets", help="output directory")
    p_cast.add_argument("--size", type=int, default=192, help="sprite size (px)")
    p_cast.set_defaults(func=cmd_cast)

    p_new = sub.add_parser("new", help="write a spec JSON to edit")
    p_new.add_argument("name")
    p_new.add_argument("--body", default="round", choices=list(va.BODY_SHAPES))
    p_new.add_argument("--colour", "--color", dest="colour", type=parse_colour)
    p_new.add_argument("--belly", type=parse_colour)
    p_new.add_argument("--beak", type=parse_colour)
    p_new.add_argument("--scale", type=float)
    p_new.add_argument("--tail", type=float, help="side-view tail length, 0 for none")
    p_new.add_argument("--out", help="spec path (default <name>.json)")
    p_new.set_defaults(func=cmd_new)

    p_render = sub.add_parser("render", help="spec JSON -> front and side PNGs")
    p_render.add_argument("specs", nargs="+")
    p_render.add_argument("--out", default="assets", help="output directory")
    p_render.add_argument("--size", type=int, default=192)
    p_render.set_defaults(func=cmd_render)

    p_import = sub.add_parser("import", help="any image -> clean transparent PNG")
    p_import.add_argument("image")
    p_import.add_argument("--out", required=True)
    p_import.add_argument("--mode", default="auto",
                          choices=["auto", "chroma", "rembg", "none"])
    p_import.add_argument("--key", type=parse_colour,
                          help="backdrop colour (default: measured from border)")
    p_import.add_argument("--tolerance", type=float, default=42.0,
                          help="RGB distance treated as background")
    p_import.add_argument("--size", type=int,
                          help="also centre on a square canvas of this size")
    p_import.set_defaults(func=cmd_import)

    p_sheet = sub.add_parser("sheet", help="contact sheet of a folder of PNGs")
    p_sheet.add_argument("directory")
    p_sheet.add_argument("--out", default="sheet.png")
    p_sheet.add_argument("--size", type=int, default=160, help="cell size (px)")
    p_sheet.add_argument("--columns", type=int, default=4)
    p_sheet.set_defaults(func=cmd_sheet)

    p_list = sub.add_parser("list", help="show shapes, cast, and backend status")
    p_list.set_defaults(func=cmd_list)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
