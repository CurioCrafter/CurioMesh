"""Build docs/trailer/trailer.webp and docs/images/hero_styles.jpg from existing pictures.

Run from the repo root:  python scripts/make_trailer.py [path/to/godot_export/sheet_user.png]
Swap the source pictures below when better results arrive, then re-run.
"""
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / "docs" / "images"
OUT = ROOT / "docs" / "trailer"
SHEET = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent / "bench" / "engine" / "godot_export" / "sheet_user.png"
# small copy kept in the repo so the script re-runs without the bench folder
SHEET_COPY = OUT / "game_asset_sheet.jpg"

BG, ACC, TXT, DIM = (27, 29, 34), (245, 166, 35), (235, 236, 238), (150, 155, 165)
W, H = 1280, 720
FONT = "C:/Windows/Fonts/segoeui.ttf"
FONTB = "C:/Windows/Fonts/segoeuib.ttf"


def font(size, bold=False):
    try:
        return ImageFont.truetype(FONTB if bold else FONT, size)
    except OSError:
        return ImageFont.load_default(size)


def canvas(w=W, h=H):
    im = Image.new("RGB", (w, h), BG)
    return im, ImageDraw.Draw(im)


def center(d, text, y, size, fill=TXT, bold=False, w=W):
    f = font(size, bold)
    d.text(((w - d.textlength(text, font=f)) / 2, y), text, font=f, fill=fill)


def crop(path, box, width):
    im = Image.open(path).convert("RGB").crop(box)
    return im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)


def sheet_rows():
    OUT.mkdir(exist_ok=True)
    if SHEET.exists():
        s = Image.open(SHEET).convert("RGB")
        s.save(SHEET_COPY, quality=85)
    else:
        s = Image.open(SHEET_COPY).convert("RGB")
    return [s.crop((0, a, 1024, b)) for a, b in ((15, 240), (275, 485), (535, 745))]


# ---------- trailer scenes ----------
def title():
    im, d = canvas()
    center(d, "CurioMesh", 230, 110, ACC, True)
    center(d, "Free quad retopology for Blender", 370, 46)
    center(d, "Sculpts, scans and game assets. GPL-3.0-or-later.", 450, 26, DIM)
    return im


def before_after(path, title_text, sub):
    im, d = canvas()
    center(d, title_text, 40, 44, ACC, True)
    im.paste(crop(path, (0, 54, 1400, 540), 1200), (40, 150))
    d.text((40, 118), "Original", font=font(24), fill=DIM)
    d.text((640, 118), "CurioMesh result", font=font(24), fill=DIM)
    center(d, sub, 600, 28)
    return im


def choose():
    im, d = canvas()
    center(d, "Choose your style", 30, 48, ACC, True)
    cols = [("Clean Rows", "even quads for sculpts and scans", "bust_rows.jpg"),
            ("Balanced", "tidy layout, more faces", "bust_balanced.jpg"),
            ("Keep Detail", "fewest faces, quads follow the shape", "old_military_compressor.jpg")]
    for i, (name, use, f) in enumerate(cols):
        x = 40 + i * 405
        im.paste(crop(IMG / f, (705, 54, 1400, 540), 390), (x, 140))
        d.text((x, 440), name, font=font(36, True), fill=TXT)
        d.text((x, 490), use, font=font(21), fill=DIM)
    center(d, "Pick what matters more: even rows or fewest faces.", 620, 26)
    return im


def game_asset():
    rows = sheet_rows()
    im, d = canvas()
    center(d, "Game asset: a baked normal map, lit in Godot", 28, 40, ACC, True)
    labels = [("Original", "47,214 faces"), ("No normal map", "1,068 faces"),
              ("CurioMesh export", "1,068 faces + normal map")]
    for i, (r, (a, b)) in enumerate(zip(rows, labels)):
        y = 115 + i * 195
        im.paste(r.resize((780, round(r.height * 780 / 1024)), Image.LANCZOS), (460, y))
        d.text((40, y + 50), a, font=font(30, True), fill=TXT)
        d.text((40, y + 92), b, font=font(21), fill=DIM)
    d.text((460, 690), "The light moves from left to right.", font=font(20), fill=DIM)
    return im


def export():
    im, d = canvas()
    center(d, "Export for your game engine", 150, 50, ACC, True)
    center(d, "Unreal   Unity   Godot   Roblox", 290, 60)
    center(d, "FBX or glTF, normal map in the format each engine expects", 400, 28, DIM)
    return im


def endcard():
    im, d = canvas()
    center(d, "Free and open source", 170, 56, TXT, True)
    center(d, "If it saves you time, star it on GitHub", 280, 36, ACC)
    center(d, "github.com/CurioCrafter/CurioMesh", 380, 44, TXT, True)
    center(d, "Blender 3.6 to 5.2. Runs on your PC.", 480, 26, DIM)
    return im


def build_trailer():
    scenes = [(title(), 2.5),
              (before_after(IMG / "bust_rows.jpg", "Clean Rows: a scan or sculpt", "17,456 faces to 14,147 even quads"), 3.0),
              (before_after(IMG / "barrier_rows.jpg", "Clean Rows: a photo scan", "60,928 faces to 2,486 quads"), 3.0),
              (choose(), 3.5),
              (game_asset(), 4.5),
              (export(), 2.5),
              (endcard(), 3.5)]
    frames, durs = [], []
    fade_n, fade_ms = 5, 70
    for i, (im, hold) in enumerate(scenes):
        if i:
            for k in range(1, fade_n + 1):
                frames.append(Image.blend(scenes[i - 1][0], im, k / (fade_n + 1)))
                durs.append(fade_ms)
        frames.append(im)
        durs.append(int(hold * 1000))
    OUT.mkdir(exist_ok=True)
    out = OUT / "trailer.webp"
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=durs, loop=0, quality=72, method=6, kmax=1)
    print(out, round(out.stat().st_size / 1e6, 2), "MB,", round(sum(durs) / 1000, 1), "s")


# ---------- hero ----------
def build_hero():
    rows = sheet_rows()
    im, d = canvas(1400, 800)
    center(d, "Clean quads for sculpts, scans and games", 14, 38, ACC, True, 1400)
    pw = 440

    def lion():
        # three rows (original, no normal map, exported), two light positions each, scaled to fit the panel
        tile = Image.new("RGB", (pw, 628), BG)
        for k, r in enumerate(rows):
            c = r.crop((40, 0, 480, 205)).resize((396, 180), Image.LANCZOS)
            tile.paste(c, (22, 22 + k * 202))
            d2 = ImageDraw.Draw(tile)
            d2.text((4, 2 + k * 202), ("Original", "No normal map", "With normal map")[k], font=font(17, True), fill=ACC)
        return tile

    panels = [
        ("Clean Rows", "Even quads for sculpts and scans", "14,147 quads",
         crop(IMG / "bust_rows.jpg", (705, 54, 1400, 540), pw), crop(IMG / "bust_rows.jpg", (705, 548, 1400, 1040), pw)),
        ("Clean Rows", "Even quads for scans and hard surfaces", "2,486 quads",
         crop(IMG / "barrier_rows.jpg", (705, 54, 1400, 540), pw), crop(IMG / "barrier_rows.jpg", (705, 548, 1400, 1040), pw)),
        ("Game asset", "47,214 faces to 1,068, lit in Godot", None, lion(), None),
    ]
    for i, (name, use, tag, a, b) in enumerate(panels):
        x = 40 + i * 465
        im.paste(a, (x, 75))
        if b is not None:
            im.paste(b, (x, 75 + a.height + 6))
            d.text((x + 6, 80), tag, font=font(18, True), fill=ACC)
        y = 75 + 628 + 14
        d.text((x, y), name, font=font(30, True), fill=TXT)
        d.text((x, y + 42), use, font=font(21), fill=DIM)
    out = IMG / "hero_styles.jpg"
    im.save(out, quality=88)
    print(out, round(out.stat().st_size / 1e3), "KB")


if __name__ == "__main__":
    build_hero()
    build_trailer()
