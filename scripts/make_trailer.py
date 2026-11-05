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


# ---------- the choice: fewest faces <-> even rows, same shape ----------
STOPS = [("Keep Detail", "2,508 quads", "marble_bust_01.jpg"),
         ("Balanced", "3,920 quads", "bust_balanced.jpg"),
         ("Clean Rows", "14,147 quads", "bust_rows.jpg")]
LEFT, RIGHT = "Fewest faces: game assets, LODs, far objects", "Even rows: editing, subdividing, animating"


def slider(d, x0, x1, y, t, w):
    """A scroll bar from fewest faces (t=0) to even rows (t=1), with a knob at t."""
    d.rounded_rectangle([x0, y - 5, x1, y + 5], 5, fill=(70, 74, 82))
    d.rounded_rectangle([x0, y - 5, x0 + (x1 - x0) * t, y + 5], 5, fill=ACC)
    for k, (name, _faces, _f) in enumerate(STOPS):
        x = x0 + (x1 - x0) * k / (len(STOPS) - 1)
        d.ellipse([x - 6, y - 6, x + 6, y + 6], fill=ACC if k / (len(STOPS) - 1) <= t + 1e-6 else (70, 74, 82))
        f = font(20, True)
        d.text((x - d.textlength(name, font=f) / 2, y + 16), name, font=f, fill=TXT)
    kx = x0 + (x1 - x0) * t
    d.ellipse([kx - 14, y - 14, kx + 14, y + 14], fill=TXT, outline=ACC, width=4)


def stop_image(k, w, close):
    box = (705, 548, 1400, 1040) if close else (705, 54, 1400, 540)
    return crop(IMG / STOPS[k][2], box, w)


def choice_frame(t):
    """One trailer frame with the knob at t: the result blends between the stops either side."""
    im, d = canvas()
    center(d, "You choose: fewest faces or even rows", 22, 40, ACC, True)
    center(d, "Same bust, same shape: about 6% of the view changed at every setting", 76, 22, DIM)
    pos = t * (len(STOPS) - 1)
    a = min(int(pos), len(STOPS) - 2)
    u = pos - a
    for j, close in enumerate((False, True)):
        pic = Image.blend(stop_image(a, 560, close), stop_image(a + 1, 560, close), u)
        im.paste(pic, (60 + j * 600, 120))
    k = round(pos)
    center(d, f"{STOPS[k][0]}: {STOPS[k][1]}", 530, 30, TXT, True)
    slider(d, 160, 1120, 600, t, W)
    d.text((60, 660), "← " + LEFT, font=font(19), fill=DIM)
    f = font(19)
    d.text((W - 60 - d.textlength(RIGHT + " →", font=f), 660), RIGHT + " →", font=f, fill=DIM)
    return im


def choice_frames():
    """The knob rests on each stop, slides between them, then slides back to fewest faces."""
    out = []
    path = [0.0, 0.5, 1.0, 0.0]
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        out.append((choice_frame(a), 1400))
        steps = 10 if abs(b - a) <= 0.5 else 14
        out += [(choice_frame(a + (b - a) * s / steps), 60) for s in range(1, steps)]
    out.append((choice_frame(0.0), 1200))
    return out


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
              (before_after(IMG / "marble_bust_01.jpg", "A 17,456-face sculpt in 2,508 quads", "Keep Detail: the fewest faces that keep the shape"), 3.0),
              (choice_frames(), None),
              (before_after(IMG / "barrier_rows.jpg", "Clean Rows: a photo scan", "60,928 faces to 2,486 quads"), 3.0),
              (choose(), 3.5),
              (game_asset(), 4.5),
              (export(), 2.5),
              (endcard(), 3.5)]
    frames, durs = [], []
    fade_n, fade_ms = 5, 70
    last = None
    for im, hold in scenes:
        seq = im if hold is None else [(im, int(hold * 1000))]  # a scene is one still or a run of frames
        if last is not None:
            for k in range(1, fade_n + 1):
                frames.append(Image.blend(last, seq[0][0], k / (fade_n + 1)))
                durs.append(fade_ms)
        for f, ms in seq:
            frames.append(f)
            durs.append(ms)
        last = seq[-1][0]
    OUT.mkdir(exist_ok=True)
    out = OUT / "trailer.webp"
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=durs, loop=0, quality=72, method=6, kmax=1)
    print(out, round(out.stat().st_size / 1e6, 2), "MB,", round(sum(durs) / 1000, 1), "s")


# ---------- the face picture: original, fewest faces, even rows, and the bar between ----------
def build_choice():
    im, d = canvas(1400, 860)
    center(d, "Same sculpt, your choice: fewest faces or even rows", 14, 36, ACC, True, 1400)
    pw, gap, x0 = 430, 20, 25
    cols = [("Original", "17,456 triangles", None), ("Keep Detail", "2,508 quads", 0), ("Clean Rows", "14,147 quads", 2)]
    mids = []
    for i, (name, faces, k) in enumerate(cols):
        x = x0 + i * (pw + gap)
        src = IMG / ("marble_bust_01.jpg" if k is None else STOPS[k][2])
        left = (0, 54, 695, 540) if k is None else (705, 54, 1400, 540)
        lower = (0, 548, 695, 1040) if k is None else (705, 548, 1400, 1040)
        a, b = crop(src, left, pw), crop(src, lower, pw)
        im.paste(a, (x, 70))
        im.paste(b, (x, 70 + a.height + 6))
        ty = 70 + a.height + b.height + 16
        d.text((x, ty), name, font=font(28, True), fill=TXT)
        d.text((x + d.textlength(name, font=font(28, True)) + 14, ty + 8), faces, font=font(20), fill=DIM)
        mids.append(x + pw / 2)
    y = 818
    xa, xb = mids[1], mids[2]  # a two-way bar from fewest faces to even rows
    d.rounded_rectangle([xa, y - 4, xb, y + 4], 4, fill=ACC)
    for xx in (xa, xb):
        d.ellipse([xx - 10, y - 10, xx + 10, y + 10], fill=TXT, outline=ACC, width=3)
    f = font(19, True)
    d.text((xa + 18, y - 34), "fewest faces", font=f, fill=ACC)
    d.text((xb - 18 - d.textlength("even rows", font=f), y - 34), "even rows", font=f, fill=ACC)
    d.text((x0, y - 12), "Both keep about 6% view change", font=font(19), fill=DIM)
    out = IMG / "hero_choice.jpg"
    im.save(out, quality=88)
    print(out, round(out.stat().st_size / 1e3), "KB")


def build_choice_anim():
    """The README's first picture: the knob scrolls from fewest faces to even rows and back, the result follows."""
    seq = choice_frames()
    out = IMG / "choice.webp"
    seq[0][0].save(out, save_all=True, append_images=[f for f, _ in seq[1:]], duration=[ms for _, ms in seq], loop=0, quality=75, method=6)
    print(out, round(out.stat().st_size / 1e6, 2), "MB")


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
    build_choice()
    build_choice_anim()
    build_hero()
    build_trailer()
