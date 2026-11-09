# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
"""CurioMesh in Spanish, Portuguese (Brazil), Japanese and Simplified Chinese.

Static text (bl_label, bl_description, text="...", property names and tooltips) goes through Blender's own
translation system: register() hands Blender one dict per language. Text built at run time ("12,345 faces") is
translated by tr(): an exact match first, then the templates ("{0} faces") made from the f-strings in the code by
translations/catalog.py, whose values are translated again when they are text themselves. Each language file is a
plain dict {English: translation}; no network, no extra dependencies."""

from __future__ import annotations

import re
import string
import textwrap
import unicodedata

from . import es, ja_JP, pt_BR, zh_HANS

try:
    import bpy
except ImportError:  # unit tests run without Blender
    bpy = None

# (Blender locale, name in that language). English is Blender's own text.
LANGUAGES = (
    ("en_US", "English"),
    ("es", "Español"),
    ("pt_BR", "Português (Brasil)"),
    ("ja_JP", "日本語"),
    ("zh_HANS", "简体中文"),
)
MESSAGES = {"es": es.MESSAGES, "pt_BR": pt_BR.MESSAGES, "ja_JP": ja_JP.MESSAGES, "zh_HANS": zh_HANS.MESSAGES}
LOWER_CASE = {"es", "pt_BR"}  # languages where "good for animation" keeps a small first letter mid-sentence
CONTEXTS = ("*", "Operator")  # button text of an operator is looked up in the "Operator" context
# Parts of a message translated one by one: "A | B" status lines and "A   B" key-hint bars.
SPLIT = re.compile(r"( \| |\s{2,})")

_cache: dict = {}
_patterns: dict = {}


def split_parts(text: str) -> list[str]:
    """The text and its separators, in order: 'A | B' -> ['A', ' | ', 'B']."""
    return SPLIT.split(text)


def fields(template: str) -> list[str]:
    """The placeholder names of a template, in order: 'Part {0} of {1}' -> ['0', '1']."""
    try:
        return [f for _lit, f, _spec, _conv in string.Formatter().parse(template) if f is not None]
    except ValueError:
        return []


def _compile(template: str):
    """A regex that matches the English text a template makes, one group per placeholder, or None."""
    parts, letters = [], 0
    try:
        parsed = list(string.Formatter().parse(template))
    except ValueError:
        return None
    if not any(f is not None for _l, f, _s, _c in parsed):
        return None
    for literal, field, _spec, _conv in parsed:
        parts.append(re.escape(literal))
        letters += sum(c.isalpha() for c in literal)
        if field is not None:
            parts.append(r"(.+?)")
    if letters < 3:  # "{0}: {1}" would match nearly anything
        return None
    return re.compile("".join(parts), re.S)


def _templates(lang: str) -> list:
    found = _patterns.get(lang)
    if found is None:
        found = []
        for key in MESSAGES[lang]:
            if "{" in key:
                rx = _compile(key)
                if rx is not None:
                    found.append((len(re.sub(r"\{[^}]*\}", "", key)), key, rx, fields(key)))
        found.sort(key=lambda t: -t[0])  # the most specific template wins
        _patterns[lang] = found
    return found


def _translate(lang: str, text: str, depth: int = 2) -> str:
    return _best(lang, text, depth)[1]


def _best(lang: str, text: str, depth: int) -> tuple[int, str]:
    """(how many letters of the text were translated, the translation). When several templates fit, the one that
    translates the most wins: "Laying rows on piece 3 of 8, about 1 min left" is "{0}, about {1} left" around
    "Laying rows on piece {0} of {1}", not that template with "8, about 1 min left" as its last value."""
    table = MESSAGES[lang]
    hit = table.get(text)
    if hit:
        return len(text), hit
    stripped = text.strip()
    if stripped != text:
        if not stripped:
            return 0, text
        score, hit = _best(lang, stripped, depth)
        return score, text.replace(stripped, hit) if score else text
    if SPLIT.search(text):
        pieces = split_parts(text)
        if len(pieces) > 1:
            done = [(0, p) if i % 2 else _best(lang, p, depth) for i, p in enumerate(pieces)]
            return sum(d[0] for d in done), "".join(d[1] for d in done)
    best, best_joins = (0, text), 0
    for size, key, rx, names in _templates(lang):
        m = rx.fullmatch(text)
        if m is None:
            continue
        out = table.get(key)
        if not out:
            continue
        scored = [_best(lang, g, depth - 1) if depth > 0 and any(c.isalpha() for c in g) else (0, g) for g in m.groups()]
        args = {}
        for name, value in zip(names, scored):
            args.setdefault(name, value)
        try:
            made = out.format(*[args.get(str(i), (0, ""))[1] for i in range(len(names))])
        except (IndexError, KeyError, ValueError):
            continue
        score = size + sum(v[0] for v in args.values())
        joins = sum(g.count(", ") + g.count(". ") + g.count("; ") for g in m.groups())  # values are seldom whole clauses
        if (score, -joins) > (best[0], -best_joins):
            best, best_joins = (score, made), joins
    if best[0]:
        return best
    if text[:1].islower():  # "good for animation": the Use names are capitalised in the tables
        for guess in (text[:1].upper() + text[1:], text.title()):
            hit = table.get(guess)
            if hit:
                return len(text), hit.lower() if lang in LOWER_CASE else hit
    return 0, text


def language(tooltip: bool = False) -> str | None:
    """The CurioMesh language in use now, or None for English (or when Blender translates nothing)."""
    if bpy is None:
        return None
    try:
        view = bpy.context.preferences.view
        if not (view.use_translate_tooltips if tooltip else view.use_translate_interface):
            return None
        locale = bpy.app.translations.locale
    except Exception:
        return None
    if locale in MESSAGES:
        return locale
    return next((key for key in MESSAGES if locale.startswith(key)), None)


def tr(text, lang: str | None = None, tooltip: bool = False):
    """Text built at run time in the user's language. Non-text and unknown text come back unchanged."""
    if not text or not isinstance(text, str):
        return text
    lang = lang or language(tooltip)
    if lang is None:
        return text
    key = (lang, text)
    hit = _cache.get(key)
    if hit is None:
        if len(_cache) > 20000:
            _cache.clear()
        hit = _cache[key] = _translate(lang, text)
    return hit


def tip(text, lang: str | None = None):
    """tr() for tooltips and operator descriptions: follows Blender's Tooltips switch."""
    return tr(text, lang, tooltip=True)


def cell(ch: str) -> int:
    """How many Latin letters wide a character draws: 2 for Chinese and Japanese, 1 for the rest."""
    return 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1


def text_width(text: str) -> int:
    return sum(cell(c) for c in text)


def clip(text: str, width: int) -> str:
    """The start of the text that fits in `width` Latin letters."""
    used = 0
    for i, ch in enumerate(text):
        used += cell(ch)
        if used > width:
            return text[:i]
    return text


NO_LINE_START = set("。、，．・）」』】！？：；…ー")


def wrap(text: str, width: int) -> list[str]:
    """Lines of at most `width` Latin letters. Text without wide characters wraps at spaces as before;
    Chinese and Japanese, which have no spaces, break between characters."""
    text = str(text)
    if not any(cell(c) == 2 for c in text):
        return textwrap.wrap(text, width=width)
    lines, line, used = [], "", 0
    for ch in text:
        w = cell(ch)
        if used + w > width and line and ch not in NO_LINE_START:
            cut = line.rfind(" ")
            if cut > len(line) // 2 and all(cell(c) == 1 for c in line[cut:]):  # a Latin word at the end: move it down whole
                lines.append(line[:cut].rstrip())
                line = line[cut + 1:]
            else:
                lines.append(line.rstrip())
                line = ""
            used = text_width(line)
            if ch == " ":
                continue
        line += ch
        used += w
    if line.strip():
        lines.append(line.rstrip())
    return lines


def blender_dict() -> dict:
    """{locale: {(context, English): translation}} for bpy.app.translations.register()."""
    out = {}
    for lang, table in MESSAGES.items():
        out[lang] = {(ctx, src): dst for src, dst in table.items() if dst and "{" not in src for ctx in CONTEXTS}
    return out


def register():
    try:
        bpy.app.translations.unregister(__package__)
    except Exception:
        pass
    bpy.app.translations.register(__package__, blender_dict())
    _cache.clear()


def unregister():
    try:
        bpy.app.translations.unregister(__package__)
    except Exception:
        pass
    _cache.clear()


# --------------------------------------------------------------------------- the Language switch in CurioMesh

OTHER = "OTHER"
LANGUAGE_ITEMS = [(code, name, "") for code, name in LANGUAGES] + [(OTHER, "Other", "Another language picked in Blender's Preferences")]
LANGUAGE_TIP = ("Blender's interface language. Picking one here changes the language of all of Blender, not only "
                "CurioMesh (Preferences > Interface > Translation), and turns on translated interface and tooltips")


def get_language(_self) -> int:
    try:
        current = bpy.context.preferences.view.language
    except Exception:
        return 0
    codes = [code for code, _name in LANGUAGES]
    if current in codes:
        return codes.index(current)
    return len(codes)  # "Other": Automatic, French and so on


def set_language(_self, value: int) -> None:
    if not 0 <= value < len(LANGUAGES):
        return  # "Other" leaves Blender's choice alone
    view = bpy.context.preferences.view
    try:
        view.language = LANGUAGES[value][0]
    except (TypeError, ValueError):  # a Blender build without that language
        return
    view.use_translate_interface = True
    view.use_translate_tooltips = True
    _cache.clear()
