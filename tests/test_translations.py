# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
"""The four language tables: complete, same placeholders as the English, and tr() fills templates right."""

import collections
import string
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import translations as T  # noqa: E402
from translations import catalog  # noqa: E402


def placeholders(text: str):
    return collections.Counter(f"{{{f}}}" for _l, f, _s, _c in string.Formatter().parse(text) if f is not None)


class Translations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = catalog.collect()

    def test_four_languages(self):
        self.assertEqual(set(T.MESSAGES), {"es", "pt_BR", "ja_JP", "zh_HANS"})

    def test_same_placeholders_and_no_empty(self):
        for lang, table in T.MESSAGES.items():
            for src, dst in table.items():
                self.assertTrue(isinstance(dst, str) and dst.strip(), f"{lang}: empty translation of {src!r}")
                self.assertEqual(placeholders(src), placeholders(dst), f"{lang}: {src!r} -> {dst!r}")

    def test_every_msgid_in_every_language(self):
        # Every text the panels show (catalog.py reads it from the code) must be in all four tables.
        # New text? Run: python translations/catalog.py  (it lists what each language is missing)
        for lang, table in T.MESSAGES.items():
            gaps = catalog.missing(table, self.catalog)
            self.assertEqual(gaps, [], f"{lang} misses {len(gaps)} msgids, e.g. {gaps[:5]}")

    def test_blender_dict(self):
        d = T.blender_dict()
        self.assertEqual(d["es"][("*", "Start")], T.MESSAGES["es"]["Start"])
        self.assertEqual(d["es"][("Operator", "Start")], T.MESSAGES["es"]["Start"])  # button text of an operator
        self.assertFalse(any("{" in src for (_ctx, src) in d["ja_JP"]))  # templates are for tr(), not Blender

    def test_tr_templates(self):
        es = T.MESSAGES["es"]
        self.assertEqual(T.tr("12,345 faces", "es"), es["{0} faces"].format("12,345"))
        self.assertEqual(T.tr("Fix Bad Spots (3)", "es"), es["Fix Bad Spots ({0})"].format(3))
        # values are translated again: a stage inside "{0}, about {1} left"
        stage = T.tr("Laying rows on piece 3 of 8", "ja_JP")
        self.assertEqual(T.tr("Laying rows on piece 3 of 8, about 1 min 05 s left", "ja_JP"), T.MESSAGES["ja_JP"]["{0}, about {1} left"].format(stage, T.tr("1 min 05 s", "ja_JP")))
        # status lines and key hints are translated piece by piece
        self.assertEqual(T.tr("Click: select   Esc: done", "zh_HANS"), T.MESSAGES["zh_HANS"]["Click: select"] + "   " + T.MESSAGES["zh_HANS"]["Esc: done"])
        self.assertEqual(T.tr("A mesh called Bob", "es"), "A mesh called Bob")  # unknown text stays English
        self.assertEqual(T.tr("Start"), "Start")  # no Blender: English

    def test_wrap_wide_characters(self):
        text = T.MESSAGES["ja_JP"]["Click a mesh in the 3D view. CurioMesh makes a clean, lighter copy; yours is not changed until you press Apply."]
        lines = T.wrap(text, 36)
        self.assertGreater(len(lines), 1)
        self.assertTrue(all(T.text_width(line) <= 38 for line in lines), lines)  # one wide character more: a 。 or ー never starts a line
        self.assertEqual("".join(lines).replace(" ", ""), text.replace(" ", ""))
        self.assertEqual(T.wrap("plain words wrap at spaces", 12), ["plain words", "wrap at", "spaces"])
        self.assertEqual(T.text_width(T.clip("日本語のテキスト", 7)), 6)


if __name__ == "__main__":
    unittest.main()
