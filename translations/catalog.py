# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
"""Every English text the CurioMesh panels can show, read from the code (no Blender needed).

    python translations/catalog.py            prints the text a language file is missing (all languages)
    python translations/catalog.py --json F   writes every msgid to F (for translators)

Static strings are msgids as they are. An f-string or a "+" of strings becomes a template with {0}, {1}... where the
values go ("{0} faces"); "s"/"" plural endings make one template each. Text is split at " | " and at runs of spaces,
the same way tr() splits it. Log lines, errors, identifiers, icon names and file names are left out."""

from __future__ import annotations

import ast
import itertools
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from translations import SPLIT  # noqa: E402

# The files whose text people see. Left out: ui.py and operators.py (the classic Pro tools), the engines' internals
# and the agent command line.
SKIP_FILES = {
    "ui.py", "operators.py", "turbo.py", "atlasq.py", "bridge.py", "agentcmd.py", "agentops_parts.py", "agentops_run.py",
    "agentops_tools.py", "rows_worker.py", "parallel.py", "metrics.py", "jobpack.py", "partpack.py", "native.py",
}
# Text that never gets past the filter on its own: the dashboard's capital-letter labels.
EXTRA = ("CURIOMESH", "FACES", "TRIS", "POLES", "WORKING", "ORIGINAL", "CLEAN", "MIXED", "ROUGH", "Pro", "Hide", "Stop", "done", "quads")

IGNORE_CALLS = {
    "print", "log", "_log", "debug", "info", "warning", "warn", "error", "exception", "critical", "getattr", "setattr",
    "hasattr", "delattr", "open", "startswith", "endswith", "split", "rsplit", "replace", "strip", "lstrip", "rstrip",
    "Path", "compile", "match", "search", "sub", "fullmatch", "findall", "encode", "decode", "dumps", "loads", "environ",
    "getenv", "get_icon", "kw", "icon", "register_class", "property_unset", "is_property_set", "keys", "path", "glob",
    "mkdir", "exists", "joinpath", "with_suffix", "undo_push", "frombuffer", "dtype", "astype", "template_list", "time",
    "_dbg", "trace", "note_debug", "enum_items", "rna_type", "is_registered", "get_prefs", "object", "new", "remove",
    "vertex_groups", "uv_layers", "attributes", "_attr", "foreach_get", "foreach_set", "tag",
}
IGNORE_KW = {
    "icon", "idname", "bl_idname", "type", "mode", "kind", "panel", "operator", "options", "subtype", "unit", "data_path",
    "sep", "end", "encoding", "key", "default", "propname", "list_id", "translation_context", "dtype", "file", "suffix",
    "name_prefix", "prefix", "attr", "domain", "layer", "group", "filter_glob", "search", "search_property", "rows",
    "text_ctxt", "icon_value", "align", "alignment", "space_type", "region_type", "role", "name_key",
}
CAP = 16


def texty(text: str) -> bool:
    """True for words a person reads; False for identifiers, icon names, paths and format codes."""
    bare = re.sub(r"\{\d+\}", "", text).strip()
    if sum(c.isalpha() for c in bare) < 2:
        return False
    if any(s in text for s in ("\\", "://", ".py", ".json", ".blend", ".zip", ".npz", "%s", "%d", "__", "{}", "\n\n\n")):
        return False
    if re.search(r"[a-z0-9]_[a-z0-9]|[A-Z]_[A-Z]", bare):  # snake_case and ICON_NAMES
        return False
    if " " not in bare:
        if re.fullmatch(r"[A-Z][a-z]+[a-z0-9]*[.!?:]*|[A-Z][a-z]+-[A-Za-z]+|[A-Z][a-z]+\+?", bare):
            return True  # "Start", "Auto", "Fine-tune"
        return bool(re.search(r"[a-z]", bare)) and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.\-]*", bare) or "{" in text and bool(re.search(r"[a-z]{2}", bare))
    return True


def _name(func) -> str:
    return func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else ""


def _escape(s: str) -> str:
    return s.replace("{", "{{").replace("}", "}}")


def shapes(node) -> list[list]:
    """The texts an expression can make: lists of literal pieces, with None where a value goes."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [[node.value]]
    if isinstance(node, ast.JoinedStr):
        options = []
        for part in node.values:
            if isinstance(part, ast.Constant):
                options.append([[part.value]])
            elif isinstance(part.value, ast.IfExp) and all(isinstance(b, ast.Constant) and isinstance(b.value, str) and len(b.value) <= 4 for b in (part.value.body, part.value.orelse)):
                options.append([[part.value.body.value], [part.value.orelse.value]])  # spot{'s' if n != 1 else ''}
            else:
                options.append([[None]])
        return _product(options)
    if isinstance(node, ast.IfExp):
        return (shapes(node.body) + shapes(node.orelse))[:CAP]
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
        return [s for v in node.values for s in shapes(v)][:CAP]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _product([shapes(node.left), shapes(node.right)])
    return [[None]]


def _product(options) -> list[list]:
    out = []
    for combo in itertools.product(*options):
        out.append([p for piece in combo for p in piece])
        if len(out) >= CAP:
            break
    return out


def templates(pieces: list) -> list[str]:
    """Template msgids of one shape, split like tr() splits text."""
    text, n = "", 0
    for p in pieces:
        if p is None:
            text += "\x00"
        else:
            text += _escape(p)
    if "\x00" not in text and not text.strip():
        return []
    out = []
    for i, part in enumerate(SPLIT.split(text)):
        if i % 2:
            continue
        part = part.strip()
        n = 0
        while "\x00" in part:
            part = part.replace("\x00", "{%d}" % n, 1)
            n += 1
        if part and part.replace("{0}", "") and texty(part):
            out.append(part)
    return out


def _is_string_shape(node) -> bool:
    return isinstance(node, (ast.JoinedStr, ast.IfExp, ast.BoolOp, ast.BinOp)) or isinstance(node, ast.Constant) and isinstance(node.value, str)


def _component(node, parent) -> bool:
    if isinstance(parent, ast.JoinedStr) or isinstance(parent, ast.FormattedValue):
        return True
    if isinstance(parent, ast.IfExp):
        return node is not parent.test
    if isinstance(parent, ast.BinOp):
        return isinstance(parent.op, ast.Add)
    if isinstance(parent, ast.BoolOp):
        return isinstance(parent.op, ast.Or)
    return False


def _skipped(node, parents) -> bool:
    child = node
    for parent in parents:
        if isinstance(parent, (ast.Raise, ast.Assert, ast.Compare, ast.Subscript, ast.Import, ast.ImportFrom)):
            if not isinstance(parent, ast.Subscript) or child is parent.slice:
                return True
        if isinstance(parent, ast.Expr) and isinstance(child, ast.Constant):
            return True  # a docstring
        if isinstance(parent, ast.Dict) and child in parent.keys:
            return True
        if isinstance(parent, ast.keyword) and parent.arg in IGNORE_KW:
            return True
        if isinstance(parent, ast.Call):
            name = _name(parent.func)
            if name in IGNORE_CALLS and child is not parent.func:
                return True
            if name in {"prop", "prop_enum", "operator", "popover", "menu", "prop_menu_enum", "template_icon", "prop_search"} and child in parent.args:
                return True  # layout.prop(data, "name"): the second argument is a property name
        if isinstance(parent, ast.Attribute):
            return True  # "x".join(...)
        child = parent
    return False


def collect_file(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    parents: dict = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    found = set()
    for node in ast.walk(tree):
        if not _is_string_shape(node):
            continue
        parent = parents.get(node)
        if _component(node, parent):
            continue  # part of a bigger text: that one makes the template
        chain, p = [], parent
        while p is not None:
            chain.append(p)
            p = parents.get(p)
        if _skipped(node, chain):
            continue
        shaped = shapes(node)
        if not any(any(isinstance(x, str) and x.strip() for x in s) for s in shaped):
            continue
        for s in shaped:
            found.update(templates(s))
    return found


def sources(root: Path = ROOT) -> list[Path]:
    return [p for p in sorted(root.glob("*.py")) if p.name not in SKIP_FILES] + [root / "translations" / "__init__.py"]


def collect(root: Path = ROOT) -> dict[str, list[str]]:
    """{msgid: [files it is in]}."""
    out: dict = {}
    for path in sources(root):
        for msg in collect_file(path):
            out.setdefault(msg, []).append(path.name)
    for msg in EXTRA:
        out.setdefault(msg, ["(dashboard)"])
    return dict(sorted(out.items()))


def missing(table: dict, catalog: dict | None = None) -> list[str]:
    catalog = catalog or collect()
    return [m for m in catalog if not table.get(m)]


if __name__ == "__main__":
    cat = collect()
    if "--json" in sys.argv:
        import json

        Path(sys.argv[sys.argv.index("--json") + 1]).write_text(json.dumps(list(cat), ensure_ascii=False, indent=0), encoding="utf-8")
        print(len(cat), "msgids")
    else:
        from translations import MESSAGES

        for lang, table in MESSAGES.items():
            gaps = missing(table, cat)
            print(f"{lang}: {len(gaps)} missing of {len(cat)}")
            for m in gaps[:40]:
                print("   ", repr(m))
