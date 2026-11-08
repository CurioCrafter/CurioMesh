# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
"""Plain-words help for every step of the panel. `help_button(layout, topic)` draws a small "?" that pops the text up."""

try:
    import bpy
except ImportError:  # unit tests run without Blender
    bpy = None

WIDTH = 44  # characters per popup line

# topic -> (title, text). Titles are the real panel and button names.
HELP = {
    "empty": ("Start here", "Click a mesh in the 3D view. CurioMesh makes a clean, lighter copy next to it. Your mesh is not changed until you press Apply."),
    "dashboard": ("Dashboard", "A small box in the corner of the 3D view with the quad count, quality and live progress. It stays off until you press Show Dashboard."),
    "use": ("What Is It For", "Pick what the new mesh is for: a game, animation, a scan clean-up and so on. Each choice sets many options for you. Not sure? Pick Auto and Start will choose."),
    "detail": ("Detail", "How much of the shape to keep. Higher keeps more small shapes and uses more quads. Lower gives a lighter, simpler mesh."),
    "shape_flow": ("Shape the Flow", "Guide which way the quad rows run, for example along a limb or round an eye. Skip it if the first result already looks right."),
    "options": ("Options", "Less common choices: symmetry, hard edges, output type and more. The defaults work for most meshes."),
    "mirror": ("Mirror", "Make the left and right halves match. Pick the axis the model is symmetric across."),
    "start": ("Start", "Builds the new mesh. Nothing in your scene changes yet. You can watch it work and press Stop at any time."),
    "working": ("Working", "CurioMesh is building the result. A long model can take minutes. Blender stays usable. Stop keeps what is done so far."),
    "verdict": ("Result check", "A short verdict on the new mesh: how many quads, how much the look changed and whether any faces are bad. Green is good to go, yellow and red say what to look at."),
    "adjust": ("Adjust", "Move a slider to see another version right away. Nothing is lost until you Apply, and Ctrl+Z steps back."),
    "finetune": ("Fine-tune", "Small changes to the result: tidy the rows, even the sizes, open up poles. Run a tool again on its own result to keep improving it."),
    "tools": ("Tools", "Each tool shows its change on the mesh first. Click Apply to keep it, Esc to cancel. Run it again while it keeps helping."),
    "fix": ("Fix", "Go through the parts that need a look. Mark the bad ones, then redo them with smaller or bigger quads, or keep the original."),
    "part_edit": ("Edit Parts", "Merge, split, lock or rename the parts of the result. Locked parts stay as they are when you redo others."),
    "part_finder": ("Part Finder", "Find parts by what is wrong with them, such as folded or crossing faces, and select them."),
    "better": ("Help Make It Better", "Found a spot CurioMesh gets wrong, or fixed one yourself? Press Pick Bad Spots, click those spots in the 3D view, then Send as a Problem or Share a Fix. CurioMesh saves a zip and opens the GitHub form; nothing is sent until you attach the zip and post it. Code and test help: How to Help."),
    "finish": ("Finish", "Write the result into your scene. Then bake textures or export a file if you need to."),
    "bake": ("Bake", "Copy the colours and detail of your original onto the new mesh as textures."),
    "lod": ("LOD Set", "Make a few lighter copies of the result for a game, each with fewer faces than the last."),
    "export": ("Export", "Save the result as a file for a game engine or another program."),
    "history": ("History", "Every Apply is a step in the list. Jump back to any earlier step, or press Ctrl+Z."),
    "undo": ("Undo", "Ctrl+Z undoes the last thing you did, Ctrl+Shift+Z does it again. Nothing is lost by trying."),
    "pro": ("Pro", "Every setting and the classic tools. You never need these for a good result."),
    "engine": ("Engine", "Which method builds the quads. Auto picks for you. Change it only if you know what you want."),
    "computer": ("This Computer", "How much of this computer CurioMesh may use, and the switch that lets an AI agent drive CurioMesh."),
    "shapes": ("Inside Shapes", "Tell CurioMesh which parts of the model are separate shapes, such as eyes or bolts, so each gets its own clean rows."),
    "teach": ("Show It How", "Draw a few example strokes. CurioMesh follows their direction in the result."),
    "reports": ("Issue Reports", "Select the faces that have problems and save a report you can share."),
    "classic": ("Classic Guided", "The older step-by-step tools, kept for people who like them."),
    "advanced": ("Advanced", "Fine controls for the search itself. Most people never open this."),
    "prompt": ("Write a Prompt for My AI", "Pick your goal and press the button. CurioMesh copies a ready prompt for your AI agent, with your model and settings filled in. Nothing is sent anywhere."),
    "agent": ("Let an Agent Drive CurioMesh", "Lets an AI agent send commands through a folder on this computer. Files only, no network. It is off until you turn it on."),
}


def text(topic: str) -> str:
    title, body = HELP.get(topic, ("Help", "No help for this step yet."))
    return f"{title}: {body}"


def _wrap(layout, body: str) -> None:
    import textwrap

    for line in textwrap.wrap(body, WIDTH):
        layout.label(text=line)


if bpy is not None:

    class CURIOMESH_OT_help(bpy.types.Operator):
        bl_idname = "curiomesh.help"
        bl_label = "What Is This?"
        bl_description = "Explain this step in plain words"
        bl_options = {"INTERNAL"}

        topic: bpy.props.StringProperty(default="empty", options={"SKIP_SAVE"})

        def execute(self, context):
            title, body = HELP.get(self.topic, ("Help", "No help for this step yet."))

            def draw(menu, _context):
                _wrap(menu.layout, body)

            context.window_manager.popup_menu(draw, title=title, icon="QUESTION")
            return {"FINISHED"}

    CLASSES = (CURIOMESH_OT_help,)
else:
    CLASSES = ()


def help_button(layout, topic: str) -> None:
    """A small "?" with no frame; put it at the end of a step's title row."""
    layout.operator("curiomesh.help", text="", icon="QUESTION", emboss=False).topic = topic


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
