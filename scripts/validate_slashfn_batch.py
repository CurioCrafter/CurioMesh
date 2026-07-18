"""Independent, deterministic validation for the SlashFn Atlas-Q batch.

Run this script with the same Blender release used to save the production file.
The baseline pass fingerprints the untouched SlashFn source meshes.  The final
pass rechecks those fingerprints, then validates every retopology output using
CurioMesh's production diagnostics rather than trusting batch-report claims.

Examples::

    blender pristine.blend --background --python-exit-code 1 \
      --python scripts/validate_slashfn_batch.py -- \
      --mode baseline --report artifacts/slashfn-baseline.json

    blender finished.blend --background --python-exit-code 1 \
      --python scripts/validate_slashfn_batch.py -- \
      --mode final --baseline-report artifacts/slashfn-baseline.json \
      --report artifacts/slashfn-final-validation.json
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import traceback
from typing import Any

import bpy
import numpy as np


DEFAULT_SOURCE_COLLECTION = "SlashFn"
DEFAULT_OUTPUT_COLLECTION = "SlashFn_AtlasQ_Retopology"
DEFAULT_WORK_COLLECTION = "_CurioMesh_SlashFn_Component_Work"
DEFAULT_EXPECTED_SOURCES = 22
SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(
        description="Fingerprint SlashFn sources or independently validate an Atlas-Q batch."
    )
    parser.add_argument("--mode", choices=("baseline", "final"), required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path)
    parser.add_argument("--source-collection", default=DEFAULT_SOURCE_COLLECTION)
    parser.add_argument("--output-collection", default=DEFAULT_OUTPUT_COLLECTION)
    parser.add_argument("--work-collection", default=DEFAULT_WORK_COLLECTION)
    parser.add_argument("--expected-sources", type=int, default=DEFAULT_EXPECTED_SOURCES)
    parser.add_argument(
        "--max-output-quads",
        type=int,
        default=999,
        help="Maximum allowed quad faces per output; may not exceed 999.",
    )
    parser.add_argument(
        "--expected-texture-resolution",
        type=int,
        default=512,
        help="Required square packed base-color texture resolution; must be below 1000.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="CurioMesh package root containing __init__.py.",
    )
    args = parser.parse_args(argv)
    if args.mode == "final" and args.baseline_report is None:
        parser.error("--baseline-report is required in final mode")
    if args.expected_sources <= 0:
        parser.error("--expected-sources must be positive")
    if not 1 <= args.max_output_quads <= 999:
        parser.error("--max-output-quads must be between 1 and 999")
    if not 1 <= args.expected_texture_resolution < 1000:
        parser.error("--expected-texture-resolution must be between 1 and 999")
    return args


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_array(values: np.ndarray, dtype: str) -> str:
    """Hash an array with an explicit little-endian dtype and shape header."""

    array = np.ascontiguousarray(np.asarray(values, dtype=np.dtype(dtype)))
    header = canonical_json({"dtype": array.dtype.str, "shape": list(array.shape)})
    digest = hashlib.sha256()
    digest.update(struct.pack("<Q", len(header)))
    digest.update(header)
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def find_collection(name: str) -> bpy.types.Collection | None:
    direct = bpy.data.collections.get(name)
    if direct is not None:
        return direct
    folded = name.casefold()
    matches = [item for item in bpy.data.collections if item.name.casefold() == folded]
    if len(matches) == 1:
        return matches[0]
    return None


def recursive_objects(collection: bpy.types.Collection) -> list[bpy.types.Object]:
    found: dict[int, bpy.types.Object] = {}

    def visit(current: bpy.types.Collection) -> None:
        for obj in current.objects:
            found[obj.as_pointer()] = obj
        for child in current.children:
            visit(child)

    visit(collection)
    return sorted(found.values(), key=lambda obj: (obj.name.casefold(), obj.name))


def recursive_meshes(collection: bpy.types.Collection) -> list[bpy.types.Object]:
    return [
        obj
        for obj in recursive_objects(collection)
        if obj.type == "MESH" and obj.data is not None and len(obj.data.polygons) > 0
    ]


def foreach_array(
    data: Any,
    attribute: str,
    count: int,
    width: int,
    dtype: str,
) -> np.ndarray:
    values = np.empty(count * width, dtype=np.dtype(dtype))
    if values.size:
        data.foreach_get(attribute, values)
    return values.reshape((count, width)) if width > 1 else values


def uv_layer_record(mesh: bpy.types.Mesh, layer: Any, index: int) -> dict[str, Any]:
    count = len(layer.data)
    try:
        values = foreach_array(layer.data, "uv", count, 2, "<f4")
    except Exception:
        values = np.asarray([tuple(item.uv) for item in layer.data], dtype="<f4")
    finite = bool(np.isfinite(values).all())
    return {
        "name": layer.name,
        "index": index,
        "active": bool(mesh.uv_layers.active == layer),
        "active_render": bool(getattr(layer, "active_render", False)),
        "active_clone": bool(getattr(layer, "active_clone", False)),
        "value_count": count,
        "expected_value_count": len(mesh.loops),
        "finite": finite,
        "values_sha256": sha256_array(values, "<f4"),
    }


def color_domain_size(mesh: bpy.types.Mesh, domain: str) -> int | None:
    return {
        "POINT": len(mesh.vertices),
        "EDGE": len(mesh.edges),
        "FACE": len(mesh.polygons),
        "CORNER": len(mesh.loops),
    }.get(domain)


def color_layer_record(mesh: bpy.types.Mesh, attribute: Any, index: int) -> dict[str, Any]:
    count = len(attribute.data)
    try:
        values = foreach_array(attribute.data, "color", count, 4, "<f4")
    except Exception:
        values = np.asarray([tuple(item.color) for item in attribute.data], dtype="<f4")
    record = {
        "name": attribute.name,
        "index": index,
        "domain": attribute.domain,
        "data_type": attribute.data_type,
        "value_count": count,
        "expected_value_count": color_domain_size(mesh, attribute.domain),
        "finite": bool(np.isfinite(values).all()),
        "values_sha256": sha256_array(values, "<f4"),
    }

    # BYTE_COLOR is exposed through converted float values.  Hashing the sRGB
    # view as well makes accidental byte-value changes independently visible.
    try:
        srgb_values = foreach_array(attribute.data, "color_srgb", count, 4, "<f4")
    except Exception:
        srgb_values = None
    if srgb_values is not None:
        record["srgb_values_sha256"] = sha256_array(srgb_values, "<f4")
        record["srgb_finite"] = bool(np.isfinite(srgb_values).all())
    return record


def source_mesh_record(obj: bpy.types.Object) -> dict[str, Any]:
    """Return stable field digests for one raw Blender mesh object."""

    mesh = obj.data
    vertices = foreach_array(mesh.vertices, "co", len(mesh.vertices), 3, "<f4")
    edges = foreach_array(mesh.edges, "vertices", len(mesh.edges), 2, "<i4")
    loop_vertices = foreach_array(mesh.loops, "vertex_index", len(mesh.loops), 1, "<i4")
    loop_edges = foreach_array(mesh.loops, "edge_index", len(mesh.loops), 1, "<i4")
    loop_starts = foreach_array(mesh.polygons, "loop_start", len(mesh.polygons), 1, "<i4")
    loop_totals = foreach_array(mesh.polygons, "loop_total", len(mesh.polygons), 1, "<i4")
    material_indices = foreach_array(
        mesh.polygons,
        "material_index",
        len(mesh.polygons),
        1,
        "<i4",
    )
    polygon_layout = np.column_stack((loop_starts, loop_totals, material_indices)).astype(
        "<i4",
        copy=False,
    )
    polygon_payload = np.concatenate(
        (
            polygon_layout.reshape(-1),
            loop_vertices.reshape(-1),
            loop_edges.reshape(-1),
        )
    )
    matrix_world = np.asarray(
        [value for row in obj.matrix_world for value in row],
        dtype="<f8",
    ).reshape((4, 4))
    material_names = [material.name if material is not None else None for material in mesh.materials]
    uv_layers = [uv_layer_record(mesh, layer, index) for index, layer in enumerate(mesh.uv_layers)]
    color_layers = [
        color_layer_record(mesh, attribute, index)
        for index, attribute in enumerate(mesh.color_attributes)
    ]
    fingerprints = {
        "vertices_sha256": sha256_array(vertices, "<f4"),
        "edges_sha256": sha256_array(edges, "<i4"),
        "polygon_loops_materials_sha256": sha256_array(polygon_payload, "<i4"),
        "matrix_world_sha256": sha256_array(matrix_world, "<f8"),
        "material_names_sha256": sha256_bytes(canonical_json(material_names)),
        "uv_layers_sha256": sha256_bytes(canonical_json(uv_layers)),
        "color_attributes_sha256": sha256_bytes(canonical_json(color_layers)),
    }
    stable = {
        "object": obj.name,
        "mesh": mesh.name,
        "vertices": len(mesh.vertices),
        "edges": len(mesh.edges),
        "loops": len(mesh.loops),
        "polygons": len(mesh.polygons),
        "materials": material_names,
        "uv_layers": uv_layers,
        "color_attributes": color_layers,
        "fingerprints": fingerprints,
    }
    stable["content_sha256"] = sha256_bytes(canonical_json(stable))
    return stable


def source_manifest(collection: bpy.types.Collection) -> dict[str, Any]:
    meshes = recursive_meshes(collection)
    records = [source_mesh_record(obj) for obj in meshes]
    aggregate = {
        "objects": len(records),
        "vertices": sum(item["vertices"] for item in records),
        "edges": sum(item["edges"] for item in records),
        "loops": sum(item["loops"] for item in records),
        "polygons": sum(item["polygons"] for item in records),
    }
    aggregate["manifest_sha256"] = sha256_bytes(
        canonical_json([(item["object"], item["content_sha256"]) for item in records])
    )
    return {"aggregate": aggregate, "objects": records}


def named_records(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {record["object"]: record for record in records}


def compare_source_manifest(
    baseline: dict[str, Any],
    current: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    baseline_records = named_records(baseline.get("objects", []))
    current_records = named_records(current.get("objects", []))
    baseline_names = set(baseline_records)
    current_names = set(current_records)
    if baseline_names != current_names:
        errors.append(
            "Source object names differ from baseline: "
            f"missing={sorted(baseline_names - current_names)}, "
            f"unexpected={sorted(current_names - baseline_names)}"
        )
    for name in sorted(baseline_names & current_names, key=str.casefold):
        expected = baseline_records[name]
        actual = current_records[name]
        if expected.get("content_sha256") == actual.get("content_sha256"):
            continue
        differing_fields = []
        expected_fingerprints = expected.get("fingerprints", {})
        actual_fingerprints = actual.get("fingerprints", {})
        for field in sorted(set(expected_fingerprints) | set(actual_fingerprints)):
            if expected_fingerprints.get(field) != actual_fingerprints.get(field):
                differing_fields.append(field)
        for field in ("mesh", "vertices", "edges", "loops", "polygons"):
            if expected.get(field) != actual.get(field):
                differing_fields.append(field)
        errors.append(
            f"Source {name!r} differs from baseline in: "
            + (", ".join(differing_fields) if differing_fields else "stable metadata")
        )
    expected_digest = baseline.get("aggregate", {}).get("manifest_sha256")
    current_digest = current.get("aggregate", {}).get("manifest_sha256")
    if expected_digest != current_digest and not errors:
        errors.append("Source manifest digest differs from baseline")
    return errors


def load_curiomesh(repo_root: Path) -> tuple[Any, Any]:
    root = repo_root.resolve()
    initializer = root / "__init__.py"
    if not initializer.is_file():
        raise RuntimeError(f"CurioMesh package initializer not found: {initializer}")
    for module_name in list(sys.modules):
        if module_name == "curiomesh" or module_name.startswith("curiomesh."):
            del sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(
        "curiomesh",
        initializer,
        submodule_search_locations=[str(root)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not construct CurioMesh import spec for {root}")
    package = importlib.util.module_from_spec(spec)
    sys.modules["curiomesh"] = package
    spec.loader.exec_module(package)
    metrics = importlib.import_module("curiomesh.metrics")
    triadq = importlib.import_module("curiomesh.triadq")
    return metrics, triadq


def normalized_topology(obj: bpy.types.Object, triadq: Any) -> dict[str, Any]:
    mesh = obj.data
    coordinates = foreach_array(mesh.vertices, "co", len(mesh.vertices), 3, "<f8")
    faces = [tuple(int(index) for index in polygon.vertices) for polygon in mesh.polygons]
    mesh_data = triadq.MeshData(vertices=coordinates, faces=faces, name=obj.name)
    normalized, fan_splits = triadq.split_non_manifold_vertex_fans(mesh_data)
    summary = triadq.topology_summary(normalized)
    return {
        "vertex_fan_splits": int(fan_splits),
        "connected_components": int(summary.connected_components),
        "component_signature": [list(item) for item in summary.component_signature],
        "non_manifold_edges": int(summary.non_manifold_edges),
        "non_manifold_vertices": int(summary.non_manifold_vertices),
        "branched_boundary_vertices": int(summary.branched_boundary_vertices),
        "inconsistent_edges": int(summary.inconsistent_edges),
        "duplicate_faces": int(summary.duplicate_faces),
    }


def uv_structure(obj: bpy.types.Object) -> dict[str, Any]:
    mesh = obj.data
    active = mesh.uv_layers.active.name if mesh.uv_layers.active is not None else None
    return {
        "names": sorted(layer.name for layer in mesh.uv_layers),
        "active": active,
        "layers": [
            {
                "name": layer.name,
                "values": len(layer.data),
                "expected_values": len(mesh.loops),
                "finite": uv_layer_record(mesh, layer, index)["finite"],
            }
            for index, layer in enumerate(mesh.uv_layers)
        ],
    }


def color_structure(obj: bpy.types.Object) -> list[dict[str, Any]]:
    mesh = obj.data
    return [
        {
            "name": attribute.name,
            "domain": attribute.domain,
            "data_type": attribute.data_type,
            "values": len(attribute.data),
            "expected_values": color_domain_size(mesh, attribute.domain),
            "finite": color_layer_record(mesh, attribute, index)["finite"],
        }
        for index, attribute in enumerate(mesh.color_attributes)
    ]


def material_structure(obj: bpy.types.Object) -> dict[str, Any]:
    names = [material.name for material in obj.data.materials if material is not None]
    invalid_polygons = sum(
        1
        for polygon in obj.data.polygons
        if polygon.material_index < 0
        or polygon.material_index >= len(obj.data.materials)
        or obj.data.materials[polygon.material_index] is None
    )
    return {"names": sorted(set(names)), "invalid_polygon_assignments": invalid_polygons}


def id_property_string_list(value: Any) -> list[str]:
    """Normalize JSON strings, ID-property arrays, and plain sequences."""

    decoded = value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            decoded = [value]
    if isinstance(decoded, str):
        return [decoded]
    if isinstance(decoded, (list, tuple)):
        return [str(item) for item in decoded]
    try:
        return [str(item) for item in decoded.to_list()]
    except (AttributeError, TypeError):
        return []


def image_pixel_record(image: bpy.types.Image) -> dict[str, Any]:
    width = int(image.size[0]) if len(image.size) >= 1 else 0
    height = int(image.size[1]) if len(image.size) >= 2 else 0
    pixel_count = len(image.pixels)
    pixels = np.empty(pixel_count, dtype="<f4")
    if pixel_count:
        image.pixels.foreach_get(pixels)
    finite = bool(pixel_count > 0 and np.isfinite(pixels).all())
    rgb = pixels.reshape((-1, 4))[:, :3] if pixel_count and pixel_count % 4 == 0 else None
    return {
        "name": image.name,
        "width": width,
        "height": height,
        "channels": int(image.channels),
        "pixel_values": pixel_count,
        "expected_rgba_values": width * height * 4,
        "packed": image.packed_file is not None,
        "finite": finite,
        "pixels_sha256": sha256_array(pixels, "<f4"),
        "coverage": (
            float(np.mean(np.any(rgb > 1e-4, axis=1))) if rgb is not None and len(rgb) else 0.0
        ),
        "dynamic_range": (
            float(np.max(rgb) - np.min(rgb)) if rgb is not None and len(rgb) else 0.0
        ),
    }


def validate_base_color_bake(
    output: bpy.types.Object,
    source_material_names: list[str],
    expected_resolution: int,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    labeled_nodes: list[tuple[bpy.types.Material, Any]] = []
    seen_materials: set[int] = set()
    for material in output.data.materials:
        if material is None or material.as_pointer() in seen_materials:
            continue
        seen_materials.add(material.as_pointer())
        if not material.use_nodes or material.node_tree is None:
            continue
        for node in material.node_tree.nodes:
            if node.type == "TEX_IMAGE" and node.label == "CurioMesh_BaseColor":
                labeled_nodes.append((material, node))

    if len(labeled_nodes) != 1:
        errors.append(
            "expected exactly one TEX_IMAGE node labeled 'CurioMesh_BaseColor', "
            f"found {len(labeled_nodes)}"
        )

    linked_nodes = []
    referenced_images: dict[int, bpy.types.Image] = {}
    node_records = []
    for material, node in labeled_nodes:
        image = node.image
        if image is not None:
            referenced_images[image.as_pointer()] = image
        linked = any(
            link.from_node == node
            and link.from_socket.name == "Color"
            and link.to_node.type == "BSDF_PRINCIPLED"
            and link.to_socket.name == "Base Color"
            for link in material.node_tree.links
        )
        if linked:
            linked_nodes.append(node)
        else:
            errors.append(
                f"base-color node {node.name!r} in material {material.name!r} is not "
                "linked directly to a Principled Base Color input"
            )
        if image is None:
            errors.append(
                f"base-color node {node.name!r} in material {material.name!r} has no image"
            )
        node_records.append(
            {
                "material": material.name,
                "node": node.name,
                "label": node.label,
                "image": image.name if image is not None else None,
                "linked_to_principled_base_color": linked,
            }
        )

    images = list(referenced_images.values())
    if len(images) != 1:
        errors.append(f"expected exactly one referenced baked image, found {len(images)}")
    image_records = [image_pixel_record(image) for image in images]
    for image_record in image_records:
        dimensions = (image_record["width"], image_record["height"])
        if dimensions != (expected_resolution, expected_resolution):
            errors.append(
                f"baked image {image_record['name']!r} is {dimensions[0]}x{dimensions[1]}, "
                f"expected {expected_resolution}x{expected_resolution}"
            )
        if max(dimensions) >= 1000:
            errors.append(
                f"baked image {image_record['name']!r} violates the sub-1000 dimension limit"
            )
        if not image_record["packed"]:
            errors.append(f"baked image {image_record['name']!r} is not packed")
        if not image_record["finite"]:
            errors.append(f"baked image {image_record['name']!r} has non-finite or empty pixels")
        if image_record["pixel_values"] != image_record["expected_rgba_values"]:
            errors.append(
                f"baked image {image_record['name']!r} does not contain one RGBA value per pixel"
            )

    source_provenance = id_property_string_list(
        output.get("curiomesh_bake_source_materials")
    )
    image_provenance = id_property_string_list(output.get("curiomesh_bake_images"))
    try:
        resolution_provenance = int(output.get("curiomesh_bake_resolution", -1))
    except (TypeError, ValueError):
        resolution_provenance = -1
    if "Atlantis_400k" not in source_provenance:
        errors.append("curiomesh_bake_source_materials does not contain 'Atlantis_400k'")
    missing_source_materials = sorted(set(source_material_names) - set(source_provenance))
    if missing_source_materials:
        errors.append(
            f"bake source-material provenance is missing {missing_source_materials}"
        )
    if resolution_provenance != expected_resolution:
        errors.append(
            f"curiomesh_bake_resolution is {resolution_provenance}, expected {expected_resolution}"
        )
    referenced_names = sorted(image.name for image in images)
    if sorted(image_provenance) != referenced_names:
        errors.append(
            f"curiomesh_bake_images {sorted(image_provenance)} does not match "
            f"the shader image {referenced_names}"
        )

    record = {
        "nodes": node_records,
        "images": image_records,
        "provenance": {
            "source_materials": source_provenance,
            "resolution": resolution_provenance,
            "images": image_provenance,
        },
        "linked_node_count": len(linked_nodes),
        "valid": not errors,
        "errors": errors,
    }
    return record, errors


def structure_keys(structure: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
    return sorted(
        (item["name"], item["domain"], item["data_type"])
        for item in structure
    )


def validate_one_output(
    source: bpy.types.Object,
    output: bpy.types.Object,
    metrics: Any,
    triadq: Any,
    max_output_quads: int,
    expected_texture_resolution: int,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    source_topology = normalized_topology(source, triadq)
    diagnostics = metrics.mesh_diagnostics(output, 0)
    diagnostic_record = asdict(diagnostics)
    diagnostic_record["component_signature"] = [list(item) for item in diagnostics.component_signature]
    diagnostic_record["component_euler"] = list(diagnostics.component_euler)

    zero_defect_fields = (
        "non_manifold_edges",
        "non_manifold_vertices",
        "branched_boundary_vertices",
        "inconsistent_edges",
        "duplicate_faces",
        "degenerate_faces",
        "invalid_quads",
    )
    nonzero_defects = {
        field: int(getattr(diagnostics, field))
        for field in zero_defect_fields
        if int(getattr(diagnostics, field)) != 0
    }
    if not diagnostics.quad_only:
        errors.append("output is not certified quad-only")
    if not 1 <= diagnostics.faces <= max_output_quads:
        errors.append(
            f"output has {diagnostics.faces} quad faces; required range is "
            f"1..{max_output_quads}"
        )
    if nonzero_defects:
        errors.append(f"output has topology/quality defects {nonzero_defects}")
    if not diagnostics.uv_valid:
        errors.append("output UV mapping is invalid")
    expected_signature = source_topology["component_signature"]
    actual_signature = [list(item) for item in diagnostics.component_signature]
    if actual_signature != expected_signature:
        errors.append(
            f"normalized component signature mismatch: expected {expected_signature}, "
            f"got {actual_signature}"
        )
    if source_topology["non_manifold_edges"] or source_topology["non_manifold_vertices"]:
        errors.append(
            "source vertex-fan normalization did not produce manifold topology: "
            f"edges={source_topology['non_manifold_edges']}, "
            f"vertices={source_topology['non_manifold_vertices']}"
        )

    mapped_source = output.get("curiomesh_batch_source")
    if mapped_source != source.name:
        errors.append(f"curiomesh_batch_source is {mapped_source!r}, expected {source.name!r}")
    if int(output.get("curiomesh_batch_source_faces", -1)) != len(source.data.polygons):
        errors.append("curiomesh_batch_source_faces does not match the source")
    if int(output.get("curiomesh_batch_component_count", -1)) != source_topology["connected_components"]:
        errors.append("curiomesh_batch_component_count does not match normalized topology")
    if output.get("curiomesh_batch_method") != "FACE_EDGE_COMPONENT_ATLAS_Q":
        errors.append("curiomesh_batch_method is missing or unexpected")

    source_materials = material_structure(source)
    output_materials = material_structure(output)
    if "Atlantis_400k" not in source_materials["names"]:
        errors.append("source object does not reference the required Atlantis_400k material")
    if not output_materials["names"]:
        errors.append("output has no material for the baked base color")
    if output_materials["invalid_polygon_assignments"]:
        errors.append("one or more output polygons have invalid material assignments")
    source_material_pointers = {
        material.as_pointer() for material in source.data.materials if material is not None
    }
    output_material_pointers = {
        material.as_pointer() for material in output.data.materials if material is not None
    }
    if source_material_pointers & output_material_pointers:
        errors.append("baked output still shares a material datablock with its source")

    bake, bake_errors = validate_base_color_bake(
        output,
        source_materials["names"],
        expected_texture_resolution,
    )
    errors.extend(bake_errors)

    source_uv = uv_structure(source)
    output_uv = uv_structure(output)
    if source_uv["names"] != output_uv["names"]:
        errors.append(
            f"UV layer names differ: expected {source_uv['names']}, got {output_uv['names']}"
        )
    if source_uv["active"] != output_uv["active"]:
        errors.append(
            f"active UV layer differs: expected {source_uv['active']!r}, got {output_uv['active']!r}"
        )
    for layer in output_uv["layers"]:
        if layer["values"] != layer["expected_values"] or not layer["finite"]:
            errors.append(f"output UV layer {layer['name']!r} has invalid structure or values")

    source_colors = color_structure(source)
    output_colors = color_structure(output)
    if structure_keys(source_colors) != structure_keys(output_colors):
        errors.append(
            "color attribute structure differs: "
            f"expected {structure_keys(source_colors)}, got {structure_keys(output_colors)}"
        )
    for attribute in output_colors:
        if attribute["values"] != attribute["expected_values"] or not attribute["finite"]:
            errors.append(
                f"output color attribute {attribute['name']!r} has invalid structure or values"
            )

    record = {
        "source": source.name,
        "source_faces": len(source.data.polygons),
        "source_normalized_topology": source_topology,
        "output": output.name,
        "output_faces": len(output.data.polygons),
        "source_mapping": mapped_source,
        "diagnostics": diagnostic_record,
        "materials": output_materials,
        "bake": bake,
        "uv": output_uv,
        "color_attributes": output_colors,
        "valid": not errors,
        "errors": errors,
    }
    return record, errors


def work_collections(name: str) -> list[str]:
    folded = name.casefold()
    dotted_prefix = folded + "."
    return sorted(
        collection.name
        for collection in bpy.data.collections
        if collection.name.casefold() == folded
        or collection.name.casefold().startswith(dotted_prefix)
    )


def validate_baseline(args: argparse.Namespace, report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    source_collection = find_collection(args.source_collection)
    if source_collection is None:
        return [f"Source collection {args.source_collection!r} was not found"]
    manifest = source_manifest(source_collection)
    report["source"] = manifest
    count = manifest["aggregate"]["objects"]
    if count != args.expected_sources:
        errors.append(f"Expected {args.expected_sources} source meshes, found {count}")
    object_names = [item["object"] for item in manifest["objects"]]
    if len(object_names) != len(set(object_names)):
        errors.append("Source object names are not unique")
    if find_collection(args.output_collection) is not None:
        errors.append(
            f"Baseline is not pristine: output collection {args.output_collection!r} exists"
        )
    stale_work = work_collections(args.work_collection)
    if stale_work:
        errors.append(f"Baseline is not pristine: work collections exist {stale_work}")
    return errors


def validate_final(args: argparse.Namespace, report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    baseline_path = args.baseline_report.resolve()
    with baseline_path.open("r", encoding="utf-8") as handle:
        baseline_report = json.load(handle)
    report["baseline_report"] = str(baseline_path)
    if baseline_report.get("schema_version") != SCHEMA_VERSION:
        errors.append("Baseline report schema is unsupported")
    if baseline_report.get("mode") != "baseline" or not baseline_report.get("valid"):
        errors.append("Baseline report is not a successful baseline validation")
    baseline_source = baseline_report.get("source")
    if not isinstance(baseline_source, dict):
        raise RuntimeError("Baseline report does not contain a source manifest")

    source_collection = find_collection(args.source_collection)
    if source_collection is None:
        raise RuntimeError(f"Source collection {args.source_collection!r} was not found")
    current_source = source_manifest(source_collection)
    report["source"] = current_source
    if current_source["aggregate"]["objects"] != args.expected_sources:
        errors.append(
            f"Expected {args.expected_sources} current source meshes, "
            f"found {current_source['aggregate']['objects']}"
        )
    errors.extend(compare_source_manifest(baseline_source, current_source))

    stale_work = work_collections(args.work_collection)
    report["work_collections"] = stale_work
    if stale_work:
        errors.append(f"Temporary component work collections remain: {stale_work}")

    output_collection = find_collection(args.output_collection)
    if output_collection is None:
        raise RuntimeError(f"Output collection {args.output_collection!r} was not found")
    output_objects = recursive_objects(output_collection)
    output_meshes = recursive_meshes(output_collection)
    report["output_object_count"] = len(output_objects)
    report["output_mesh_count"] = len(output_meshes)
    if len(output_objects) != args.expected_sources:
        errors.append(
            f"Expected exactly {args.expected_sources} total output objects, found {len(output_objects)}"
        )
    if len(output_meshes) != args.expected_sources:
        errors.append(
            f"Expected exactly {args.expected_sources} non-empty output meshes, found {len(output_meshes)}"
        )
    non_mesh_outputs = [obj.name for obj in output_objects if obj not in output_meshes]
    if non_mesh_outputs:
        errors.append(f"Output collection contains non-mesh or empty objects: {non_mesh_outputs}")

    outputs_by_source: dict[str, list[bpy.types.Object]] = {}
    for output in output_meshes:
        mapped = output.get("curiomesh_batch_source")
        key = str(mapped) if mapped is not None else ""
        outputs_by_source.setdefault(key, []).append(output)
    sources = {obj.name: obj for obj in recursive_meshes(source_collection)}
    for mapped, outputs in sorted(outputs_by_source.items()):
        if mapped not in sources:
            errors.append(
                f"Output mapping {mapped!r} has no source: {[item.name for item in outputs]}"
            )
    metrics, triadq = load_curiomesh(args.repo_root)
    output_records = []
    for source_name in sorted(sources, key=str.casefold):
        matches = outputs_by_source.get(source_name, [])
        if len(matches) != 1:
            errors.append(
                f"Source {source_name!r} has {len(matches)} mapped outputs; expected exactly one"
            )
            continue
        output = matches[0]
        if output.as_pointer() == sources[source_name].as_pointer():
            errors.append(f"Output for {source_name!r} is the source object itself")
            continue
        record, object_errors = validate_one_output(
            sources[source_name],
            output,
            metrics,
            triadq,
            args.max_output_quads,
            args.expected_texture_resolution,
        )
        output_records.append(record)
        errors.extend(f"{source_name}: {message}" for message in object_errors)
    report["outputs"] = output_records

    image_owners: dict[str, list[str]] = {}
    for record in output_records:
        for image_record in record.get("bake", {}).get("images", []):
            image_owners.setdefault(image_record["name"], []).append(record["source"])
    shared_bakes = {
        image_name: owners
        for image_name, owners in sorted(image_owners.items())
        if len(owners) != 1
    }
    report["baked_image_owners"] = image_owners
    if shared_bakes:
        errors.append(f"Baked base-color images are shared across outputs: {shared_bakes}")

    source_faces = current_source["aggregate"]["polygons"]
    output_faces = sum(item["output_faces"] for item in output_records)
    report["aggregate"] = {
        "source_faces": source_faces,
        "output_faces": output_faces,
        "face_reduction": source_faces - output_faces,
        "output_to_source_ratio": output_faces / max(1, source_faces),
    }
    if len(output_records) == args.expected_sources and output_faces >= source_faces:
        errors.append(
            f"Aggregate output faces ({output_faces}) are not lower than source faces ({source_faces})"
        )
    return errors


def write_report(path: Path, report: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "mode": args.mode,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "blender_version": bpy.app.version_string,
        "blend_file": bpy.data.filepath,
        "source_collection": args.source_collection,
        "output_collection": args.output_collection,
        "expected_sources": args.expected_sources,
        "validation_contract": {
            "max_output_quads": args.max_output_quads,
            "expected_texture_resolution": args.expected_texture_resolution,
            "maximum_texture_dimension_exclusive": 1000,
            "required_base_color_node_label": "CurioMesh_BaseColor",
            "required_source_material": "Atlantis_400k",
        },
        "valid": False,
        "errors": [],
    }
    caught: Exception | None = None
    try:
        if args.mode == "baseline":
            errors = validate_baseline(args, report)
        else:
            errors = validate_final(args, report)
        report["errors"] = errors
        report["valid"] = not errors
    except Exception as exc:
        caught = exc
        report["errors"] = [f"{type(exc).__name__}: {exc}"]
        report["traceback"] = traceback.format_exc()
        report["valid"] = False
    write_report(args.report, report)
    print(
        f"CURIO_SLASHFN_VALIDATION mode={args.mode} valid={report['valid']} "
        f"report={args.report.resolve()}"
    )
    if not report["valid"]:
        detail = "; ".join(report["errors"][:10])
        if caught is not None:
            raise RuntimeError(f"SlashFn validation failed: {detail}") from caught
        raise RuntimeError(f"SlashFn validation failed: {detail}")


if __name__ == "__main__":
    main()
