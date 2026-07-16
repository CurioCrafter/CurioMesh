from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import math
from typing import Any

import bpy
from mathutils import Vector

from .metrics import MeshDiagnostics, mesh_diagnostics
from .triadq import component_topology_preserved


@dataclass(slots=True)
class ScanProfile:
    category: str
    recommended_strategy: str
    face_count: int
    vertex_count: int
    sampled_faces: int
    exact_topology: bool
    topology_basis: str
    boundary_edges: int
    boundary_loops: int
    connected_components: int
    component_euler: tuple[int, ...]
    component_signature: tuple[tuple[int, int], ...]
    non_manifold_edges: int
    non_manifold_vertices: int
    branched_boundary_vertices: int
    inconsistent_edges: int
    duplicate_faces: int
    euler_characteristic: int | None
    degenerate_ratio: float
    sharp_ratio: float
    triangle_ratio: float
    target_edge_length: float
    diagonal: float
    texture_critical: bool
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def summary(self) -> str:
        qualifier = {
            "SOURCE_EXACT": "exact source",
            "PROXY_EXACT": "exact proxy (source sampled)",
        }.get(self.topology_basis, "sampled")
        return (
            f"{self.category}: {self.face_count:,} faces; {qualifier} analysis; "
            f"recommended {self.recommended_strategy}."
        )


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    strategy: str
    label: str
    seed: int
    repair: bool = False
    preserve_sharp: bool = True
    preserve_boundary: bool = True
    pair_closure: bool = False


@dataclass(slots=True)
class CandidateEvaluation:
    strategy: str
    label: str
    valid: bool
    score: float
    reason: str
    faces: int
    quad_ratio: float
    target_error: float
    geometry_rms: float
    geometry_p95: float
    geometry_max: float
    min_scaled_jacobian: float
    p95_aspect_ratio: float
    extraordinary_ratio: float
    elapsed_ms: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key, value in payload.items():
            if isinstance(value, float) and not math.isfinite(value):
                payload[key] = None
        return payload


@dataclass(frozen=True, slots=True)
class QualityGates:
    max_target_error: float
    max_rms_edge_units: float
    max_p95_edge_units: float
    max_error_edge_units: float
    min_p05_scaled_jacobian: float
    max_p95_aspect_ratio: float


def quality_gates(quality: str) -> QualityGates:
    return {
        "DRAFT": QualityGates(0.75, 0.60, 1.50, 4.00, 0.01, 12.0),
        "HERO": QualityGates(0.25, 0.25, 0.75, 2.00, 0.05, 6.0),
    }.get(str(quality).upper(), QualityGates(0.35, 0.40, 1.00, 3.00, 0.03, 8.0))


def candidate_gate_failures(
    diagnostics: MeshDiagnostics,
    *,
    target_faces: int,
    geometry_rms: float,
    geometry_p95: float,
    geometry_max: float,
    target_edge_normalized: float,
    require_quads: bool,
    topology_policy: str,
    profile: ScanProfile | None,
    quality: str,
) -> list[str]:
    failures: list[str] = []
    if diagnostics.faces <= 0:
        failures.append("empty output")
    if require_quads and not diagnostics.quad_only:
        failures.append("local quad/manifold gate failed")
    for count, label in (
        (diagnostics.degenerate_faces, "degenerate faces"),
        (diagnostics.invalid_quads, "invalid quads"),
        (diagnostics.non_manifold_edges, "non-manifold edges"),
        (diagnostics.non_manifold_vertices, "non-manifold vertices"),
        (diagnostics.branched_boundary_vertices, "branched boundary vertices"),
        (diagnostics.inconsistent_edges, "inconsistently wound edges"),
        (diagnostics.duplicate_faces, "duplicate faces"),
    ):
        if count:
            failures.append(f"{count} {label}")
    if diagnostics.quads and diagnostics.min_scaled_jacobian <= 0.0:
        failures.append("inverted, crossed, or concave quad")
    if not (math.isfinite(geometry_rms) and math.isfinite(geometry_p95) and math.isfinite(geometry_max)):
        failures.append("surface-error measurement failed")

    gates = quality_gates(quality)
    target_error = abs(diagnostics.faces - target_faces) / max(1, target_faces)
    edge_scale = max(target_edge_normalized, 1e-9)
    rms_edge_units = geometry_rms / edge_scale
    p95_edge_units = geometry_p95 / edge_scale
    max_edge_units = geometry_max / edge_scale
    if target_error > gates.max_target_error:
        failures.append(
            f"target error {target_error:.1%} exceeds {gates.max_target_error:.0%} {quality.lower()} gate"
        )
    if math.isfinite(rms_edge_units) and rms_edge_units > gates.max_rms_edge_units:
        failures.append(
            f"RMS error {rms_edge_units:.2f} target edges exceeds {gates.max_rms_edge_units:.2f} gate"
        )
    if math.isfinite(p95_edge_units) and p95_edge_units > gates.max_p95_edge_units:
        failures.append(
            f"p95 error {p95_edge_units:.2f} target edges exceeds {gates.max_p95_edge_units:.2f} gate"
        )
    if math.isfinite(max_edge_units) and max_edge_units > gates.max_error_edge_units:
        failures.append(
            f"max sampled error {max_edge_units:.2f} target edges exceeds "
            f"{gates.max_error_edge_units:.2f} gate"
        )
    if diagnostics.quads and diagnostics.p05_scaled_jacobian < gates.min_p05_scaled_jacobian:
        failures.append(
            f"p05 Jacobian {diagnostics.p05_scaled_jacobian:.3f} is below "
            f"{gates.min_p05_scaled_jacobian:.3f} gate"
        )
    if diagnostics.quads and diagnostics.p95_aspect_ratio > gates.max_p95_aspect_ratio:
        failures.append(
            f"p95 aspect {diagnostics.p95_aspect_ratio:.2f} exceeds "
            f"{gates.max_p95_aspect_ratio:.2f} gate"
        )

    if topology_policy == "PRESERVE" and profile is not None and profile.topology_basis != "SAMPLED":
        if (
            profile.non_manifold_edges > 0
            or profile.non_manifold_vertices > 0
            or profile.branched_boundary_vertices > 0
            or profile.duplicate_faces > 0
        ):
            failures.append("source topology is not a certifiable manifold; choose Repair topology")
        if profile.euler_characteristic is not None and diagnostics.euler_characteristic != profile.euler_characteristic:
            failures.append("Euler characteristic changed")
        if diagnostics.boundary_loops != profile.boundary_loops:
            failures.append("boundary-loop count changed")
        if diagnostics.connected_components != profile.connected_components:
            failures.append("connected-component count changed")
        if diagnostics.component_euler != profile.component_euler:
            failures.append("per-component Euler characteristics changed")
        if not component_topology_preserved(profile.component_signature, diagnostics.component_signature):
            failures.append("per-component Euler/boundary signatures changed")
        if (diagnostics.boundary_edges == 0) != (profile.boundary_edges == 0):
            failures.append("open/closed topology changed")
    return failures


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = max(0.0, min(1.0, fraction)) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _boundary_components(boundary_edges: list[tuple[int, int]]) -> int:
    adjacency: dict[int, set[int]] = defaultdict(set)
    for a, b in boundary_edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    remaining = set(adjacency)
    components = 0
    while remaining:
        components += 1
        stack = [remaining.pop()]
        while stack:
            current = stack.pop()
            linked = adjacency[current] & remaining
            remaining.difference_update(linked)
            stack.extend(linked)
    return components


def analyze_object(
    obj: bpy.types.Object,
    target_faces: int,
    *,
    feature_angle_deg: float = 35.0,
    max_exact_faces: int = 500_000,
    max_sample_faces: int = 120_000,
) -> ScanProfile:
    mesh = obj.data
    face_count = len(mesh.polygons)
    vertex_count = len(mesh.vertices)
    exact = face_count <= max(1, int(max_exact_faces))
    stride = 1 if exact else max(1, math.ceil(face_count / max(1, int(max_sample_faces))))
    sample_indices = range(0, face_count, stride)

    bounds = [obj.matrix_world @ Vector(corner) for corner in (getattr(obj, "bound_box", ()) or ())]
    if bounds:
        xs = [corner[0] for corner in bounds]
        ys = [corner[1] for corner in bounds]
        zs = [corner[2] for corner in bounds]
        diagonal = math.sqrt(
            (max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2 + (max(zs) - min(zs)) ** 2
        )
    else:
        diagonal = 0.0
    area_epsilon = max(diagonal * diagonal * 1e-14, 1e-30)

    edge_use: Counter[tuple[int, int]] = Counter()
    first_normal: dict[tuple[int, int], tuple[float, float, float]] = {}
    sharp_edges: set[tuple[int, int]] = set()
    sampled = 0
    triangles = 0
    degenerates = 0
    sampled_area = 0.0
    threshold = math.cos(math.radians(max(0.0, min(180.0, feature_angle_deg))))
    normal_matrix = obj.matrix_world.to_3x3().inverted_safe().transposed()

    for index in sample_indices:
        poly = mesh.polygons[index]
        sampled += 1
        triangles += int(len(poly.vertices) == 3)
        coordinates = [obj.matrix_world @ mesh.vertices[vertex].co for vertex in poly.vertices]
        origin = coordinates[0] if coordinates else None
        area = (
            0.5
            * sum(
                (coordinates[corner] - origin).cross(coordinates[corner + 1] - origin).length
                for corner in range(1, len(coordinates) - 1)
            )
            if origin is not None
            else 0.0
        )
        sampled_area += max(0.0, area)
        degenerates += int(not math.isfinite(area) or area <= area_epsilon)
        if not exact:
            continue
        world_normal = normal_matrix @ poly.normal
        world_normal.normalize()
        normal = (float(world_normal.x), float(world_normal.y), float(world_normal.z))
        for raw_edge in poly.edge_keys:
            edge = tuple(sorted(raw_edge))
            edge_use[edge] += 1
            previous = first_normal.get(edge)
            if previous is None:
                first_normal[edge] = normal
            elif edge_use[edge] == 2:
                dot = previous[0] * normal[0] + previous[1] * normal[1] + previous[2] * normal[2]
                if dot < threshold:
                    sharp_edges.add(edge)

    scale = face_count / max(1, sampled)
    estimated_area = sampled_area * scale
    target_edge = math.sqrt(max(estimated_area, 1e-20) / max(1, int(target_faces)))

    exact_diagnostics = mesh_diagnostics(obj, target_faces) if exact else None
    boundary_count = exact_diagnostics.boundary_edges if exact_diagnostics is not None else -1
    boundary_loops = exact_diagnostics.boundary_loops if exact_diagnostics is not None else -1
    connected_components = exact_diagnostics.connected_components if exact_diagnostics is not None else -1
    component_euler = exact_diagnostics.component_euler if exact_diagnostics is not None else ()
    component_signature = exact_diagnostics.component_signature if exact_diagnostics is not None else ()
    non_manifold = exact_diagnostics.non_manifold_edges if exact_diagnostics is not None else -1
    non_manifold_vertices = exact_diagnostics.non_manifold_vertices if exact_diagnostics is not None else -1
    branched_boundary_vertices = (
        exact_diagnostics.branched_boundary_vertices if exact_diagnostics is not None else -1
    )
    inconsistent_edges = exact_diagnostics.inconsistent_edges if exact_diagnostics is not None else -1
    duplicate_faces = exact_diagnostics.duplicate_faces if exact_diagnostics is not None else -1
    euler = exact_diagnostics.euler_characteristic if exact_diagnostics is not None else None
    sharp_ratio = len(sharp_edges) / max(1, len(edge_use)) if exact else 0.0
    degenerate_ratio = degenerates / max(1, sampled)
    triangle_ratio = triangles / max(1, sampled)
    texture_critical = bool(mesh.uv_layers) or len(mesh.materials) > 1

    if exact and (
        non_manifold / max(1, len(edge_use)) > 0.001
        or non_manifold_vertices > 0
        or branched_boundary_vertices > 0
        or inconsistent_edges > 0
        or duplicate_faces > 0
        or degenerate_ratio > 0.005
    ):
        category = "DIRTY_SCAN"
        strategy = "DIRTY_FLOW"
    elif exact and boundary_count > 0:
        category = "OPEN_SURFACE"
        strategy = "OPEN_FLOW"
    elif sharp_ratio > 0.05:
        category = "HARD_SURFACE"
        strategy = "FEATURE_FLOW"
    elif not exact:
        category = "LARGE_SCAN"
        strategy = "FLOW_ENSEMBLE"
    else:
        category = "ORGANIC"
        strategy = "FLOW_ENSEMBLE"

    warnings: list[str] = []
    if not exact:
        warnings.append("Topology counts are sampled because the source exceeds the exact-analysis limit.")
    if category == "DIRTY_SCAN":
        warnings.append("Repair may close intentional holes and erase sheets thinner than the voxel size.")
    if category == "OPEN_SURFACE":
        warnings.append("Boundary preservation is enabled; voxel repair is unsafe for an intentional open fragment.")
    if texture_critical:
        warnings.append("Low-poly geometry cannot retain scan-frequency detail without UV transfer or baking.")
    if face_count / max(1, target_faces) > 250:
        warnings.append("The requested reduction is extreme; inspect silhouette and p95 projection error.")
    scales = [abs(float(value)) for value in obj.scale if abs(float(value)) > 1e-12]
    if scales and max(scales) / min(scales) > 1.01:
        warnings.append(
            "Metrics are evaluated in world space, but applying non-uniform scale can improve solver flow."
        )

    return ScanProfile(
        category=category,
        recommended_strategy=strategy,
        face_count=face_count,
        vertex_count=vertex_count,
        sampled_faces=sampled,
        exact_topology=exact,
        topology_basis="SOURCE_EXACT" if exact else "SAMPLED",
        boundary_edges=boundary_count,
        boundary_loops=boundary_loops,
        connected_components=connected_components,
        component_euler=component_euler,
        component_signature=component_signature,
        non_manifold_edges=non_manifold,
        non_manifold_vertices=non_manifold_vertices,
        branched_boundary_vertices=branched_boundary_vertices,
        inconsistent_edges=inconsistent_edges,
        duplicate_faces=duplicate_faces,
        euler_characteristic=euler,
        degenerate_ratio=degenerate_ratio,
        sharp_ratio=sharp_ratio,
        triangle_ratio=triangle_ratio,
        target_edge_length=target_edge,
        diagonal=max(diagonal, 1e-12),
        texture_critical=texture_critical,
        warnings=tuple(warnings),
    )


def build_candidate_specs(
    profile: ScanProfile,
    *,
    strategy: str,
    quality: str,
    seed: int,
    candidate_count: int,
    topology_policy: str,
    voxel_repair: bool = True,
) -> list[CandidateSpec]:
    requested = (strategy or "AUTO").upper()
    selected = profile.recommended_strategy if requested == "AUTO" else requested
    count = max(1, min(int(candidate_count), 8))
    if quality == "DRAFT":
        count = 1
    elif quality == "HERO":
        count = max(count, 5)

    if selected == "PAIR_CLOSURE":
        return [CandidateSpec("PAIR_CLOSURE", "PairClosure exact all-quad fallback", seed, pair_closure=True)]

    repair_allowed = voxel_repair and topology_policy == "REPAIR" and selected == "DIRTY_FLOW"
    preserve_sharp = selected in {"FEATURE_FLOW", "OPEN_FLOW"} or profile.category == "HARD_SURFACE"
    preserve_boundary = selected != "DIRTY_FLOW" or topology_policy == "PRESERVE"
    specs = []
    for index in range(count):
        repair = repair_allowed and index == 0
        label = "RepairFlow" if repair else ("FeatureFlow" if preserve_sharp else "FlowEnsemble")
        specs.append(
            CandidateSpec(
                strategy="DIRTY_FLOW" if repair else ("FEATURE_FLOW" if preserve_sharp else "FLOW_ENSEMBLE"),
                label=f"{label} candidate {index + 1}",
                seed=max(0, int(seed)) + index * 7919,
                repair=repair,
                preserve_sharp=preserve_sharp,
                preserve_boundary=preserve_boundary,
            )
        )
    return specs


def evaluate_candidate(
    spec: CandidateSpec,
    diagnostics: MeshDiagnostics,
    *,
    target_faces: int,
    geometry_rms: float,
    geometry_p95: float,
    geometry_max: float,
    target_edge_normalized: float,
    require_quads: bool,
    topology_policy: str,
    profile: ScanProfile,
    quality: str,
    elapsed_ms: float,
) -> CandidateEvaluation:
    failures = candidate_gate_failures(
        diagnostics,
        target_faces=target_faces,
        geometry_rms=geometry_rms,
        geometry_p95=geometry_p95,
        geometry_max=geometry_max,
        target_edge_normalized=target_edge_normalized,
        require_quads=require_quads,
        topology_policy=topology_policy,
        profile=profile,
        quality=quality,
    )

    edge_units = geometry_p95 / max(target_edge_normalized, 1e-9)
    geometry_quality = math.exp(-edge_units / 0.35) if math.isfinite(edge_units) else 0.0
    jacobian_quality = max(0.0, min(1.0, diagnostics.p05_scaled_jacobian))
    aspect_quality = math.exp(-max(0.0, diagnostics.p95_aspect_ratio - 1.0) / 3.0)
    count_quality = math.exp(-diagnostics.face_count_error)
    pole_quality = math.exp(-diagnostics.extraordinary_ratio * 4.0)
    score = (
        0.35 * geometry_quality
        + 0.25 * jacobian_quality
        + 0.15 * aspect_quality
        + 0.10 * count_quality
        + 0.10 * pole_quality
        + 0.05 * diagnostics.quad_ratio
    )
    return CandidateEvaluation(
        strategy=spec.strategy,
        label=spec.label,
        valid=not failures,
        score=score if not failures else -1.0,
        reason=(
            "accepted against exact reduced proxy; full source topology remains unverified"
            if not failures and topology_policy == "PRESERVE" and profile.topology_basis == "PROXY_EXACT"
            else ("accepted" if not failures else "; ".join(failures))
        ),
        faces=diagnostics.faces,
        quad_ratio=diagnostics.quad_ratio,
        target_error=abs(diagnostics.faces - target_faces) / max(1, target_faces),
        geometry_rms=geometry_rms,
        geometry_p95=geometry_p95,
        geometry_max=geometry_max,
        min_scaled_jacobian=diagnostics.min_scaled_jacobian,
        p95_aspect_ratio=diagnostics.p95_aspect_ratio,
        extraordinary_ratio=diagnostics.extraordinary_ratio,
        elapsed_ms=elapsed_ms,
    )
