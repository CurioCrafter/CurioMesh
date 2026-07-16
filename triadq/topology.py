from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np

from .types import Face, MeshData


MAX_QUAD_WARP_DEGREES = 60.0


@dataclass(slots=True)
class TopologySummary:
    used_vertices: int = 0
    edges: int = 0
    faces: int = 0
    quads: int = 0
    boundary_edges: int = 0
    boundary_components: int = 0
    connected_components: int = 0
    component_euler: tuple[int, ...] = ()
    component_signature: tuple[tuple[int, int], ...] = ()
    non_manifold_edges: int = 0
    non_manifold_vertices: int = 0
    branched_boundary_vertices: int = 0
    inconsistent_edges: int = 0
    duplicate_faces: int = 0
    euler_characteristic: int = 0
    extraordinary_vertices: int = 0
    defect_charge: int = 0
    degenerate_faces: int = 0
    invalid_quads: int = 0
    all_quads: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _edge_occurrences(
    faces: list[Face],
) -> dict[tuple[int, int], list[tuple[int, int, int]]]:
    occurrences: dict[tuple[int, int], list[tuple[int, int, int]]] = defaultdict(list)
    for face_index, face in enumerate(faces):
        for index in range(len(face)):
            a = face[index]
            b = face[(index + 1) % len(face)]
            occurrences[tuple(sorted((a, b)))].append((face_index, a, b))
    return occurrences


def _boundary_components(boundary_edges: list[tuple[int, int]]) -> tuple[int, int]:
    if not boundary_edges:
        return 0, 0
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
            vertex = stack.pop()
            linked = adjacency[vertex] & remaining
            remaining.difference_update(linked)
            stack.extend(linked)
    branched_vertices = sum(1 for neighbors in adjacency.values() if len(neighbors) != 2)
    return components, branched_vertices


def _canonical_face(face: Face) -> Face:
    """Return a rotation- and winding-independent face key."""

    rotations = [face[index:] + face[:index] for index in range(len(face))]
    reverse = tuple(reversed(face))
    rotations.extend(reverse[index:] + reverse[:index] for index in range(len(reverse)))
    return min(rotations)


def _component_topology_signatures(
    faces: list[Face],
    edge_occurrences: dict[tuple[int, int], list[tuple[int, int, int]]],
) -> tuple[tuple[int, int], ...]:
    face_count = len(faces)
    if face_count == 0:
        return ()
    adjacency: list[set[int]] = [set() for _ in range(face_count)]
    for occurrences in edge_occurrences.values():
        face_indices = {item[0] for item in occurrences}
        for face_index in face_indices:
            adjacency[face_index].update(face_indices - {face_index})
    remaining = set(range(face_count))
    signatures: list[tuple[int, int]] = []
    while remaining:
        seed = remaining.pop()
        component_faces = {seed}
        stack = [seed]
        while stack:
            linked = adjacency[stack.pop()] & remaining
            remaining.difference_update(linked)
            stack.extend(linked)
            component_faces.update(linked)
        vertices = {vertex for face_index in component_faces for vertex in faces[face_index]}
        edges: set[tuple[int, int]] = set()
        for face_index in component_faces:
            face = faces[face_index]
            edges.update(
                tuple(sorted((face[index], face[(index + 1) % len(face)])))
                for index in range(len(face))
            )
        boundary_edges = [
            edge
            for edge in edges
            if len(edge_occurrences[edge]) == 1
        ]
        boundary_cycles, _branched_vertices = _boundary_components(boundary_edges)
        euler_characteristic = len(vertices) - len(edges) + len(component_faces)
        signatures.append((euler_characteristic, boundary_cycles))
    return tuple(sorted(signatures))


def component_topology_preserved(
    source: tuple[tuple[int, int], ...],
    candidate: tuple[tuple[int, int], ...],
) -> bool:
    """Return whether component Euler/boundary signatures match exactly.

    The signatures are sorted, so component ordering and mesh indices do not
    affect the comparison. Pairing both invariants prevents boundary cycles
    from moving between equal-Euler components without detection.
    """

    return tuple(source) == tuple(candidate)


def _non_manifold_vertex_count(
    faces: list[Face],
    edge_occurrences: dict[tuple[int, int], list[tuple[int, int, int]]],
) -> int:
    """Count vertices whose incident faces form more than one edge-connected fan."""

    incident: dict[int, set[int]] = defaultdict(set)
    fan_adjacency: dict[int, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
    for face_index, face in enumerate(faces):
        for vertex in set(face):
            incident[vertex].add(face_index)
    for edge, occurrences in edge_occurrences.items():
        face_indices = {item[0] for item in occurrences}
        if len(face_indices) != 2:
            continue
        first, second = tuple(face_indices)
        for vertex in edge:
            fan_adjacency[vertex][first].add(second)
            fan_adjacency[vertex][second].add(first)

    non_manifold = 0
    for vertex, incident_faces in incident.items():
        if not incident_faces:
            continue
        seen = {next(iter(incident_faces))}
        queue = deque(seen)
        while queue:
            for linked in fan_adjacency[vertex][queue.popleft()]:
                if linked in incident_faces and linked not in seen:
                    seen.add(linked)
                    queue.append(linked)
        if seen != incident_faces:
            non_manifold += 1
    return non_manifold


def _face_area(vertices: np.ndarray, face: Face) -> float:
    if len(face) < 3 or len(set(face)) != len(face):
        return 0.0
    origin = vertices[face[0]]
    area_vector = np.zeros(3, dtype=np.float64)
    for index in range(1, len(face) - 1):
        area_vector += np.cross(vertices[face[index]] - origin, vertices[face[index + 1]] - origin)
    return 0.5 * float(np.linalg.norm(area_vector))


def _segments_cross(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, epsilon: float) -> bool:
    def orientation(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
        return float((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))

    first = orientation(a, b, c)
    second = orientation(a, b, d)
    third = orientation(c, d, a)
    fourth = orientation(c, d, b)
    return (
        ((first > epsilon and second < -epsilon) or (first < -epsilon and second > epsilon))
        and ((third > epsilon and fourth < -epsilon) or (third < -epsilon and fourth > epsilon))
    )


def _quad_is_locally_valid(vertices: np.ndarray, face: Face, epsilon: float) -> bool:
    if len(face) != 4 or len(set(face)) != 4:
        return False
    points = vertices[list(face)]
    if not np.all(np.isfinite(points)):
        return False

    normal = np.zeros(3, dtype=np.float64)
    for index in range(4):
        current = points[index]
        following = points[(index + 1) % 4]
        normal += np.cross(current, following)
    normal_length = float(np.linalg.norm(normal))
    if normal_length <= epsilon:
        return False
    normal /= normal_length

    def triangle_normal(first: int, second: int, third: int) -> np.ndarray | None:
        candidate = np.cross(points[second] - points[first], points[third] - points[first])
        length = float(np.linalg.norm(candidate))
        return None if length <= epsilon else candidate / length

    warp_cosine = math.cos(math.radians(MAX_QUAD_WARP_DEGREES))
    for first_triangle, second_triangle in (
        ((0, 1, 2), (0, 2, 3)),
        ((1, 2, 3), (1, 3, 0)),
    ):
        first_normal = triangle_normal(*first_triangle)
        second_normal = triangle_normal(*second_triangle)
        if first_normal is None or second_normal is None:
            return False
        if float(np.dot(first_normal, second_normal)) < warp_cosine:
            return False

    for index in range(4):
        current = points[index]
        following = points[(index + 1) % 4] - current
        previous = points[(index - 1) % 4] - current
        denominator = float(np.linalg.norm(following) * np.linalg.norm(previous))
        if denominator <= epsilon:
            return False
        scaled_jacobian = float(np.dot(np.cross(following, previous), normal)) / denominator
        if not math.isfinite(scaled_jacobian) or scaled_jacobian <= 1e-8:
            return False

    projection_axis = int(np.argmax(np.abs(normal)))
    projected = np.delete(points, projection_axis, axis=1)
    if _segments_cross(projected[0], projected[1], projected[2], projected[3], epsilon):
        return False
    if _segments_cross(projected[1], projected[2], projected[3], projected[0], epsilon):
        return False
    return True


def quad_is_locally_valid(vertices: np.ndarray, face: Face) -> bool:
    """Validate one quad's finite, convex, non-crossed local geometry."""

    coordinates = np.asarray(vertices, dtype=np.float64)
    if not len(coordinates):
        return False
    span = np.ptp(coordinates, axis=0)
    epsilon = max(float(np.dot(span, span)) * 1e-14, 1e-20)
    return _quad_is_locally_valid(coordinates, face, epsilon)


def topology_summary(mesh: MeshData) -> TopologySummary:
    edge_occurrences = _edge_occurrences(mesh.faces)
    edge_counts = Counter({edge: len(items) for edge, items in edge_occurrences.items()})
    used_vertices = {vertex for face in mesh.faces for vertex in face}
    adjacency: dict[int, set[int]] = defaultdict(set)
    for a, b in edge_counts:
        adjacency[a].add(b)
        adjacency[b].add(a)

    boundary_edges = [edge for edge, count in edge_counts.items() if count == 1]
    boundary_components, branched_boundary_vertices = _boundary_components(boundary_edges)
    non_manifold_edges = sum(1 for count in edge_counts.values() if count > 2)
    inconsistent_edges = sum(
        1
        for occurrences in edge_occurrences.values()
        if len(occurrences) == 2
        and occurrences[0][1:] == occurrences[1][1:]
    )
    duplicate_faces = len(mesh.faces) - len({_canonical_face(face) for face in mesh.faces})
    non_manifold_vertices = _non_manifold_vertex_count(mesh.faces, edge_occurrences)
    boundary_vertices = {vertex for edge in boundary_edges for vertex in edge}
    extraordinary = 0
    defect_charge = 0
    for vertex in used_vertices:
        valence = len(adjacency[vertex])
        ideal = 3 if vertex in boundary_vertices else 4
        if valence != ideal:
            extraordinary += 1
        defect_charge += ideal - valence

    if len(mesh.vertices):
        span = np.ptp(np.asarray(mesh.vertices, dtype=np.float64), axis=0)
        area_epsilon = max(float(np.dot(span, span)) * 1e-14, 1e-20)
    else:
        area_epsilon = 1e-20
    degenerate = sum(1 for face in mesh.faces if _face_area(mesh.vertices, face) <= area_epsilon)
    quads = sum(1 for face in mesh.faces if len(face) == 4)
    invalid_quads = sum(
        1
        for face in mesh.faces
        if len(face) == 4 and not _quad_is_locally_valid(mesh.vertices, face, area_epsilon)
    )
    component_signature = _component_topology_signatures(mesh.faces, edge_occurrences)
    component_euler = tuple(signature[0] for signature in component_signature)
    return TopologySummary(
        used_vertices=len(used_vertices),
        edges=len(edge_counts),
        faces=len(mesh.faces),
        quads=quads,
        boundary_edges=len(boundary_edges),
        boundary_components=boundary_components,
        connected_components=len(component_euler),
        component_euler=component_euler,
        component_signature=component_signature,
        non_manifold_edges=non_manifold_edges,
        non_manifold_vertices=non_manifold_vertices,
        branched_boundary_vertices=branched_boundary_vertices,
        inconsistent_edges=inconsistent_edges,
        duplicate_faces=duplicate_faces,
        euler_characteristic=len(used_vertices) - len(edge_counts) + len(mesh.faces),
        extraordinary_vertices=extraordinary,
        defect_charge=defect_charge,
        degenerate_faces=degenerate,
        invalid_quads=invalid_quads,
        all_quads=(
            bool(mesh.faces)
            and quads == len(mesh.faces)
            and degenerate == 0
            and non_manifold_edges == 0
            and non_manifold_vertices == 0
            and branched_boundary_vertices == 0
            and inconsistent_edges == 0
            and duplicate_faces == 0
            and invalid_quads == 0
        ),
    )


def quad_only_closure(mesh: MeshData, *, name: str | None = None) -> MeshData:
    """Conformingly split every polygon corner into a quad.

    Edge midpoints are shared globally, so adjacent faces receive matching
    topology. UV midpoints remain face-local, which preserves UV seams.
    """

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    output_vertices = vertices.tolist()
    edge_midpoints: dict[tuple[int, int], int] = {}
    output_faces: list[Face] = []
    output_materials: list[str] = []

    output_uvs = None if mesh.uvs is None else np.asarray(mesh.uvs, dtype=np.float64).tolist()
    output_face_uvs: list[Face] | None = [] if output_uvs is not None and mesh.face_uvs is not None else None

    for face_index, face in enumerate(mesh.faces):
        if len(face) < 3:
            raise ValueError(f"Face {face_index} has fewer than three corners.")
        if len(set(face)) != len(face):
            raise ValueError(f"Face {face_index} repeats a vertex and cannot be closed safely.")
        if any(vertex < 0 or vertex >= len(vertices) for vertex in face):
            raise ValueError(f"Face {face_index} has an out-of-range vertex index.")

        mids: list[int] = []
        for corner, vertex in enumerate(face):
            following = face[(corner + 1) % len(face)]
            edge = tuple(sorted((vertex, following)))
            midpoint = edge_midpoints.get(edge)
            if midpoint is None:
                midpoint = len(output_vertices)
                edge_midpoints[edge] = midpoint
                output_vertices.append(((vertices[vertex] + vertices[following]) * 0.5).tolist())
            mids.append(midpoint)

        center = len(output_vertices)
        output_vertices.append(np.mean(vertices[list(face)], axis=0).tolist())
        material = mesh.face_materials[face_index] if face_index < len(mesh.face_materials) else "default"

        face_uv = None
        uv_mids: list[int] = []
        uv_center = -1
        if output_face_uvs is not None and mesh.face_uvs is not None and output_uvs is not None:
            candidate = mesh.face_uvs[face_index] if face_index < len(mesh.face_uvs) else ()
            if len(candidate) == len(face) and all(0 <= uv < len(mesh.uvs) for uv in candidate):
                face_uv = candidate
                for corner, uv_index in enumerate(face_uv):
                    following_uv = face_uv[(corner + 1) % len(face_uv)]
                    uv_mids.append(len(output_uvs))
                    output_uvs.append(((mesh.uvs[uv_index] + mesh.uvs[following_uv]) * 0.5).tolist())
                uv_center = len(output_uvs)
                output_uvs.append(np.mean(mesh.uvs[list(face_uv)], axis=0).tolist())

        for corner, vertex in enumerate(face):
            output_faces.append((vertex, mids[corner], center, mids[(corner - 1) % len(face)]))
            output_materials.append(material)
            if output_face_uvs is not None:
                if face_uv is None:
                    output_face_uvs = None
                else:
                    output_face_uvs.append(
                        (face_uv[corner], uv_mids[corner], uv_center, uv_mids[(corner - 1) % len(face)])
                    )

    result = MeshData(
        vertices=np.asarray(output_vertices, dtype=np.float64),
        faces=output_faces,
        face_materials=output_materials,
        uvs=None if output_uvs is None else np.asarray(output_uvs, dtype=np.float64),
        face_uvs=output_face_uvs,
        name=name or f"{mesh.name}_quad_closure",
    )
    summary = topology_summary(result)
    if not summary.all_quads:
        raise ValueError(
            "Quad closure failed validation: "
            f"quads={summary.quads}/{summary.faces}, degenerate={summary.degenerate_faces}, "
            f"invalid_quads={summary.invalid_quads}, non_manifold_edges={summary.non_manifold_edges}, "
            f"non_manifold_vertices={summary.non_manifold_vertices}, "
            f"branched_boundary_vertices={summary.branched_boundary_vertices}, "
            f"inconsistent_edges={summary.inconsistent_edges}, duplicate_faces={summary.duplicate_faces}."
        )
    return result
