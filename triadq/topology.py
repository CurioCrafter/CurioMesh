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
    component_detail: tuple[tuple[int, int, int], ...] = ()
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
) -> tuple[tuple[int, int, int], ...]:
    """Sorted (euler, boundary cycles, face count) per connected component."""

    face_count = len(faces)
    if face_count == 0:
        return ()
    adjacency: list[set[int]] = [set() for _ in range(face_count)]
    for occurrences in edge_occurrences.values():
        face_indices = {item[0] for item in occurrences}
        for face_index in face_indices:
            adjacency[face_index].update(face_indices - {face_index})
    remaining = set(range(face_count))
    signatures: list[tuple[int, int, int]] = []
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
        signatures.append((euler_characteristic, boundary_cycles, len(component_faces)))
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


def _face_areas(vertices: np.ndarray, faces: list[Face]) -> np.ndarray:
    """Fan areas for every face; faces with a repeated vertex get zero."""

    areas = np.zeros(len(faces), dtype=np.float64)
    by_length: dict[int, list[int]] = defaultdict(list)
    for face_index, face in enumerate(faces):
        if len(face) >= 3:
            by_length[len(face)].append(face_index)
    for length, indices in by_length.items():
        index_array = np.asarray([faces[i] for i in indices], dtype=np.int64)
        points = vertices[index_array]
        relative = points[:, 1:] - points[:, :1]
        area_vector = np.cross(relative[:, :-1], relative[:, 1:]).sum(axis=1)
        local = 0.5 * np.linalg.norm(area_vector, axis=1)
        ordered = np.sort(index_array, axis=1)
        local[np.any(ordered[:, 1:] == ordered[:, :-1], axis=1)] = 0.0
        areas[indices] = local
    return areas


def _segments_cross(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, epsilon: np.ndarray) -> np.ndarray:
    """Proper 2D segment crossing for row-stacked points (M, 2)."""

    def orientation(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> np.ndarray:
        return (q[:, 0] - p[:, 0]) * (r[:, 1] - p[:, 1]) - (q[:, 1] - p[:, 1]) * (r[:, 0] - p[:, 0])

    first = orientation(a, b, c)
    second = orientation(a, b, d)
    third = orientation(c, d, a)
    fourth = orientation(c, d, b)
    return (
        (((first > epsilon) & (second < -epsilon)) | ((first < -epsilon) & (second > epsilon)))
        & (((third > epsilon) & (fourth < -epsilon)) | ((third < -epsilon) & (fourth > epsilon)))
    )


_PROJECTION_KEEP = np.asarray([[1, 2], [0, 2], [0, 1]], dtype=np.int64)


def _quad_points_valid(points: np.ndarray, epsilon: float | np.ndarray) -> np.ndarray:
    """Vectorized finite, convex, non-crossed, low-warp test for (M, 4, 3) quads.

    `epsilon` is a scalar or one tolerance per quad.
    """

    count = len(points)
    if count == 0:
        return np.zeros(0, dtype=bool)
    epsilon = np.broadcast_to(np.asarray(epsilon, dtype=np.float64), (count,))
    valid = np.all(np.isfinite(points), axis=(1, 2))
    with np.errstate(all="ignore"):
        following_points = np.roll(points, -1, axis=1)
        normal = np.cross(points, following_points).sum(axis=1)
        normal_length = np.linalg.norm(normal, axis=1)
        valid &= normal_length > epsilon
        normal = normal / np.where(normal_length > 0.0, normal_length, 1.0)[:, None]

        warp_cosine = math.cos(math.radians(MAX_QUAD_WARP_DEGREES))

        def triangle_normal(first: int, second: int, third: int) -> tuple[np.ndarray, np.ndarray]:
            candidate = np.cross(points[:, second] - points[:, first], points[:, third] - points[:, first])
            length = np.linalg.norm(candidate, axis=1)
            return candidate / np.where(length > 0.0, length, 1.0)[:, None], length > epsilon

        for first_triangle, second_triangle in (
            ((0, 1, 2), (0, 2, 3)),
            ((1, 2, 3), (1, 3, 0)),
        ):
            first_normal, first_ok = triangle_normal(*first_triangle)
            second_normal, second_ok = triangle_normal(*second_triangle)
            valid &= first_ok & second_ok
            valid &= ~(np.einsum("ij,ij->i", first_normal, second_normal) < warp_cosine)

        following = following_points - points
        previous = np.roll(points, 1, axis=1) - points
        denominator = np.linalg.norm(following, axis=2) * np.linalg.norm(previous, axis=2)
        valid &= np.all(denominator > epsilon[:, None], axis=1)
        scaled_jacobian = np.einsum("mkj,mj->mk", np.cross(following, previous), normal) / np.where(
            denominator > 0.0, denominator, 1.0
        )
        valid &= np.all(np.isfinite(scaled_jacobian) & (scaled_jacobian > 1e-8), axis=1)

        axis = np.argmax(np.abs(np.nan_to_num(normal)), axis=1)
        keep = np.broadcast_to(_PROJECTION_KEEP[axis][:, None, :], (count, 4, 2))
        projected = np.take_along_axis(points, keep, axis=2)
        valid &= ~_segments_cross(projected[:, 0], projected[:, 1], projected[:, 2], projected[:, 3], epsilon)
        valid &= ~_segments_cross(projected[:, 1], projected[:, 2], projected[:, 3], projected[:, 0], epsilon)
    return valid


def _quads_valid(vertices: np.ndarray, quads: np.ndarray, epsilon: float) -> np.ndarray:
    quads = np.asarray(quads, dtype=np.int64).reshape(-1, 4)
    ordered = np.sort(quads, axis=1)
    distinct = np.all(ordered[:, 1:] != ordered[:, :-1], axis=1)
    return distinct & _quad_points_valid(vertices[quads], epsilon)


def _quad_is_locally_valid(vertices: np.ndarray, face: Face, epsilon: float) -> bool:
    if len(face) != 4:
        return False
    return bool(_quads_valid(vertices, np.asarray([face]), epsilon)[0])


def _span_epsilon(coordinates: np.ndarray) -> float:
    if not len(coordinates):
        return 1e-20
    span = np.ptp(coordinates, axis=0)
    return max(float(np.dot(span, span)) * 1e-14, 1e-20)


def quads_locally_valid(vertices: np.ndarray, quads: np.ndarray) -> np.ndarray:
    """Batch form of `quad_is_locally_valid` for an (M, 4) index array."""

    coordinates = np.asarray(vertices, dtype=np.float64)
    if not len(coordinates):
        return np.zeros(len(quads), dtype=bool)
    return _quads_valid(coordinates, quads, _span_epsilon(coordinates))


def quad_is_locally_valid(vertices: np.ndarray, face: Face) -> bool:
    """Validate one quad's finite, convex, non-crossed local geometry."""

    coordinates = np.asarray(vertices, dtype=np.float64)
    if not len(coordinates):
        return False
    return _quad_is_locally_valid(coordinates, face, _span_epsilon(coordinates))


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

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    area_epsilon = _span_epsilon(vertices)
    degenerate = int(np.count_nonzero(_face_areas(vertices, mesh.faces) <= area_epsilon))
    quad_faces = [face for face in mesh.faces if len(face) == 4]
    quads = len(quad_faces)
    invalid_quads = (
        int(np.count_nonzero(~_quads_valid(vertices, np.asarray(quad_faces), area_epsilon)))
        if quad_faces
        else 0
    )
    component_detail = _component_topology_signatures(mesh.faces, edge_occurrences)
    component_signature = tuple(sorted((item[0], item[1]) for item in component_detail))
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
        component_detail=component_detail,
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


def split_non_manifold_vertex_fans(mesh: MeshData) -> tuple[MeshData, int]:
    """Duplicate vertices whose incident faces form disconnected edge fans.

    A vertex-only bow-tie is not a manifold surface even when every edge is
    used by at most two faces.  The repair is deliberately coordinate
    preserving: each additional fan receives a duplicate of the original
    vertex, while face order, materials, and face-corner UV indices remain
    unchanged.  The returned count is the number of duplicate vertices added.

    This function does not attempt to repair non-manifold edges, duplicate
    faces, or geometric degeneracy.  Callers should gate those conditions
    separately before treating the result as a certifiable manifold.
    """

    if not mesh.faces:
        return mesh, 0

    edge_occurrences = _edge_occurrences(mesh.faces)
    if any(len(occurrences) > 2 for occurrences in edge_occurrences.values()):
        return mesh, 0

    incident_faces: dict[int, set[int]] = defaultdict(set)
    vertex_edge_faces: dict[int, dict[int, set[int]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for face_index, face in enumerate(mesh.faces):
        for vertex in set(face):
            incident_faces[vertex].add(face_index)
        for corner, vertex in enumerate(face):
            following = face[(corner + 1) % len(face)]
            vertex_edge_faces[vertex][following].add(face_index)
            vertex_edge_faces[following][vertex].add(face_index)

    duplicate_coordinates: list[np.ndarray] = []
    remapped_corners: dict[tuple[int, int], int] = {}

    for vertex in sorted(incident_faces):
        local_faces = incident_faces[vertex]
        if len(local_faces) <= 1:
            continue

        adjacency: dict[int, set[int]] = {face_index: set() for face_index in local_faces}
        for sharing_faces in vertex_edge_faces[vertex].values():
            linked = sorted(sharing_faces & local_faces)
            if len(linked) < 2:
                continue
            anchor = linked[0]
            for other in linked[1:]:
                adjacency[anchor].add(other)
                adjacency[other].add(anchor)

        remaining = set(local_faces)
        fans: list[list[int]] = []
        while remaining:
            seed = min(remaining)
            remaining.remove(seed)
            fan = [seed]
            queue = deque([seed])
            while queue:
                linked = adjacency[queue.popleft()] & remaining
                for face_index in sorted(linked):
                    remaining.remove(face_index)
                    queue.append(face_index)
                    fan.append(face_index)
            fans.append(sorted(fan))

        fans.sort(key=lambda fan: fan[0])
        for fan in fans[1:]:
            duplicate = mesh.vertex_count + len(duplicate_coordinates)
            duplicate_coordinates.append(np.array(mesh.vertices[vertex], dtype=np.float64, copy=True))
            for face_index in fan:
                remapped_corners[(face_index, vertex)] = duplicate

    if not duplicate_coordinates:
        return mesh, 0
    output_vertices = np.vstack(
        (mesh.vertices, np.asarray(duplicate_coordinates, dtype=np.float64))
    )
    output_faces = [
        tuple(remapped_corners.get((face_index, vertex), vertex) for vertex in face)
        for face_index, face in enumerate(mesh.faces)
    ]
    return (
        mesh.copy_with(
            vertices=output_vertices,
            faces=output_faces,
        ),
        len(duplicate_coordinates),
    )


def _toggle_edge(edges: set[tuple[int, int]], edge: tuple[int, int]) -> None:
    """Toggle an edge in a GF(2) edge set."""

    if edge in edges:
        edges.remove(edge)
    else:
        edges.add(edge)


def _parity_path_edges(mesh: MeshData) -> set[tuple[int, int]]:
    """Solve sparse transition parity on the quadrilateral dual graph.

    Every triangle first marks its three edges. That gives each residual
    triangle the robust six-corner template used by global corner closure,
    while allowing already-valid quads to remain untouched. Quads incident to
    an odd number of marked edges are then terminals of a T-join over the
    quad-only dual, with mesh boundaries acting as exterior terminals. Paths
    are combined by symmetric difference, so overlaps cancel exactly.
    """

    occurrences = _edge_occurrences(mesh.faces)
    selected: set[tuple[int, int]] = {
        tuple(sorted((face[corner], face[(corner + 1) % len(face)])))
        for face in mesh.faces
        if len(face) == 3
        for corner in range(3)
    }
    adjacency: list[list[tuple[int, tuple[int, int]]]] = [[] for _face in mesh.faces]
    boundary_by_face: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for edge, uses in occurrences.items():
        if len(uses) > 2:
            raise ValueError(f"Edge {edge} has {len(uses)} incident faces.")
        if len(uses) == 1:
            face_index = uses[0][0]
            if len(mesh.faces[face_index]) == 4:
                boundary_by_face[face_index].append(edge)
        elif len(uses) == 2:
            first, second = uses[0][0], uses[1][0]
            if len(mesh.faces[first]) == 4 and len(mesh.faces[second]) == 4:
                adjacency[first].append((second, edge))
                adjacency[second].append((first, edge))

    for linked in adjacency:
        linked.sort(key=lambda item: (item[0], item[1]))
    for edges in boundary_by_face.values():
        edges.sort()

    terminals = {
        face_index
        for face_index, face in enumerate(mesh.faces)
        if len(face) == 4
        and sum(
            tuple(sorted((face[corner], face[(corner + 1) % len(face)]))) in selected
            for corner in range(4)
        )
        % 2
    }
    while terminals:
        start = min(terminals)
        distances = {start: 0}
        parents: dict[int, tuple[int, tuple[int, int]]] = {}
        queue = deque([start])
        # Every candidate costs at least its BFS distance, so once the queue
        # reaches the best known cost no cheaper (or tie-breaking) target
        # remains undiscovered and the search can stop.
        best_cost = 1 if start in boundary_by_face else math.inf
        while queue:
            current = queue.popleft()
            if distances[current] >= best_cost:
                break
            for linked, edge in adjacency[current]:
                if linked in distances:
                    continue
                distance = distances[current] + 1
                distances[linked] = distance
                parents[linked] = (current, edge)
                queue.append(linked)
                if linked in terminals:
                    best_cost = min(best_cost, distance)
                elif linked in boundary_by_face:
                    best_cost = min(best_cost, distance + 1)

        # Pairing to another transition quad costs one edge per dual step.
        # Reaching the exterior additionally costs one boundary edge. On an
        # equal cost, prefer face-to-face pairing because it discharges two
        # terminals.
        candidates: list[tuple[int, int, int, tuple[int, int], int | None]] = []
        for target in sorted(terminals - {start}):
            if target in distances:
                candidates.append((distances[target], 0, target, (-1, -1), target))
        for target, boundary_edges in sorted(boundary_by_face.items()):
            if target not in distances:
                continue
            for edge in boundary_edges:
                candidates.append((distances[target] + 1, 1, target, edge, None))

        if not candidates:
            raise ValueError(
                "Transition parity cannot be discharged through quad-only "
                "adjacency or the mesh boundary."
            )

        _cost, _kind, target, exterior_edge, paired_terminal = min(candidates)
        current = target
        while current != start:
            parent, edge = parents[current]
            _toggle_edge(selected, edge)
            current = parent
        if paired_terminal is None:
            _toggle_edge(selected, exterior_edge)
            terminals.remove(start)
        else:
            terminals.remove(start)
            terminals.remove(paired_terminal)

    for face_index, face in enumerate(mesh.faces):
        split_count = sum(
            tuple(sorted((face[corner], face[(corner + 1) % len(face)]))) in selected
            for corner in range(len(face))
        )
        if (len(face) + split_count) % 2:
            raise ValueError(f"Parity solve failed on face {face_index}.")
    return selected


def _parity_template_validity(
    mesh: MeshData,
    vertices: np.ndarray,
    base_coordinates: np.ndarray,
    selected_edges: set[tuple[int, int]],
    midpoint_indices: dict[tuple[int, int], int],
) -> list[list[bool]]:
    """Validate every ParityPath template candidate of every face in one batch.

    Per face, candidates are listed in the order `parity_path_quad_closure`
    tries them: the untouched quad, the three two-quad diagonal cuts, then the
    two center fans. Each cell is checked with a tolerance from its own four
    corners, as `quad_is_locally_valid` does on a local coordinate array.
    """

    center_offset = len(base_coordinates)
    centers = [vertices[list(face)].mean(axis=0) for face in mesh.faces]
    quads: list[tuple[int, int, int, int]] = []
    patch_sizes: list[int] = []
    per_face: list[int] = []
    for face_index, face in enumerate(mesh.faces):
        augmented: list[int] = []
        for corner, vertex in enumerate(face):
            augmented.append(vertex)
            edge = tuple(sorted((vertex, face[(corner + 1) % len(face)])))
            if edge in selected_edges:
                augmented.append(midpoint_indices[edge])
        split_count = len(augmented) - len(face)
        patches: list[list[tuple[int, int, int, int]]] = []
        if len(face) == 4 and split_count == 0:
            patches.append([tuple(face)])
        if len(face) == 4 and split_count == 2:
            for offset in range(3):
                patches.append([
                    tuple(augmented[(offset + step) % 6] for step in range(4)),
                    tuple(augmented[(offset + 3 + step) % 6] for step in range(4)),
                ])
        if len(augmented) % 2 == 0 and len(augmented) >= 4:
            center = center_offset + face_index
            for offset in (0, 1):
                patches.append([
                    (
                        center,
                        augmented[(offset + 2 * cell) % len(augmented)],
                        augmented[(offset + 2 * cell + 1) % len(augmented)],
                        augmented[(offset + 2 * cell + 2) % len(augmented)],
                    )
                    for cell in range(len(augmented) // 2)
                ])
        per_face.append(len(patches))
        for patch in patches:
            quads.extend(patch)
            patch_sizes.append(len(patch))

    if not quads:
        return [[] for _face in mesh.faces]
    coordinates = np.vstack((base_coordinates, np.asarray(centers, dtype=np.float64)))
    points = coordinates[np.asarray(quads, dtype=np.int64)]
    span = np.ptp(points, axis=1)
    epsilon = np.maximum(np.einsum("ij,ij->i", span, span) * 1e-14, 1e-20)
    quad_ok = _quad_points_valid(points, epsilon)
    starts = np.concatenate(([0], np.cumsum(patch_sizes)[:-1]))
    patch_ok = np.logical_and.reduceat(quad_ok, starts).tolist()
    results: list[list[bool]] = []
    cursor = 0
    for count in per_face:
        results.append(patch_ok[cursor : cursor + count])
        cursor += count
    return results


def parity_path_quad_closure(mesh: MeshData, *, name: str | None = None) -> MeshData:
    """Convert a triangle/quad mesh with a sparse parity-path closure.

    Valid unaffected quads remain one face. Odd triangle cells are connected
    in the dual graph or routed to a boundary; only edges on those paths are
    split. Shared geometry midpoints prevent T-junctions while face-local UV
    interpolation preserves seams. Every local patch and the final component
    Euler/boundary signature are validated transactionally.
    """

    if not mesh.faces:
        raise ValueError("ParityPath closure requires at least one face.")
    if any(len(face) not in {3, 4} for face in mesh.faces):
        raise ValueError("ParityPath closure accepts triangle/quad meshes only.")

    source_summary = topology_summary(mesh)
    source_errors = {
        "degenerate_faces": source_summary.degenerate_faces,
        "invalid_quads": source_summary.invalid_quads,
        "non_manifold_edges": source_summary.non_manifold_edges,
        "non_manifold_vertices": source_summary.non_manifold_vertices,
        "branched_boundary_vertices": source_summary.branched_boundary_vertices,
        "inconsistent_edges": source_summary.inconsistent_edges,
        "duplicate_faces": source_summary.duplicate_faces,
    }
    failed = {key: value for key, value in source_errors.items() if value}
    if failed:
        details = ", ".join(f"{key}={value}" for key, value in failed.items())
        raise ValueError(f"ParityPath source topology is not certifiable: {details}.")

    selected_edges = _parity_path_edges(mesh)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    output_vertices = vertices.tolist()
    midpoint_indices: dict[tuple[int, int], int] = {}
    for edge in sorted(selected_edges):
        midpoint_indices[edge] = len(output_vertices)
        output_vertices.append(((vertices[edge[0]] + vertices[edge[1]]) * 0.5).tolist())

    output_faces: list[Face] = []
    output_materials: list[str] = []
    output_uvs = None if mesh.uvs is None else np.asarray(mesh.uvs, dtype=np.float64).tolist()
    output_face_uvs: list[Face] | None = (
        [] if output_uvs is not None and mesh.face_uvs is not None else None
    )

    template_results = _parity_template_validity(
        mesh, vertices, np.asarray(output_vertices, dtype=np.float64), selected_edges, midpoint_indices
    )

    for face_index, face in enumerate(mesh.faces):
        # Results arrive in the same order the templates are tried below.
        patch_is_valid = iter(template_results[face_index]).__next__
        material = (
            mesh.face_materials[face_index]
            if face_index < len(mesh.face_materials)
            else "default"
        )
        augmented: list[int] = []
        face_uv: Face | None = None
        augmented_uv: list[int] = []
        if output_face_uvs is not None and output_uvs is not None and mesh.face_uvs is not None:
            candidate_uv = mesh.face_uvs[face_index]
            if (
                len(candidate_uv) == len(face)
                and mesh.uvs is not None
                and all(0 <= uv < len(mesh.uvs) for uv in candidate_uv)
            ):
                face_uv = candidate_uv
            else:
                output_face_uvs = None

        split_count = 0
        for corner, vertex in enumerate(face):
            following = face[(corner + 1) % len(face)]
            edge = tuple(sorted((vertex, following)))
            augmented.append(vertex)
            if face_uv is not None:
                augmented_uv.append(face_uv[corner])
            if edge not in selected_edges:
                continue
            split_count += 1
            augmented.append(midpoint_indices[edge])
            if face_uv is not None and output_uvs is not None and mesh.uvs is not None:
                following_uv = face_uv[(corner + 1) % len(face_uv)]
                uv_midpoint = len(output_uvs)
                output_uvs.append(
                    ((mesh.uvs[face_uv[corner]] + mesh.uvs[following_uv]) * 0.5).tolist()
                )
                augmented_uv.append(uv_midpoint)

        patch: list[Face] | None = None
        patch_uvs: list[Face] | None = None
        if len(face) == 4 and split_count == 0:
            candidate_patch = [tuple(face)]
            if patch_is_valid():
                patch = candidate_patch
                if face_uv is not None:
                    patch_uvs = [tuple(face_uv)]

        # A six-corner transition cell can sometimes be cut directly into two
        # valid quads. Try each opposite boundary diagonal before spending a
        # center vertex and a third quad.
        if patch is None and len(face) == 4 and split_count == 2:
            for offset in range(3):
                candidate_patch = [
                    tuple(augmented[(offset + step) % 6] for step in range(4)),
                    tuple(augmented[(offset + 3 + step) % 6] for step in range(4)),
                ]
                if not patch_is_valid():
                    continue
                patch = candidate_patch
                if face_uv is not None:
                    patch_uvs = [
                        tuple(augmented_uv[(offset + step) % 6] for step in range(4)),
                        tuple(augmented_uv[(offset + 3 + step) % 6] for step in range(4)),
                    ]
                break

        if patch is None:
            if len(augmented) % 2 or len(augmented) < 4:
                raise ValueError(
                    f"Face {face_index} has no even transition boundary "
                    f"(corners={len(augmented)})."
                )
            center = len(output_vertices)
            output_vertices.append(np.mean(vertices[list(face)], axis=0).tolist())
            center_uv = -1
            if face_uv is not None and output_uvs is not None and mesh.uvs is not None:
                center_uv = len(output_uvs)
                output_uvs.append(np.mean(mesh.uvs[list(face_uv)], axis=0).tolist())

            cell_count = len(augmented) // 2
            for offset in (0, 1):
                candidate_patch = [
                    (
                        center,
                        augmented[(offset + 2 * cell) % len(augmented)],
                        augmented[(offset + 2 * cell + 1) % len(augmented)],
                        augmented[(offset + 2 * cell + 2) % len(augmented)],
                    )
                    for cell in range(cell_count)
                ]
                if not patch_is_valid():
                    continue
                patch = candidate_patch
                if face_uv is not None:
                    patch_uvs = [
                        (
                            center_uv,
                            augmented_uv[(offset + 2 * cell) % len(augmented_uv)],
                            augmented_uv[(offset + 2 * cell + 1) % len(augmented_uv)],
                            augmented_uv[(offset + 2 * cell + 2) % len(augmented_uv)],
                        )
                        for cell in range(cell_count)
                    ]
                break

        if patch is None:
            raise ValueError(
                f"Face {face_index} has no locally valid ParityPath quad template."
            )
        output_faces.extend(patch)
        output_materials.extend([material] * len(patch))
        if output_face_uvs is not None:
            if patch_uvs is None:
                output_face_uvs = None
            else:
                output_face_uvs.extend(patch_uvs)

    result = MeshData(
        vertices=np.asarray(output_vertices, dtype=np.float64),
        faces=output_faces,
        face_materials=output_materials,
        uvs=None if output_uvs is None else np.asarray(output_uvs, dtype=np.float64),
        face_uvs=output_face_uvs,
        name=name or f"{mesh.name}_parity_path_closure",
    )
    summary = topology_summary(result)
    if not summary.all_quads:
        raise ValueError(
            "ParityPath closure failed validation: "
            f"quads={summary.quads}/{summary.faces}, degenerate={summary.degenerate_faces}, "
            f"invalid_quads={summary.invalid_quads}, non_manifold_edges={summary.non_manifold_edges}, "
            f"non_manifold_vertices={summary.non_manifold_vertices}, "
            f"branched_boundary_vertices={summary.branched_boundary_vertices}, "
            f"inconsistent_edges={summary.inconsistent_edges}, duplicate_faces={summary.duplicate_faces}."
        )
    if not component_topology_preserved(
        source_summary.component_signature,
        summary.component_signature,
    ):
        raise ValueError(
            "ParityPath closure changed the component Euler/boundary signature: "
            f"{source_summary.component_signature} -> {summary.component_signature}."
        )
    return result


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
