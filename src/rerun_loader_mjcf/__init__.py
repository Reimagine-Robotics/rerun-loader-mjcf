from __future__ import annotations

import pathlib

import mujoco
import numpy as np
import numpy.typing as npt
import rerun as rr

# MuJoCo uses -1 to indicate "no reference" for IDs (material, texture, mesh, etc.)
_MJCF_NO_ID = -1
_PLANE_SIZE = 5.0

# Type alias for mesh data tuple (vertices, faces, normals)
MeshTuple = tuple[
    npt.NDArray[np.float32], npt.NDArray[np.int32], npt.NDArray[np.float32]
]


def create_sphere(radius: float = 1.0, subdivisions: int = 3) -> MeshTuple:
    """Create an icosphere mesh.

    Returns (vertices, faces, normals).
    """
    # Start with icosahedron vertices
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    vertices = np.array(
        [
            [-1, phi, 0],
            [1, phi, 0],
            [-1, -phi, 0],
            [1, -phi, 0],
            [0, -1, phi],
            [0, 1, phi],
            [0, -1, -phi],
            [0, 1, -phi],
            [phi, 0, -1],
            [phi, 0, 1],
            [-phi, 0, -1],
            [-phi, 0, 1],
        ],
        dtype=np.float32,
    )
    vertices /= np.linalg.norm(vertices[0])

    faces = np.array(
        [
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ],
        dtype=np.int32,
    )

    # Subdivide
    for _ in range(subdivisions):
        vertices, faces = _subdivide_icosphere(vertices, faces)

    # Scale by radius
    vertices = vertices * radius

    # For a sphere, normals are just normalized vertex positions
    normals = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)

    return vertices, faces, normals.astype(np.float32)


def _subdivide_icosphere(
    vertices: npt.NDArray[np.float32], faces: npt.NDArray[np.int32]
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int32]]:
    """Subdivide icosphere faces."""
    edge_midpoints = {}
    new_faces = []

    def get_midpoint(i1: int, i2: int) -> int:
        key = (min(i1, i2), max(i1, i2))
        if key in edge_midpoints:
            return edge_midpoints[key]
        mid = (vertices[i1] + vertices[i2]) / 2
        mid = mid / np.linalg.norm(mid)  # Project to unit sphere
        idx = len(vertices) + len(edge_midpoints)
        edge_midpoints[key] = idx
        return idx

    for f in faces:
        v0, v1, v2 = f
        a = get_midpoint(v0, v1)
        b = get_midpoint(v1, v2)
        c = get_midpoint(v2, v0)
        new_faces.extend([[v0, a, c], [v1, b, a], [v2, c, b], [a, b, c]])

    # Build new vertices array
    new_verts = [vertices[i] for i in range(len(vertices))]
    for key in sorted(edge_midpoints.keys(), key=lambda k: edge_midpoints[k]):
        i1, i2 = key
        mid = (vertices[i1] + vertices[i2]) / 2
        mid = mid / np.linalg.norm(mid)
        new_verts.append(mid)

    return np.array(new_verts, dtype=np.float32), np.array(new_faces, dtype=np.int32)


def create_cylinder(
    radius: float = 1.0, height: float = 2.0, segments: int = 32
) -> MeshTuple:
    """Create a cylinder mesh centered at origin, aligned with Z axis.

    Returns (vertices, faces, normals).
    """
    half_h = height / 2
    angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)

    # Side vertices (need duplicates for sharp normals at caps)
    side_verts_bottom = np.column_stack(
        [radius * np.cos(angles), radius * np.sin(angles), np.full(segments, -half_h)]
    )
    side_verts_top = np.column_stack(
        [radius * np.cos(angles), radius * np.sin(angles), np.full(segments, half_h)]
    )

    # Cap center vertices
    center_bottom = np.array([[0, 0, -half_h]])
    center_top = np.array([[0, 0, half_h]])

    # Cap edge vertices (separate from side for different normals)
    cap_verts_bottom = side_verts_bottom.copy()
    cap_verts_top = side_verts_top.copy()

    vertices = np.vstack(
        [
            side_verts_bottom,  # 0 to segments-1
            side_verts_top,  # segments to 2*segments-1
            center_bottom,  # 2*segments
            center_top,  # 2*segments+1
            cap_verts_bottom,  # 2*segments+2 to 3*segments+1
            cap_verts_top,  # 3*segments+2 to 4*segments+1
        ]
    ).astype(np.float32)

    faces = []

    # Side faces
    for i in range(segments):
        i_next = (i + 1) % segments
        faces.append([i, i_next, i_next + segments])
        faces.append([i, i_next + segments, i + segments])

    # Bottom cap
    center_bottom_idx = 2 * segments
    cap_bottom_start = 2 * segments + 2
    for i in range(segments):
        i_next = (i + 1) % segments
        faces.append(
            [center_bottom_idx, cap_bottom_start + i_next, cap_bottom_start + i]
        )

    # Top cap
    center_top_idx = 2 * segments + 1
    cap_top_start = 3 * segments + 2
    for i in range(segments):
        i_next = (i + 1) % segments
        faces.append([center_top_idx, cap_top_start + i, cap_top_start + i_next])

    faces = np.array(faces, dtype=np.int32)

    # Normals
    normals = np.zeros_like(vertices)
    # Side normals (radial)
    for i in range(segments):
        n = np.array([np.cos(angles[i]), np.sin(angles[i]), 0])
        normals[i] = n
        normals[i + segments] = n
    # Cap normals
    normals[2 * segments] = [0, 0, -1]  # bottom center
    normals[2 * segments + 1] = [0, 0, 1]  # top center
    for i in range(segments):
        normals[2 * segments + 2 + i] = [0, 0, -1]  # bottom cap edge
        normals[3 * segments + 2 + i] = [0, 0, 1]  # top cap edge

    return vertices, faces, normals.astype(np.float32)


def create_capsule(
    radius: float = 1.0, height: float = 2.0, segments: int = 32, rings: int = 8
) -> MeshTuple:
    """Create a capsule mesh centered at origin, aligned with Z axis.

    Returns (vertices, faces, normals).
    """
    half_h = height / 2
    verts = []
    norms = []

    # Bottom hemisphere (from bottom pole to equator)
    for i in range(rings + 1):
        phi = (np.pi / 2) * (rings - i) / rings  # from pi/2 (pole) to 0 (equator)
        r = radius * np.cos(phi)
        z = -half_h - radius * np.sin(
            phi
        )  # from -half_h - radius (pole) to -half_h (equator)
        for j in range(segments):
            theta = 2 * np.pi * j / segments
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            verts.append([x, y, z])
            # Normal points outward (downward for bottom hemisphere)
            nx = np.cos(phi) * np.cos(theta)
            ny = np.cos(phi) * np.sin(theta)
            nz = -np.sin(phi)
            norms.append([nx, ny, nz])

    bottom_rows = rings + 1

    # Top hemisphere
    for i in range(rings + 1):
        phi = (np.pi / 2) * i / rings  # from 0 to pi/2
        r = radius * np.cos(phi)
        z = half_h + radius * np.sin(phi)
        for j in range(segments):
            theta = 2 * np.pi * j / segments
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            verts.append([x, y, z])
            nx = np.cos(phi) * np.cos(theta)
            ny = np.cos(phi) * np.sin(theta)
            nz = np.sin(phi)
            norms.append([nx, ny, nz])

    vertices = np.array(verts, dtype=np.float32)
    normals = np.array(norms, dtype=np.float32)

    faces = []

    # Bottom hemisphere faces
    for i in range(rings):
        for j in range(segments):
            j_next = (j + 1) % segments
            v0 = i * segments + j
            v1 = i * segments + j_next
            v2 = (i + 1) * segments + j_next
            v3 = (i + 1) * segments + j
            faces.append([v0, v1, v2])
            faces.append([v0, v2, v3])

    # Cylinder section (connect bottom hemisphere edge to top hemisphere edge)
    bottom_edge_start = rings * segments
    top_edge_start = bottom_rows * segments
    for j in range(segments):
        j_next = (j + 1) % segments
        v0 = bottom_edge_start + j
        v1 = bottom_edge_start + j_next
        v2 = top_edge_start + j_next
        v3 = top_edge_start + j
        faces.append([v0, v1, v2])
        faces.append([v0, v2, v3])

    # Top hemisphere faces
    for i in range(rings):
        for j in range(segments):
            j_next = (j + 1) % segments
            v0 = bottom_rows * segments + i * segments + j
            v1 = bottom_rows * segments + i * segments + j_next
            v2 = bottom_rows * segments + (i + 1) * segments + j_next
            v3 = bottom_rows * segments + (i + 1) * segments + j
            faces.append([v0, v1, v2])
            faces.append([v0, v2, v3])

    return vertices, np.array(faces, dtype=np.int32), normals


def create_box(
    half_extents: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> MeshTuple:
    """Create a box mesh centered at origin.

    Returns (vertices, faces, normals).
    """
    hx, hy, hz = half_extents

    # 24 vertices (4 per face for sharp edges)
    vertices = np.array(
        [
            # Front face (+Z)
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz],
            # Back face (-Z)
            [hx, -hy, -hz],
            [-hx, -hy, -hz],
            [-hx, hy, -hz],
            [hx, hy, -hz],
            # Top face (+Y)
            [-hx, hy, hz],
            [hx, hy, hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
            # Bottom face (-Y)
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, -hy, hz],
            [-hx, -hy, hz],
            # Right face (+X)
            [hx, -hy, hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [hx, hy, hz],
            # Left face (-X)
            [-hx, -hy, -hz],
            [-hx, -hy, hz],
            [-hx, hy, hz],
            [-hx, hy, -hz],
        ],
        dtype=np.float32,
    )

    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],  # Front
            [4, 5, 6],
            [4, 6, 7],  # Back
            [8, 9, 10],
            [8, 10, 11],  # Top
            [12, 13, 14],
            [12, 14, 15],  # Bottom
            [16, 17, 18],
            [16, 18, 19],  # Right
            [20, 21, 22],
            [20, 22, 23],  # Left
        ],
        dtype=np.int32,
    )

    normals = np.array(
        [
            # Front
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            # Back
            [0, 0, -1],
            [0, 0, -1],
            [0, 0, -1],
            [0, 0, -1],
            # Top
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            # Bottom
            [0, -1, 0],
            [0, -1, 0],
            [0, -1, 0],
            [0, -1, 0],
            # Right
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            # Left
            [-1, 0, 0],
            [-1, 0, 0],
            [-1, 0, 0],
            [-1, 0, 0],
        ],
        dtype=np.float32,
    )

    return vertices, faces, normals


class MJCFLogger:
    """Class to log a MJCF model to Rerun."""

    def __init__(
        self,
        model_or_path: str | pathlib.Path | mujoco.MjModel,
        entity_path_prefix: str = "",
    ) -> None:
        self.model: mujoco.MjModel = (
            model_or_path
            if isinstance(model_or_path, mujoco.MjModel)
            else mujoco.MjModel.from_xml_path(str(model_or_path))
        )
        self.entity_path_prefix = entity_path_prefix
        self._body_paths: list[str] = []

    def _add_entity_path_prefix(self, entity_path: str) -> str:
        """Add prefix (if passed) to entity path."""
        if self.entity_path_prefix:
            return f"{self.entity_path_prefix}/{entity_path}"
        return entity_path

    def _is_visual_geom(self, geom: mujoco.MjsGeom) -> bool:
        """Check if geom is a visual-only geom (not for collision).

        Collision class convention (MuJoCo Menagerie style):
        - "visual" class: contype="0" conaffinity="0" group="2" (rendering only, no collision)
        - "collision" class: group="3" (physics simulation, collision enabled by default)
        """
        return (geom.contype.item() == 0 and geom.conaffinity.item() == 0) and (
            geom.group.item() != 3
        )

    def log_model(self, recording: rr.RecordingStream | None = None) -> None:
        """Log MJCF model geometry to Rerun.

        Creates MjData internally to compute forward kinematics and set initial transforms.
        """
        self._body_paths = []

        # First pass: collect body paths
        for body_id in range(self.model.nbody):
            body = self.model.body(body_id)
            body_name = body.name if body.name else "world"
            body_path = self._add_entity_path_prefix(body_name)
            self._body_paths.append(body_path)

        # Group geoms by body and separate visual from collision
        body_geoms: dict[int, list] = {i: [] for i in range(self.model.nbody)}
        for geom_id in range(self.model.ngeom):
            geom = self.model.geom(geom_id)
            body_id = geom.bodyid.item()
            body_geoms[body_id].append(geom)

        # Log geoms for each body - prefer visual geoms if available
        for body_id, geoms in body_geoms.items():
            body_name = self.model.body(body_id).name
            body_path = self._add_entity_path_prefix(body_name)

            visual_geoms = [geom for geom in geoms if self._is_visual_geom(geom)]
            if not visual_geoms:
                # No visual geoms, fall back to all geoms
                visual_geoms = geoms

            for geom in visual_geoms:
                geom_name = geom.name if geom.name else f"geom_{geom.id}"
                geom_path = f"{body_path}/{geom_name}"
                self.log_geom(geom_path, geom, recording)

        # Create MjData and compute forward kinematics for initial state
        data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, data)
        mujoco.mj_forward(self.model, data)
        self.log_data(data, recording)

    def log_data(
        self, data: mujoco.MjData, recording: rr.RecordingStream | None = None
    ) -> None:
        """Update body transforms from MjData (simulation state)."""
        for body_id in range(self.model.nbody):
            body = self.model.body(body_id)
            body_name = body.name if body.name else "world"
            body_path = (
                self._body_paths[body_id]
                if body_id < len(self._body_paths)
                else self._add_entity_path_prefix(body_name)
            )

            rr.log(
                body_path,
                rr.Transform3D(
                    translation=data.xpos[body_id],
                    quaternion=quat_wxyz_to_xyzw(data.xquat[body_id]),
                ),
                recording=recording,
            )

    def _get_geom_material(
        self, geom: mujoco.MjsGeom
    ) -> tuple[int, int, npt.NDArray[np.float32]]:
        """Get material info for a geom.

        Returns:
            mat_id: Material ID (-1 if none)
            tex_id: Texture ID (-1 if none)
            rgba: RGBA color array
        """
        mat_id = geom.matid.item()
        tex_id = (
            self.model.mat_texid[mat_id, mujoco.mjtTextureRole.mjTEXROLE_RGB]
            if mat_id != _MJCF_NO_ID
            else _MJCF_NO_ID
        )
        rgba = self.model.mat_rgba[mat_id] if mat_id != _MJCF_NO_ID else geom.rgba
        return mat_id, tex_id, rgba

    def _log_plane_geom(
        self,
        entity_path: str,
        geom: mujoco.MjsGeom,
        mat_id: int,
        tex_id: int,
        recording: rr.RecordingStream,
    ) -> None:
        """Log a plane geom (requires texture)."""
        if tex_id == _MJCF_NO_ID:
            print(f"Warning: Skipping plane geom '{geom.name}' without texture.")
            return

        texrepeat = self.model.mat_texrepeat[mat_id]
        vertices = np.array(
            [
                [-_PLANE_SIZE, -_PLANE_SIZE, 0],
                [_PLANE_SIZE, -_PLANE_SIZE, 0],
                [_PLANE_SIZE, _PLANE_SIZE, 0],
                [-_PLANE_SIZE, _PLANE_SIZE, 0],
            ],
            dtype=np.float32,
        )
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        uvs = np.array(
            [
                [0, 0],
                [texrepeat[0], 0],
                [texrepeat[0], texrepeat[1]],
                [0, texrepeat[1]],
            ],
            dtype=np.float32,
        )
        rr.log(
            entity_path,
            rr.Mesh3D(
                vertex_positions=vertices,
                triangle_indices=faces,
                albedo_texture=self._get_texture(tex_id),
                vertex_texcoords=uvs,
            ),
            static=True,
            recording=recording,
        )

    def _log_mesh_geom(
        self,
        entity_path: str,
        geom: mujoco.MjsGeom,
        tex_id: int,
        rgba: npt.NDArray[np.float32],
        recording: rr.RecordingStream,
    ) -> None:
        """Log a mesh geom."""
        mesh_id = geom.dataid.item()
        vertices, faces, normals, texcoords = self._get_mesh_data(mesh_id)

        if tex_id != _MJCF_NO_ID and texcoords is not None:
            rr.log(
                entity_path,
                rr.Mesh3D(
                    vertex_positions=vertices,
                    triangle_indices=faces,
                    vertex_normals=normals,
                    albedo_texture=self._get_texture(tex_id),
                    vertex_texcoords=texcoords,
                ),
                static=True,
                recording=recording,
            )
        else:
            vertex_colors = np.tile((rgba * 255).astype(np.uint8), (len(vertices), 1))
            rr.log(
                entity_path,
                rr.Mesh3D(
                    vertex_positions=vertices,
                    triangle_indices=faces,
                    vertex_normals=normals,
                    vertex_colors=vertex_colors,
                ),
                static=True,
                recording=recording,
            )

        rr.log(
            entity_path,
            rr.Transform3D(
                translation=geom.pos, quaternion=quat_wxyz_to_xyzw(geom.quat)
            ),
            static=True,
            recording=recording,
        )

    def _create_primitive_mesh(self, geom: mujoco.MjsGeom) -> MeshTuple:
        """Create mesh data for primitive geom types."""
        geom_type = geom.type.item()

        match geom_type:
            case mujoco.mjtGeom.mjGEOM_SPHERE:
                radius, _, _ = geom.size
                return create_sphere(radius)

            case mujoco.mjtGeom.mjGEOM_CAPSULE:
                radius, height, _ = geom.size
                return create_capsule(radius=radius, height=2 * height)

            case mujoco.mjtGeom.mjGEOM_CYLINDER:
                radius, height, _ = geom.size
                return create_cylinder(radius=radius, height=2 * height)

            case mujoco.mjtGeom.mjGEOM_BOX:
                return create_box(half_extents=geom.size)

            case mujoco.mjtGeom.mjGEOM_ELLIPSOID:
                vertices, faces, normals = create_sphere()
                radii = np.array(geom.size, dtype=np.float32)
                vertices = vertices * radii
                normals = normals / radii
                normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
                return vertices, faces, normals.astype(np.float32)

            case _:
                raise NotImplementedError(
                    f"Unsupported geom type: {geom_type} ({mujoco.mjtGeom(geom_type).name}) "
                    f"for geom '{geom.name}'"
                )

    def log_geom(
        self,
        entity_path: str,
        geom: mujoco.MjsGeom,
        recording: rr.RecordingStream,
    ) -> None:
        """Log a single geom to Rerun."""
        geom_type = geom.type.item()
        mat_id, tex_id, rgba = self._get_geom_material(geom)

        match geom_type:
            case mujoco.mjtGeom.mjGEOM_PLANE:
                self._log_plane_geom(entity_path, geom, mat_id, tex_id, recording)
            case mujoco.mjtGeom.mjGEOM_MESH:
                self._log_mesh_geom(entity_path, geom, tex_id, rgba, recording)
            case _:
                # Handle primitive geometry types
                vertices, faces, normals = self._create_primitive_mesh(geom)
                vertex_colors = np.tile(
                    (rgba * 255).astype(np.uint8), (len(vertices), 1)
                )

                rr.log(
                    entity_path,
                    rr.Mesh3D(
                        vertex_positions=vertices,
                        triangle_indices=faces,
                        vertex_normals=normals,
                        vertex_colors=vertex_colors,
                    ),
                    static=True,
                    recording=recording,
                )
                rr.log(
                    entity_path,
                    rr.Transform3D(
                        translation=geom.pos, quaternion=quat_wxyz_to_xyzw(geom.quat)
                    ),
                    static=True,
                    recording=recording,
                )

    def _get_mesh_data(
        self, mesh_id: int
    ) -> tuple[
        npt.NDArray[np.float32],
        npt.NDArray[np.int32],
        npt.NDArray[np.float32],
        npt.NDArray[np.float32] | None,
    ]:
        """Get mesh vertices, faces, normals, and optionally texture coordinates.

        Returns:
            vertices: (N, 3) array of vertex positions
            faces: (M, 3) array of triangle indices
            normals: (N, 3) array of vertex normals
            texcoords: (N, 2) array of UV coordinates, or None if no texture coords
        """
        if mesh_id == _MJCF_NO_ID:
            raise ValueError("Cannot get mesh data: mesh_id is MJCF_NO_ID (-1)")
        if mesh_id >= self.model.nmesh:
            raise ValueError(
                f"Invalid mesh ID {mesh_id}: model only has {self.model.nmesh} meshes"
            )

        vertadr = self.model.mesh(mesh_id).vertadr.item()
        vertnum = self.model.mesh(mesh_id).vertnum.item()
        vertices = self.model.mesh_vert[vertadr : vertadr + vertnum]
        normals = self.model.mesh_normal[vertadr : vertadr + vertnum]

        faceadr = self.model.mesh(mesh_id).faceadr.item()
        facenum = self.model.mesh(mesh_id).facenum.item()
        faces = self.model.mesh_face[faceadr : faceadr + facenum]

        texcoordadr = self.model.mesh(mesh_id).texcoordadr.item()
        texcoords = (
            np.ascontiguousarray(
                self.model.mesh_texcoord[texcoordadr : texcoordadr + vertnum]
            ).astype(np.float32)
            if texcoordadr != _MJCF_NO_ID
            else None
        )

        return vertices, faces, normals, texcoords

    def _get_texture(self, tex_id: int) -> npt.NDArray[np.uint8]:
        """Extract texture data from MjModel."""
        return self.model.tex(tex_id).data


def quat_wxyz_to_xyzw(quat: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Convert quaternion from wxyz (MuJoCo) to xyzw (Rerun) format."""
    return np.array([quat[1], quat[2], quat[3], quat[0]])


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="""
    This is an executable data-loader plugin for the Rerun Viewer for MJCF files.
        """
    )
    parser.add_argument("filepath", type=str)
    parser.add_argument(
        "--application-id", type=str, help="Recommended ID for the application"
    )
    parser.add_argument(
        "--recording-id", type=str, help="optional recommended ID for the recording"
    )
    args = parser.parse_args()

    filepath = pathlib.Path(args.filepath)

    if not filepath.is_file() or filepath.suffix != ".xml":
        exit(rr.EXTERNAL_DATA_LOADER_INCOMPATIBLE_EXIT_CODE)

    app_id = args.application_id if args.application_id else str(filepath)

    rr.init(app_id, recording_id=args.recording_id, spawn=True)

    mjcf_logger = MJCFLogger(filepath)
    mjcf_logger.log_model()


if __name__ == "__main__":
    main()
