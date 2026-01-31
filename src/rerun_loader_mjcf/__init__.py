from __future__ import annotations

import dataclasses
import pathlib
from typing import TYPE_CHECKING

import mujoco
import numpy as np
import rerun as rr

if TYPE_CHECKING:
    import numpy.typing as npt

# MuJoCo uses -1 to indicate "no reference" for IDs (material, texture, mesh, etc.)
_MJCF_NO_ID = -1
# Multiplier for plane extent when size is not specified (affects number of tiles)
_PLANE_EXTENT_MULTIPLIER = 1.0
# MuJoCo collision class convention (Menagerie style)
_VISUAL_CONTYPE = 0
_VISUAL_CONAFFINITY = 0
_COLLISION_GROUP = 3
# MuJoCo's checker texture has 2x2 squares per UV tile
_CHECKER_TILES_PER_UV = 2
# UV offset matching MuJoCo's glTexGenfv: S = 0.5*scl*X - 0.5, T = -0.5*scl*Y - 0.5
_UV_CENTER_OFFSET = -0.5
# RGBA color range
_RGBA_MAX = 255


def _body_name(body: mujoco.MjsBody) -> str:
    """Get body name, defaulting to body_{id} if unnamed."""
    return body.name if body.name else f"body_{body.id}"


def _geom_name(geom: mujoco.MjsGeom) -> str:
    """Get geom name, defaulting to geom_{id} if unnamed."""
    return geom.name if geom.name else f"geom_{geom.id}"


def _is_visual_geom(geom: mujoco.MjsGeom) -> bool:
    """Check if geom is visual-only (not for collision)."""
    return (
        geom.contype.item() == _VISUAL_CONTYPE
        and geom.conaffinity.item() == _VISUAL_CONAFFINITY
    ) and (geom.group.item() != _COLLISION_GROUP)


def _build_body_geoms(model: mujoco.MjModel) -> dict[int, tuple[list, list]]:
    """Build mapping of body_id -> (visual_geoms, collision_geoms)."""
    visual: dict[int, list] = {i: [] for i in range(model.nbody)}
    collision: dict[int, list] = {i: [] for i in range(model.nbody)}
    for geom_id in range(model.ngeom):
        geom = model.geom(geom_id)
        body_id = geom.bodyid.item()
        if _is_visual_geom(geom):
            visual[body_id].append(geom)
        else:
            collision[body_id].append(geom)
    return {i: (visual[i], collision[i]) for i in range(model.nbody)}


@dataclasses.dataclass
class MJCFLogPaths:
    """Entity paths for MJCF logging."""

    root: str
    visual_root: str = dataclasses.field(init=False)
    collision_root: str = dataclasses.field(init=False)
    bodies_root: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        base = self.root.rstrip("/") if self.root else ""
        self.visual_root = f"{base}/visual_geometries" if base else "visual_geometries"
        self.collision_root = (
            f"{base}/collision_geometries" if base else "collision_geometries"
        )
        self.bodies_root = f"{base}/bodies" if base else "bodies"

    def body_path(self, body_name: str) -> str:
        """Entity path for body transform."""
        return f"{self.bodies_root}/{body_name}"

    def body_frame(self, body_name: str) -> str:
        """Implicit frame ID for body (tf#/entity/path format)."""
        return f"tf#/{self.body_path(body_name)}"


class MJCFLogger:
    """Class to log a MJCF model to Rerun."""

    def __init__(
        self,
        model_or_path: str | pathlib.Path | mujoco.MjModel,
        entity_path_prefix: str = "",
        opacity: float | None = None,
        log_collision: bool = False,
    ) -> None:
        self.model: mujoco.MjModel = (
            model_or_path
            if isinstance(model_or_path, mujoco.MjModel)
            else mujoco.MjModel.from_xml_path(str(model_or_path))
        )
        self.opacity = opacity
        self.log_collision = log_collision
        self.paths = MJCFLogPaths(entity_path_prefix)
        self._body_geoms = _build_body_geoms(self.model)

    def _get_albedo_factor(self) -> list[float] | None:
        """Get albedo factor for transparency if opacity is set."""
        if self.opacity is None:
            return None
        return [1.0, 1.0, 1.0, self.opacity]

    def log_model(
        self,
        recording: rr.RecordingStream | None = None,
    ) -> None:
        """Log MJCF model geometry to Rerun.

        Creates MjData internally to compute forward kinematics and set initial transforms.

        Note:
            Call `rr.set_time()` before this method to log geometry on a specific timeline.
            Example: `rr.set_time("sim_time", duration=0.0)` to log at t=0 on sim_time.

        Args:
            recording: Optional Rerun recording stream.
        """
        for body_id in range(self.model.nbody):
            body = self.model.body(body_id)
            body_name = _body_name(body)
            body_frame = self.paths.body_frame(body_name)
            visual_geoms, collision_geoms = self._body_geoms[body_id]

            # Visual geometries (fall back to collision if no visual)
            for geom in visual_geoms or collision_geoms:
                geom_name = _geom_name(geom)
                entity_path = f"{self.paths.visual_root}/{body_name}/{geom_name}"
                self._log_geom_with_frame(entity_path, geom, body_frame, recording)

            # Collision geometries
            if self.log_collision:
                for geom in collision_geoms:
                    geom_name = _geom_name(geom)
                    entity_path = f"{self.paths.collision_root}/{body_name}/{geom_name}"
                    self._log_geom_with_frame(entity_path, geom, body_frame, recording)

        # Create MjData and compute forward kinematics for initial state
        data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, data)
        mujoco.mj_forward(self.model, data)
        self.log_data(data, recording)

    def log_data(
        self, data: mujoco.MjData, recording: rr.RecordingStream | None = None
    ) -> None:
        """Update body transforms from MjData (simulation state).

        Note:
            Call `rr.set_time()` before this method to log on a specific timeline.
            Example: `rr.set_time("sim_time", duration=data.time)`

        Args:
            data: MuJoCo simulation data containing current body transforms.
            recording: Optional Rerun recording stream.
        """
        for body_id in range(self.model.nbody):
            body = self.model.body(body_id)
            body_name = _body_name(body)

            rr.log(
                self.paths.body_path(body_name),
                rr.Transform3D(
                    translation=data.xpos[body_id],
                    quaternion=quat_wxyz_to_xyzw(data.xquat[body_id]),
                ),
                recording=recording,
            )

    def _get_visual_geom_path(self, body_id: int, geom: mujoco.MjsGeom) -> str:
        """Get the visual geometry entity path for a geom."""
        body = self.model.body(body_id)
        body_name = _body_name(body)
        geom_name = _geom_name(geom)
        return f"{self.paths.visual_root}/{body_name}/{geom_name}"

    def set_body_color(
        self,
        body_id: int,
        rgba: list[float],
        recording: rr.RecordingStream | None = None,
    ) -> None:
        """Set color for all visual geometries of a body.

        Args:
            body_id: Body index.
            rgba: RGBA color as floats [0.0-1.0].
            recording: Optional Rerun recording stream.

        Note:
            Only affects visual geometries, not collision geometries.
        """
        visual_geoms, collision_geoms = self._body_geoms[body_id]
        rgba = np.array(rgba, dtype=np.float32)

        for geom in visual_geoms or collision_geoms:
            geom_type = geom.type.item()
            if geom_type == mujoco.mjtGeom.mjGEOM_PLANE:
                continue
            path = self._get_visual_geom_path(body_id, geom)
            if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
                rr.log(
                    path,
                    rr.Mesh3D.from_fields(albedo_factor=rgba),
                    recording=recording,
                )
            else:
                self._log_primitive_geom(path, geom, rgba, recording)

    def reset_body_color(
        self,
        body_id: int,
        opacity: float | None = None,
        recording: rr.RecordingStream | None = None,
    ) -> None:
        """Reset body geometries to their original colors.

        Args:
            body_id: Body index.
            opacity: Optional opacity override (0.0-1.0). If provided, applies this
                opacity while preserving original colors.
            recording: Optional Rerun recording stream.

        Note:
            Only affects visual geometries, not collision geometries.
            For meshes, resets albedo_factor to neutral (white) since vertex_colors
            already contain the original color.
            For primitives, restores the original material/geom color.
        """
        visual_geoms, collision_geoms = self._body_geoms[body_id]

        for geom in visual_geoms or collision_geoms:
            geom_type = geom.type.item()
            if geom_type == mujoco.mjtGeom.mjGEOM_PLANE:
                continue
            path = self._get_visual_geom_path(body_id, geom)
            if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
                # Meshes use vertex_colors for base color, so reset albedo_factor to neutral
                alpha = opacity if opacity is not None else 1.0
                rr.log(
                    path,
                    rr.Mesh3D.from_fields(albedo_factor=[1.0, 1.0, 1.0, alpha]),
                    recording=recording,
                )
            else:
                _, _, rgba = self._get_geom_material(geom)
                if opacity is not None:
                    rgba = rgba.copy()
                    rgba[3] = opacity
                self._log_primitive_geom(path, geom, rgba, recording)

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

    def _is_builtin_texture(self, tex_id: int) -> bool:
        """Check if texture is a MuJoCo builtin (generated) texture.

        Builtin textures (checker, gradient, flat) require special UV handling.
        File textures (PNG) use standard UV mapping.

        Uses tex_pathadr: -1 means builtin/generated, >=0 means file path.
        """
        return self.model.tex_pathadr[tex_id] == _MJCF_NO_ID

    def _log_plane_geom(
        self,
        entity_path: str,
        geom: mujoco.MjsGeom,
        mat_id: int,
        tex_id: int,
        recording: rr.RecordingStream,
    ) -> None:
        """Log a plane geom as a textured quad.

        MuJoCo plane geometry:
        - size[0], size[1]: half-extents for rendering (0 means use model extent)
        - size[2]: grid spacing (unused here)

        MuJoCo texture tiling (controlled by material attributes):
        - texrepeat: number of texture repeats
        - texuniform: if True, texrepeat is per spatial unit; if False, across whole plane

        For builtin checker textures: MuJoCo's checker contains a 2x2 grid of squares
        per UV tile, so we divide by 2 and offset to center origin between tiles.

        For file textures (PNG images like AprilTags): use standard UV mapping
        based on texrepeat without checker-specific adjustments.
        """
        if tex_id == _MJCF_NO_ID:
            print(f"Warning: Skipping plane geom '{geom.name}' without texture.")
            return

        # Plane half-extents: use explicit size or fall back to model extent
        extent = _PLANE_EXTENT_MULTIPLIER * max(self.model.stat.extent, 1.0)
        plane_half_x = geom.size[0] if geom.size[0] > 0 else extent
        plane_half_y = geom.size[1] if geom.size[1] > 0 else extent

        texrepeat = self.model.mat_texrepeat[mat_id]
        texuniform = self.model.mat_texuniform[mat_id]
        is_checker = self._is_builtin_texture(tex_id)

        # Calculate UV repeats across the plane
        plane_size_x = 2 * plane_half_x
        plane_size_y = 2 * plane_half_y

        if is_checker:
            # Match MuJoCo's glTexGenfv: S = 0.5*scl*x - 0.5, T = -0.5*scl*y - 0.5
            # Divide scl by 2 for checker's 2x2 pattern per UV tile
            if texuniform:
                scl_x = (texrepeat[0] * plane_size_x) / _CHECKER_TILES_PER_UV
                scl_y = (texrepeat[1] * plane_size_y) / _CHECKER_TILES_PER_UV
            else:
                scl_x = texrepeat[0] / _CHECKER_TILES_PER_UV
                scl_y = texrepeat[1] / _CHECKER_TILES_PER_UV
            # Vertices at normalized positions: -1, +1
            # U = 0.5*scl*x - 0.5, V = -0.5*scl*y - 0.5
            uvs = np.array(
                [
                    [-0.5 * scl_x - 0.5, 0.5 * scl_y - 0.5],  # (-1, -1)
                    [0.5 * scl_x - 0.5, 0.5 * scl_y - 0.5],  # (+1, -1)
                    [0.5 * scl_x - 0.5, -0.5 * scl_y - 0.5],  # (+1, +1)
                    [-0.5 * scl_x - 0.5, -0.5 * scl_y - 0.5],  # (-1, +1)
                ],
                dtype=np.float32,
            )
        else:
            # File texture (PNG): standard UV mapping [0, texrepeat]
            if texuniform:
                uv_max_x = texrepeat[0] * plane_size_x
                uv_max_y = texrepeat[1] * plane_size_y
            else:
                uv_max_x = texrepeat[0]
                uv_max_y = texrepeat[1]
            uvs = np.array(
                [
                    [0, uv_max_y],
                    [uv_max_x, uv_max_y],
                    [uv_max_x, 0],
                    [0, 0],
                ],
                dtype=np.float32,
            )

        vertices = np.array(
            [
                [-plane_half_x, -plane_half_y, 0],
                [plane_half_x, -plane_half_y, 0],
                [plane_half_x, plane_half_y, 0],
                [-plane_half_x, plane_half_y, 0],
            ],
            dtype=np.float32,
        )
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

        rr.log(
            entity_path,
            rr.Mesh3D(
                vertex_positions=vertices,
                triangle_indices=faces,
                albedo_texture=self._get_texture(tex_id),
                vertex_texcoords=uvs,
                albedo_factor=self._get_albedo_factor(),
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
                    albedo_factor=self._get_albedo_factor(),
                ),
                static=True,
                recording=recording,
            )
        else:
            vertex_colors = np.tile(
                (rgba * _RGBA_MAX).astype(np.uint8), (len(vertices), 1)
            )
            rr.log(
                entity_path,
                rr.Mesh3D(
                    vertex_positions=vertices,
                    triangle_indices=faces,
                    vertex_normals=normals,
                    vertex_colors=vertex_colors,
                    albedo_factor=self._get_albedo_factor(),
                ),
                static=True,
                recording=recording,
            )

    def _log_primitive_geom(
        self,
        entity_path: str,
        geom: mujoco.MjsGeom,
        rgba: npt.NDArray[np.float32],
        recording: rr.RecordingStream | None,
    ) -> None:
        """Log a primitive geometry with the given color.

        Re-logs the full geometry because Rerun primitives don't support
        partial color updates like meshes do.
        """
        geom_type = geom.type.item()
        color = (rgba * _RGBA_MAX).astype(np.uint8)
        if self.opacity is not None:
            color[3] = int(self.opacity * _RGBA_MAX)

        match geom_type:
            case mujoco.mjtGeom.mjGEOM_SPHERE:
                radius, _, _ = geom.size
                rr.log(
                    entity_path,
                    rr.Ellipsoids3D(
                        half_sizes=[radius, radius, radius],
                        colors=color,
                        fill_mode=rr.components.FillMode.Solid,
                    ),
                    recording=recording,
                )
            case mujoco.mjtGeom.mjGEOM_ELLIPSOID:
                rx, ry, rz = geom.size
                rr.log(
                    entity_path,
                    rr.Ellipsoids3D(
                        half_sizes=[rx, ry, rz],
                        colors=color,
                        fill_mode=rr.components.FillMode.Solid,
                    ),
                    recording=recording,
                )
            case mujoco.mjtGeom.mjGEOM_BOX:
                hx, hy, hz = geom.size
                rr.log(
                    entity_path,
                    rr.Boxes3D(
                        half_sizes=[hx, hy, hz],
                        colors=color,
                        fill_mode=rr.components.FillMode.Solid,
                    ),
                    recording=recording,
                )
            case mujoco.mjtGeom.mjGEOM_CAPSULE:
                radius, half_length, _ = geom.size
                rr.log(
                    entity_path,
                    rr.Capsules3D(
                        lengths=2 * half_length,
                        radii=radius,
                        translations=[0, 0, -half_length],
                        colors=color,
                        fill_mode=rr.components.FillMode.Solid,
                    ),
                    recording=recording,
                )
            case mujoco.mjtGeom.mjGEOM_CYLINDER:
                radius, half_height, _ = geom.size
                rr.log(
                    entity_path,
                    rr.Cylinders3D(
                        lengths=2 * half_height,
                        radii=radius,
                        centers=[0, 0, 0],
                        colors=color,
                        fill_mode=rr.components.FillMode.Solid,
                    ),
                    recording=recording,
                )
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
                self._log_primitive_geom(entity_path, geom, rgba, recording)

    def _log_geom_with_frame(
        self,
        entity_path: str,
        geom: mujoco.MjsGeom,
        parent_frame: str,
        recording: rr.RecordingStream | None,
    ) -> None:
        """Log geometry with CoordinateFrame and InstancePoses3D."""
        # Attach to parent body's frame
        rr.log(
            entity_path,
            rr.CoordinateFrame(parent_frame),
            static=True,
            recording=recording,
        )

        # Local pose within body frame
        rr.log(
            entity_path,
            rr.InstancePoses3D(
                translations=[geom.pos],
                quaternions=[quat_wxyz_to_xyzw(geom.quat)],
            ),
            static=True,
            recording=recording,
        )

        # Log geometry
        geom_type = geom.type.item()
        mat_id, tex_id, rgba = self._get_geom_material(geom)

        match geom_type:
            case mujoco.mjtGeom.mjGEOM_PLANE:
                self._log_plane_geom(entity_path, geom, mat_id, tex_id, recording)
            case mujoco.mjtGeom.mjGEOM_MESH:
                self._log_mesh_geom(entity_path, geom, tex_id, rgba, recording)
            case _:
                self._log_primitive_geom(entity_path, geom, rgba, recording)

    def _get_mesh_data(
        self, mesh_id: int
    ) -> tuple[
        npt.NDArray[np.float32],
        npt.NDArray[np.int32],
        npt.NDArray[np.float32],
        npt.NDArray[np.float32] | None,
    ]:
        """Get mesh vertices, faces, normals, and optionally texture coordinates.

        MuJoCo stores per-face-vertex normals via mesh_facenormal, which allows
        different normals for the same vertex in different faces (for hard edges).
        We "explode" the mesh so each face corner becomes a unique vertex.

        Returns:
            vertices: (M*3, 3) array of vertex positions (one per face corner)
            faces: (M, 3) array of triangle indices (sequential: 0,1,2,3,4,5,...)
            normals: (M*3, 3) array of normals (one per face corner)
            texcoords: (M*3, 2) array of UV coordinates, or None if no texture coords
        """
        if mesh_id == _MJCF_NO_ID:
            raise ValueError("Cannot get mesh data: mesh_id is MJCF_NO_ID (-1)")
        if mesh_id >= self.model.nmesh:
            raise ValueError(
                f"Invalid mesh ID {mesh_id}: model only has {self.model.nmesh} meshes"
            )

        mesh = self.model.mesh(mesh_id)
        vertadr = mesh.vertadr.item()
        faceadr = mesh.faceadr.item()
        facenum = mesh.facenum.item()
        normaladr = self.model.mesh_normaladr[mesh_id]
        texcoordadr = mesh.texcoordadr.item()

        # Get face indices for vertices, normals, and texcoords
        face_vert = self.model.mesh_face[faceadr : faceadr + facenum]
        face_norm = self.model.mesh_facenormal[faceadr : faceadr + facenum]

        # Explode mesh: create unique vertex for each face corner
        num_verts = facenum * 3
        vertices = np.zeros((num_verts, 3), dtype=np.float32)
        normals = np.zeros((num_verts, 3), dtype=np.float32)

        for i in range(facenum):
            for j in range(3):
                vertices[i * 3 + j] = self.model.mesh_vert[vertadr + face_vert[i, j]]
                normals[i * 3 + j] = self.model.mesh_normal[normaladr + face_norm[i, j]]

        # Sequential face indices
        faces = np.arange(num_verts, dtype=np.int32).reshape(-1, 3)

        # Texcoords (also per-face-vertex via mesh_facetexcoord)
        texcoords = None
        if texcoordadr != _MJCF_NO_ID:
            face_tex = self.model.mesh_facetexcoord[faceadr : faceadr + facenum]
            texcoords = np.zeros((num_verts, 2), dtype=np.float32)
            for i in range(facenum):
                for j in range(3):
                    texcoords[i * 3 + j] = self.model.mesh_texcoord[
                        texcoordadr + face_tex[i, j]
                    ]

        return vertices, faces, normals, texcoords

    def _get_texture(self, tex_id: int) -> npt.NDArray[np.uint8]:
        """Extract texture data from MjModel."""
        return self.model.tex(tex_id).data


class MJCFRecorder:
    """Context manager for efficient batched recording of MuJoCo simulations.

    Uses Rerun's columnar API for better performance with time-series data.

    Example:
        logger = MJCFLogger(model)
        logger.log_model()

        with MJCFRecorder(logger) as recorder:
            for _ in range(1000):
                mujoco.mj_step(model, data)
                recorder.record(data)
        # auto-flushes on exit
    """

    def __init__(
        self,
        logger: MJCFLogger,
        timeline_name: str = "sim_time",
        recording: rr.RecordingStream | None = None,
    ) -> None:
        """Initialize the recorder.

        Args:
            logger: MJCFLogger instance (must have log_model() called first)
            timeline_name: Name of the timeline in Rerun.
            recording: Optional Rerun recording stream.
        """
        self.logger = logger
        self.timeline_name = timeline_name
        self.recording = recording
        self._sequences: list[int] = []
        self._durations: list[float] = []
        self._timestamps: list[float] = []
        self._positions: list[npt.NDArray[np.float64]] = []
        self._quaternions: list[npt.NDArray[np.float64]] = []

    def __enter__(self) -> MJCFRecorder:
        return self

    def __exit__(self, *exc: object) -> bool:
        self.flush()
        return False

    def record(
        self,
        data: mujoco.MjData,
        *,
        sequence: int | None = None,
        duration: float | None = None,
        timestamp: float | None = None,
    ) -> None:
        """Record simulation state for later batched logging.

        Args:
            data: MuJoCo simulation data.
            sequence: Sequence index (mutually exclusive with duration/timestamp).
            duration: Duration in seconds (mutually exclusive with sequence/timestamp).
            timestamp: Timestamp in seconds (mutually exclusive with sequence/duration).

        If none specified, defaults to duration=data.time.
        """
        if sequence is not None:
            self._sequences.append(sequence)
        elif duration is not None:
            self._durations.append(duration)
        elif timestamp is not None:
            self._timestamps.append(timestamp)
        else:
            self._durations.append(data.time)
        self._positions.append(data.xpos.copy())
        self._quaternions.append(data.xquat.copy())

    def flush(self) -> None:
        """Send accumulated data using columnar API."""
        if not self._positions:
            return

        positions = np.array(self._positions)
        quaternions = np.array(self._quaternions)

        if self._sequences:
            indexes = [rr.TimeColumn(self.timeline_name, sequence=self._sequences)]
        elif self._durations:
            indexes = [rr.TimeColumn(self.timeline_name, duration=self._durations)]
        elif self._timestamps:
            indexes = [rr.TimeColumn(self.timeline_name, timestamp=self._timestamps)]
        else:
            raise RuntimeError("No timeline data recorded")

        for body_id in range(self.logger.model.nbody):
            body = self.logger.model.body(body_id)
            body_path = self.logger.paths.body_path(_body_name(body))

            # Convert wxyz (MuJoCo) to xyzw (Rerun) for all timesteps
            quats_xyzw = np.column_stack(
                [
                    quaternions[:, body_id, 1],
                    quaternions[:, body_id, 2],
                    quaternions[:, body_id, 3],
                    quaternions[:, body_id, 0],
                ]
            )

            rr.send_columns(
                body_path,
                indexes=indexes,
                columns=rr.Transform3D.columns(
                    translation=positions[:, body_id],
                    quaternion=quats_xyzw,
                ),
                recording=self.recording,
            )

        self._sequences.clear()
        self._durations.clear()
        self._timestamps.clear()
        self._positions.clear()
        self._quaternions.clear()


def quat_wxyz_to_xyzw(quat: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Convert quaternion from wxyz (MuJoCo) to xyzw (Rerun) format."""
    return np.array([quat[1], quat[2], quat[3], quat[0]])


def main() -> None:
    import argparse
    import time

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
    parser.add_argument(
        "--simulate", action="store_true", help="run real-time simulation loop"
    )
    args = parser.parse_args()

    filepath = pathlib.Path(args.filepath)

    if not filepath.is_file() or filepath.suffix != ".xml":
        exit(rr.EXTERNAL_DATA_LOADER_INCOMPATIBLE_EXIT_CODE)

    app_id = args.application_id if args.application_id else str(filepath)

    rr.init(app_id, recording_id=args.recording_id, spawn=True)

    model = mujoco.MjModel.from_xml_path(str(filepath))
    mjcf_logger = MJCFLogger(model)
    rr.set_time("sim_time", duration=0.0)
    mjcf_logger.log_model()

    if not args.simulate:
        return

    data = mujoco.MjData(model)
    # Default simulation logging rate
    log_fps = 30.0
    log_interval = 1.0 / log_fps
    last_log_time = 0.0

    while True:
        step_start = time.time()

        mujoco.mj_step(model, data)

        if data.time - last_log_time >= log_interval:
            rr.set_time("sim_time", duration=data.time)
            mjcf_logger.log_data(data)
            last_log_time = data.time

        elapsed = time.time() - step_start
        sleep_time = model.opt.timestep - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


if __name__ == "__main__":
    main()
