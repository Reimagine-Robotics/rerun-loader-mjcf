"""MJCF (MuJoCo XML) loader for Rerun visualization.

This module provides classes for logging MuJoCo models and simulation data to Rerun.

Entity Path Structure:
    /prefix/visual_geometries/{body}/{geom}     - Visual geometries
    /prefix/collision_geometries/{body}/{geom}  - Collision geometries
    /prefix/body_transforms                     - All body transforms (world-frame)

MuJoCo provides absolute world-frame transforms, so all bodies use "world" as parent_frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import mujoco
import numpy as np
import rerun as rr

if TYPE_CHECKING:
    import pathlib

    import numpy.typing as npt

_MJCF_NO_ID = -1
_PLANE_EXTENT_MULTIPLIER = 1.0
_WORLD_FRAME = "world"


def _quat_wxyz_to_xyzw(quat: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Convert quaternion from wxyz (MuJoCo) to xyzw (Rerun) format."""
    return np.array([quat[1], quat[2], quat[3], quat[0]])


@dataclass
class MJCFLogPaths:
    """Entity paths for MJCF logging."""

    root: str
    visual_root: str = field(init=False)
    collision_root: str = field(init=False)
    bodies_root: str = field(init=False)

    def __post_init__(self) -> None:
        base = self.root.rstrip("/") if self.root else ""
        self.visual_root = f"{base}/visual_geometries" if base else "visual_geometries"
        self.collision_root = f"{base}/collision_geometries" if base else "collision_geometries"
        self.bodies_root = f"{base}/bodies" if base else "bodies"

    def body_path(self, body_name: str) -> str:
        """Entity path for body transform."""
        return f"{self.bodies_root}/{body_name}"

    def body_frame(self, body_name: str) -> str:
        """Implicit frame ID for body (tf#/entity/path format)."""
        return f"tf#/{self.body_path(body_name)}"


class MJCFLogger:
    """Log a MJCF model to Rerun.

    Example:
        model = mujoco.MjModel.from_xml_path("robot.xml")
        logger = MJCFLogger(model)
        logger.log_model()

        data = mujoco.MjData(model)
        mujoco.mj_step(model, data)
        logger.log_data(data)
    """

    def __init__(
        self,
        model_or_path: str | pathlib.Path | mujoco.MjModel,
        entity_path_prefix: str = "",
        opacity: float | None = None,
    ) -> None:
        self.model: mujoco.MjModel = (
            model_or_path
            if isinstance(model_or_path, mujoco.MjModel)
            else mujoco.MjModel.from_xml_path(str(model_or_path))
        )
        self.opacity = opacity
        self.paths = MJCFLogPaths(entity_path_prefix)
        self.body_names: list[str] = []

    def _get_albedo_factor(self) -> list[float] | None:
        if self.opacity is None:
            return None
        return [1.0, 1.0, 1.0, self.opacity]

    def _is_visual_geom(self, geom: mujoco.MjsGeom) -> bool:
        """Check if geom is visual-only (not for collision)."""
        return (geom.contype.item() == 0 and geom.conaffinity.item() == 0) and (
            geom.group.item() != 3
        )

    def log_model(self, recording: rr.RecordingStream | None = None) -> None:
        """Log MJCF model geometry to Rerun."""
        # Collect body names
        self.body_names = []
        for body_id in range(self.model.nbody):
            body = self.model.body(body_id)
            self.body_names.append(body.name if body.name else _WORLD_FRAME)

        # Group geoms by body and classify
        body_visual_geoms: dict[int, list] = {i: [] for i in range(self.model.nbody)}
        body_collision_geoms: dict[int, list] = {i: [] for i in range(self.model.nbody)}

        for geom_id in range(self.model.ngeom):
            geom = self.model.geom(geom_id)
            body_id = geom.bodyid.item()
            if self._is_visual_geom(geom):
                body_visual_geoms[body_id].append(geom)
            else:
                body_collision_geoms[body_id].append(geom)

        # Log geometries with CoordinateFrame pointing to body's implicit frame
        for body_id in range(self.model.nbody):
            body_name = self.body_names[body_id]
            body_frame = self.paths.body_frame(body_name)

            # Visual geometries (fall back to collision if no visual)
            visual_geoms = body_visual_geoms[body_id]
            if not visual_geoms:
                visual_geoms = body_collision_geoms[body_id]

            for geom in visual_geoms:
                geom_name = geom.name if geom.name else f"geom_{geom.id}"
                entity_path = f"{self.paths.visual_root}/{body_name}/{geom_name}"
                self._log_geom_with_frame(entity_path, geom, body_frame, recording)

            # Collision geometries
            for geom in body_collision_geoms[body_id]:
                geom_name = geom.name if geom.name else f"geom_{geom.id}"
                entity_path = f"{self.paths.collision_root}/{body_name}/{geom_name}"
                self._log_geom_with_frame(entity_path, geom, body_frame, recording)

        # Log initial transforms
        data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, data)
        mujoco.mj_forward(self.model, data)
        self.log_data(data, recording)

    def log_data(
        self, data: mujoco.MjData, recording: rr.RecordingStream | None = None
    ) -> None:
        """Update body transforms from simulation state."""
        for body_id in range(self.model.nbody):
            body_name = self.body_names[body_id]
            rr.log(
                self.paths.body_path(body_name),
                rr.Transform3D(
                    translation=data.xpos[body_id],
                    quaternion=_quat_wxyz_to_xyzw(data.xquat[body_id]),
                ),
                recording=recording,
            )

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
                quaternions=[_quat_wxyz_to_xyzw(geom.quat)],
            ),
            static=True,
            recording=recording,
        )

        # Log geometry
        geom_type = geom.type.item()
        mat_id, tex_id, rgba = self._get_geom_material(geom)

        match geom_type:
            case mujoco.mjtGeom.mjGEOM_PLANE:
                self._log_plane(entity_path, geom, mat_id, tex_id, recording)
            case mujoco.mjtGeom.mjGEOM_MESH:
                self._log_mesh(entity_path, geom, tex_id, rgba, recording)
            case _:
                self._log_primitive(entity_path, geom, rgba, recording)

    def _get_geom_material(
        self, geom: mujoco.MjsGeom
    ) -> tuple[int, int, npt.NDArray[np.float32]]:
        mat_id = geom.matid.item()
        tex_id = (
            self.model.mat_texid[mat_id, mujoco.mjtTextureRole.mjTEXROLE_RGB]
            if mat_id != _MJCF_NO_ID
            else _MJCF_NO_ID
        )
        rgba = self.model.mat_rgba[mat_id] if mat_id != _MJCF_NO_ID else geom.rgba
        return mat_id, tex_id, rgba

    def _get_mesh_data(
        self, mesh_id: int
    ) -> tuple[
        npt.NDArray[np.float32],
        npt.NDArray[np.int32],
        npt.NDArray[np.float32],
        npt.NDArray[np.float32] | None,
    ]:
        mesh = self.model.mesh(mesh_id)
        vertadr, vertnum = mesh.vertadr.item(), mesh.vertnum.item()
        faceadr, facenum = mesh.faceadr.item(), mesh.facenum.item()
        texcoordadr = mesh.texcoordadr.item()

        vertices = self.model.mesh_vert[vertadr : vertadr + vertnum]
        normals = self.model.mesh_normal[vertadr : vertadr + vertnum]
        faces = self.model.mesh_face[faceadr : faceadr + facenum]
        texcoords = (
            np.ascontiguousarray(
                self.model.mesh_texcoord[texcoordadr : texcoordadr + vertnum]
            ).astype(np.float32)
            if texcoordadr != _MJCF_NO_ID
            else None
        )
        return vertices, faces, normals, texcoords

    def _get_texture(self, tex_id: int) -> npt.NDArray[np.uint8]:
        return self.model.tex(tex_id).data

    def _log_plane(
        self,
        entity_path: str,
        geom: mujoco.MjsGeom,
        mat_id: int,
        tex_id: int,
        recording: rr.RecordingStream | None,
    ) -> None:
        if tex_id == _MJCF_NO_ID:
            return

        extent = _PLANE_EXTENT_MULTIPLIER * max(self.model.stat.extent, 1.0)
        hx = geom.size[0] if geom.size[0] > 0 else extent
        hy = geom.size[1] if geom.size[1] > 0 else extent

        texrepeat = self.model.mat_texrepeat[mat_id]
        texuniform = self.model.mat_texuniform[mat_id]

        sx, sy = 2 * hx, 2 * hy
        if texuniform:
            rx, ry = (texrepeat[0] * sx) / 2, (texrepeat[1] * sy) / 2
        else:
            rx, ry = texrepeat[0] / 2, texrepeat[1] / 2

        ux, uy = rx / 2, ry / 2

        vertices = np.array(
            [[-hx, -hy, 0], [hx, -hy, 0], [hx, hy, 0], [-hx, hy, 0]], dtype=np.float32
        )
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        uvs = np.array(
            [
                [-ux + 0.5, -uy + 0.5],
                [ux + 0.5, -uy + 0.5],
                [ux + 0.5, uy + 0.5],
                [-ux + 0.5, uy + 0.5],
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
                albedo_factor=self._get_albedo_factor(),
            ),
            static=True,
            recording=recording,
        )

    def _log_mesh(
        self,
        entity_path: str,
        geom: mujoco.MjsGeom,
        tex_id: int,
        rgba: npt.NDArray[np.float32],
        recording: rr.RecordingStream | None,
    ) -> None:
        vertices, faces, normals, texcoords = self._get_mesh_data(geom.dataid.item())

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
            vertex_colors = np.tile((rgba * 255).astype(np.uint8), (len(vertices), 1))
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

    def _log_primitive(
        self,
        entity_path: str,
        geom: mujoco.MjsGeom,
        rgba: npt.NDArray[np.float32],
        recording: rr.RecordingStream | None,
    ) -> None:
        geom_type = geom.type.item()
        color = (rgba * 255).astype(np.uint8)
        if self.opacity is not None:
            color[3] = int(self.opacity * 255)

        match geom_type:
            case mujoco.mjtGeom.mjGEOM_SPHERE:
                r = geom.size[0]
                rr.log(
                    entity_path,
                    rr.Ellipsoids3D(
                        half_sizes=[r, r, r],
                        colors=color,
                        fill_mode=rr.components.FillMode.Solid,
                    ),
                    static=True,
                    recording=recording,
                )
            case mujoco.mjtGeom.mjGEOM_ELLIPSOID:
                rr.log(
                    entity_path,
                    rr.Ellipsoids3D(
                        half_sizes=geom.size,
                        colors=color,
                        fill_mode=rr.components.FillMode.Solid,
                    ),
                    static=True,
                    recording=recording,
                )
            case mujoco.mjtGeom.mjGEOM_BOX:
                rr.log(
                    entity_path,
                    rr.Boxes3D(
                        half_sizes=geom.size,
                        colors=color,
                        fill_mode=rr.components.FillMode.Solid,
                    ),
                    static=True,
                    recording=recording,
                )
            case mujoco.mjtGeom.mjGEOM_CAPSULE:
                r, hl = geom.size[0], geom.size[1]
                rr.log(
                    entity_path,
                    rr.Capsules3D(
                        lengths=2 * hl,
                        radii=r,
                        translations=[0, 0, -hl],
                        colors=color,
                        fill_mode=rr.components.FillMode.Solid,
                    ),
                    static=True,
                    recording=recording,
                )
            case mujoco.mjtGeom.mjGEOM_CYLINDER:
                r, hh = geom.size[0], geom.size[1]
                rr.log(
                    entity_path,
                    rr.Cylinders3D(
                        lengths=2 * hh,
                        radii=r,
                        centers=[0, 0, 0],
                        colors=color,
                        fill_mode=rr.components.FillMode.Solid,
                    ),
                    static=True,
                    recording=recording,
                )
            case _:
                raise NotImplementedError(
                    f"Unsupported geom type: {mujoco.mjtGeom(geom_type).name}"
                )


class MJCFRecorder:
    """Context manager for batched recording of MuJoCo simulations.

    Example:
        logger = MJCFLogger(model)
        logger.log_model()

        with MJCFRecorder(logger) as recorder:
            for _ in range(1000):
                mujoco.mj_step(model, data)
                recorder.record(data)
    """

    def __init__(
        self,
        logger: MJCFLogger,
        timeline_name: str = "sim_time",
        recording: rr.RecordingStream | None = None,
    ) -> None:
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
            body_name = self.logger.body_names[body_id]

            quats_xyzw = np.column_stack(
                [
                    quaternions[:, body_id, 1],
                    quaternions[:, body_id, 2],
                    quaternions[:, body_id, 3],
                    quaternions[:, body_id, 0],
                ]
            )

            rr.send_columns(
                self.logger.paths.body_path(body_name),
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


def main() -> None:
    import argparse
    import pathlib
    import time

    parser = argparse.ArgumentParser(description="Rerun MJCF loader plugin.")
    parser.add_argument("filepath", type=str)
    parser.add_argument("--application-id", type=str)
    parser.add_argument("--recording-id", type=str)
    parser.add_argument("--simulate", action="store_true")
    args = parser.parse_args()

    filepath = pathlib.Path(args.filepath)
    if not filepath.is_file() or filepath.suffix != ".xml":
        exit(rr.EXTERNAL_DATA_LOADER_INCOMPATIBLE_EXIT_CODE)

    app_id = args.application_id or str(filepath)
    rr.init(app_id, recording_id=args.recording_id, spawn=True)

    model = mujoco.MjModel.from_xml_path(str(filepath))
    logger = MJCFLogger(model)
    logger.log_model()

    if not args.simulate:
        return

    data = mujoco.MjData(model)
    log_interval = 1.0 / 30.0
    last_log_time = 0.0

    while True:
        step_start = time.time()
        mujoco.mj_step(model, data)

        if data.time - last_log_time >= log_interval:
            rr.set_time("sim_time", duration=data.time)
            logger.log_data(data)
            last_log_time = data.time

        elapsed = time.time() - step_start
        sleep_time = model.opt.timestep - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


if __name__ == "__main__":
    main()
