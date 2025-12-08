from __future__ import annotations

import pathlib

import mujoco
import numpy as np
import numpy.typing as npt
import rerun as rr

# MuJoCo uses -1 to indicate "no reference" for IDs (material, texture, mesh, etc.)
_MJCF_NO_ID = -1
_PLANE_SIZE = 5.0


class MJCFLogger:
    """Class to log a MJCF model to Rerun."""

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
        self.entity_path_prefix = entity_path_prefix
        self.opacity = opacity
        self._body_paths: list[str] = []

    def _add_entity_path_prefix(self, entity_path: str) -> str:
        """Add prefix (if passed) to entity path."""
        if self.entity_path_prefix:
            return f"{self.entity_path_prefix}/{entity_path}"
        return entity_path

    def _get_albedo_factor(self) -> list[float] | None:
        """Get albedo factor for transparency if opacity is set."""
        if self.opacity is None:
            return None
        return [1.0, 1.0, 1.0, self.opacity]

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

        rr.log(
            entity_path,
            rr.Transform3D(
                translation=geom.pos, quaternion=quat_wxyz_to_xyzw(geom.quat)
            ),
            static=True,
            recording=recording,
        )

    def _log_primitive_geom(
        self,
        entity_path: str,
        geom: mujoco.MjsGeom,
        rgba: npt.NDArray[np.float32],
        recording: rr.RecordingStream,
    ) -> None:
        """Log primitive geometry using Rerun's native primitives."""
        geom_type = geom.type.item()
        color = (rgba * 255).astype(np.uint8)

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
                    static=True,
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
                    static=True,
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
                    static=True,
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
                    static=True,
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
                    static=True,
                    recording=recording,
                )

            case _:
                raise NotImplementedError(
                    f"Unsupported geom type: {geom_type} ({mujoco.mjtGeom(geom_type).name}) "
                    f"for geom '{geom.name}'"
                )

        rr.log(
            entity_path,
            rr.Transform3D(
                translation=geom.pos, quaternion=quat_wxyz_to_xyzw(geom.quat)
            ),
            static=True,
            recording=recording,
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
