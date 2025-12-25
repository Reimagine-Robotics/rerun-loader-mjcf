# rerun-loader-mjcf
[![CI](https://github.com/Reimagine-Robotics/rerun-loader-mjcf/actions/workflows/ci.yml/badge.svg)](https://github.com/Reimagine-Robotics/rerun-loader-mjcf/actions/workflows/ci.yml)

A [Rerun](https://rerun.io/) external data loader for MJCF (MuJoCo XML) files.

https://github.com/user-attachments/assets/a9f95ed6-1441-4ce0-bef8-c3fb1e720d56

*Simulating `fourier_n1` and `boston_dynamics_spot` from [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie):*
```bash
uv run rerun-loader-mjcf --simulate mujoco_menagerie/fourier_n1/scene.xml
uv run rerun-loader-mjcf --simulate mujoco_menagerie/boston_dynamics_spot/scene.xml
```

https://github.com/user-attachments/assets/36aab5aa-134f-49a3-92d4-efb6b61e9354

*Loading all robots from [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie)*

## Installation

```bash
pip install rerun-loader-mjcf
```

## Usage

### CLI

```bash
rerun-loader-mjcf robot.xml
```

To run a real-time simulation loop:

```bash
rerun-loader-mjcf robot.xml --simulate
```

Or run directly without installing:

```bash
uvx rerun-loader-mjcf robot.xml
```

### Python API

```python
import mujoco
import rerun as rr
import rerun_loader_mjcf

model = mujoco.MjModel.from_xml_path("robot.xml")
data = mujoco.MjData(model)

rr.init("mjcf_viewer", spawn=True)
logger = rerun_loader_mjcf.MJCFLogger(model)

rr.set_time("frame", sequence=0)
logger.log_model()

data.qpos[0] += 0.5
mujoco.mj_forward(model, data)

rr.set_time("frame", sequence=1)
logger.log_data(data)
```

### Recording Simulations

For efficient batch recording of simulations, use `MJCFRecorder`:

```python
import mujoco
import rerun as rr
import rerun_loader_mjcf

model = mujoco.MjModel.from_xml_path("robot.xml")
data = mujoco.MjData(model)

rr.init("simulation", spawn=True)

logger = rerun_loader_mjcf.MJCFLogger(model)
logger.log_model()

# With simulation time (default: uses duration=data.time)
with rerun_loader_mjcf.MJCFRecorder(logger) as recorder:
    while data.time < 5.0:
        mujoco.mj_step(model, data)
        recorder.record(data)

# With explicit sequence index
with rerun_loader_mjcf.MJCFRecorder(logger, timeline_name="frame") as recorder:
    for i in range(1000):
        mujoco.mj_step(model, data)
        recorder.record(data, sequence=i)

# With explicit timestamp
with rerun_loader_mjcf.MJCFRecorder(logger, timeline_name="sim_time") as recorder:
    while data.time < 5.0:
        mujoco.mj_step(model, data)
        recorder.record(data, timestamp=data.time)
```

## Lint

```bash
uv run pre-commit run -a
```

## Credits

Inspired by [rerun-loader-python-example-urdf](https://github.com/rerun-io/rerun-loader-python-example-urdf).

## What's Next

- Integrate [mujoco-rs](https://github.com/jafarAbdi/mujoco-rs) to implement a native Rust loader similar to [loader_urdf.rs](https://github.com/rerun-io/rerun/blob/main/crates/store/re_data_loader/src/loader_urdf.rs)
