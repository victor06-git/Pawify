# Pawify Handover

This folder contains a small Python helper layer for a phone app that controls a DimOS robot dog.

## What is here

- `robot_dog_functions.py`: reusable functions for control, emergency stop, camera frame encoding, and a minimal FastAPI app.
- `__init__.py`: makes `pawify` importable as a package.

## Main entry points

```python
from pawify.robot_dog_functions import (
    make_twist,
    publish_cmd_vel,
    emergency_stop,
    come_find_me,
    create_app,
)
```

### `make_twist(...)`

Builds a bounded DimOS `Twist` message.

Use this when you want to construct a velocity command without publishing it yet.

### `publish_cmd_vel(...)`

Publishes one velocity command to the robot.

Default topic: `"/cmd_vel"`

Example:

```python
publish_cmd_vel(forward=0.2, yaw=0.1)
```

### `emergency_stop(...)`

Publishes repeated zero-velocity commands to stop the robot.

This is the safest function to call from a big red emergency button.

Example:

```python
emergency_stop()
```

### `come_find_me(...)`

Stops the robot first, then sends a natural-language instruction to the running DimOS agent stack.

This currently depends on:

- `dimos` being installed
- an agentic stack already running, for example `dimos run unitree-go2-agentic --daemon`

Example:

```python
come_find_me()
```

### `create_app(...)`

Creates a FastAPI app with:

- `GET /health`
- `GET /camera.mjpeg`
- `POST /cmd_vel`
- `POST /emergency`
- `POST /come-find-me`

Run it with:

```bash
uvicorn pawify.robot_dog_functions:create_app --factory --host 0.0.0.0 --port 8080
```

## How to test

### 1. Syntax check

```bash
python3 -m py_compile pawify/robot_dog_functions.py pawify/__init__.py
```

### 2. Pure function smoke test

```bash
python3 -c "import pawify.robot_dog_functions as f; print(f.clamp(2, -1, 1)); print(f.format_mjpeg_frame(b'abc')[:30])"
```

### 3. DimOS import test

Run this from a shell where the DimOS repo is on `PYTHONPATH`:

```bash
PYTHONPATH=/Users/landyhuang/Documents/dimos python3 -c "from pawify import robot_dog_functions as f; print(f.make_twist(forward=0.2, yaw=0.1))"
```

### 4. Emergency-stop dry run

Only run this against a real or replayed DimOS stack:

```bash
PYTHONPATH=/Users/landyhuang/Documents/dimos python3 -c "from pawify import robot_dog_functions as f; f.emergency_stop(); print('stopped')"
```

## Notes for the next person

- `come_find_me()` is intentionally thin. It assumes the existing DimOS agent stack already knows how to follow/find the user.
- `camera.mjpeg` is a simple browser-friendly stream. If you need lower latency or better compression, replace it with WebRTC later.
- `robot_dog_functions.py` is meant to be a shared utility module, not a full app architecture.
- `.DS_Store` can be ignored.

