# Gesture-Guided Intelligent Mobile Robot (Webots R2025a)

This is a modular starter project for a differential-drive robot controlled by webcam hand gestures. The webcam/MediaPipe application and Webots controller are separate processes connected over UDP on `127.0.0.1:5005`.

**Important:** this is a starter implementation, not a claim that it has been validated on your particular Windows/Webots installation. The world uses Webots' standard e-puck PROTO and proximity sensors. The occupancy-grid, A*, EKF and evaluation modules are algorithm building blocks; a full sensor-to-map-to-goal integration still needs local verification and tuning.

## Requirements
- Windows 10/11, 8 GB RAM, no GPU required
- Webots R2025a
- Python 3.10–3.12; Python 3.11 is a reasonable first try if the installed MediaPipe release supports it
- Laptop webcam

## Setup
1. Extract this ZIP.
2. In PowerShell, from the extracted folder:
   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
   If MediaPipe installation fails, check the current wheel's supported Python versions.
3. Open Webots R2025a and open `webots/worlds/gesture_arena.wbt`.
4. On first load Webots may download/cache PROTO assets. Wait. If it stalls, check the internet connection, restart Webots, and reopen the world. If the e-puck PROTO path fails, locate the installed e-puck sample world and adjust the `EXTERNPROTO` URL.
5. Press Play. The controller listens on UDP port 5005.
6. Start the gesture app in another PowerShell window, after Webots has started:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   python gesture_app/gesture_app.py
   ```
   If the venv is already active, just run the Python command.

## Python interpreter
Webots Preferences lets you configure the Python command used by controllers. The robot controller only uses the standard library and Webots API; MediaPipe runs separately, so the two processes do not need the same Python environment. For VS Code debugging, Webots supports external-controller workflows; configure the controller as external using the Webots R2025a documentation and start Webots before launching the controller process.

## Controls and gesture baseline
- `W/A/S/D`: forward / turn left / backward / turn right
- `Space`: stop
- `M`: cycle TELEOP → ASSISTED → GOAL
- `Q`: quit
- Gesture rules are intentionally conservative: open palm and fist stop; index finger extended commands forward. Rules can be sensitive to hand orientation and lighting.

UDP JSON example:
```json
{"command":"FORWARD","speed":0.4,"steering":0.0,"mode":"TELEOP","timestamp":123.4}
```
A 0.5 second watchdog stops the robot if packets become stale.

## Tests
```powershell
python -m pytest
```

## Modules
- `robot_core/kinematics.py`: differential-drive equations
- `robot_core/odometry.py`: wheel encoder dead reckoning
- `robot_core/ekf.py`: EKF prediction and generic linearized measurement update
- `robot_core/occupancy_grid.py`: log-odds grid updated from range rays
- `robot_core/planning.py`: A* on an occupancy probability grid
- `robot_core/avoidance.py`: potential-field steering suggestion
- `gesture_app/collect_gestures.py`, `train_classifier.py`: collect landmark rows and train a Random Forest
- `evaluation/`: pose-error plots and path trial summaries

## Limitations to state honestly
1. The included e-puck world uses proximity sensors, not a planar LiDAR. Occupancy-grid code accepts range rays; connect it to a selected range finder or ray-casting adapter before claiming live mapping.
2. EKF requires a meaningful measurement model, such as known landmark observations. The generic update method alone is not complete sensor fusion.
3. A* is implemented for grids, but goal selection and closed-loop path following need integration/tuning in the simulator.
4. Proximity sensor values and orientation must be checked against the actual Webots e-puck model.
5. Do not report results until experiments are run. Ground truth from Supervisor should be used only for evaluation, never as estimator input.
