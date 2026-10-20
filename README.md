# Gesture-Guided Intelligent Mobile Robot (Webots R2025a)

A simulated e-puck that you drive with webcam hand gestures. The gesture app (OpenCV + MediaPipe) and the Webots controller are separate processes that talk over UDP on `127.0.0.1:5005`. The controller fuses wheel odometry with LiDAR landmark observations in an EKF, builds an occupancy grid from the LiDAR, and can drive itself to a goal with A* or with a potential field.

## Features
| Area | Implementation |
| --- | --- |
| Gestures | MediaPipe hand landmarks → rule-based classifier, or a Random Forest trained on your own samples |
| Modes | **TELEOP** (direct, no safety), **ASSISTED** (potential-field obstacle avoidance plus a stop), **GOAL** (autonomous navigation) |
| Localization | Wheel odometry → EKF prediction. Four known landmark posts detected in the LiDAR scan → range-bearing EKF updates with χ² gating |
| Mapping | Log-odds occupancy grid from a 360° LiDAR, updated at the EKF pose |
| Planning | A* on the live map with obstacle inflation, periodic replanning and a carrot path follower. A potential-field planner is the reactive baseline |
| Safety | 0.5 s command watchdog, LiDAR front-clearance stop, IR last-resort stop |
| Evaluation | Supervisor ground truth (logged for scoring only) → pose-error plots, map plots, A* vs. potential-field trials, gesture confusion matrix |

## Requirements
- Windows 10/11, 8 GB RAM, no GPU required
- Webots R2025a
- Python 3.10–3.12 (MediaPipe 0.10.14 has no wheels for newer versions)
- A webcam, for the gesture app only. Scripted trials need none.

## Setup
From the repository folder in PowerShell:
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
The Webots controller imports `numpy`, `PyYAML` and this repository's `robot_core` package. Webots runs it with the interpreter set in **Tools → Preferences → General → Python command** (default `python`). Set that to the full path of `.venv\Scripts\python.exe`, or make sure the interpreter it uses has `requirements.txt` installed. If it doesn't, the Webots console says so when the controller starts.

## Running
1. Open `webots/worlds/gesture_arena.wbt` in Webots R2025a. On first load Webots downloads the e-puck and background PROTOs; wait for it to finish.
2. Press Play. The controller prints `Robot controller listening at 127.0.0.1:5005`.
3. In another terminal with the venv active:
   ```powershell
   python gesture_app/gesture_app.py              # --classifier rules|rf|auto, --camera N
   ```
4. Use **Reset Simulation** (Ctrl+Shift+T) between experiments. The EKF assumes the robot starts at `robot.start_pose` in `config.yaml`, so don't drag the robot and then save the world.

### Controls
| Input | TELEOP / ASSISTED | GOAL |
| --- | --- | --- |
| Index finger pointing up | forward | go / resume |
| Index finger pointing down | backward | — |
| Index finger pointing left / right | turn on the spot | previous / next goal |
| Open palm or fist | stop | pause |
| No hand visible | stop | keep going |
| `W` `S` `A` `D` (each press holds for 0.6 s) | drive | `W` go, `A`/`D` previous/next goal |
| `Space` | stop | pause |
| `1`–`5` | — | select goal HOME, A, B, C, D |
| `M` / `P` / `Q` | cycle mode / switch planner (A* ↔ potential field) / quit | |

A gesture is used once it shows up in 4 of the last 7 frames, so a few misread frames neither start nor stop a command. The image is mirrored, so "left" is your left. The window shows the robot's navigation state, EKF pose and number of landmarks matched. Goals and every other tunable value live in `config.yaml`.

UDP packet sent by the app:
```json
{"command": "FORWARD", "speed": 0.4, "steering": 0.0, "mode": "GOAL",
 "goal": [1.3, -0.35], "goal_name": "B", "planner": "astar", "timestamp": 123.4}
```
`command` is one of STOP, NONE (no hand), FORWARD, BACKWARD, LEFT, RIGHT. The controller replies to the sender with a JSON status packet about 5 times per second.

## Evaluation
With the simulation running (fast mode is fine; the watchdog uses wall-clock time):
```powershell
python evaluation/run_trials.py                 # A* and potential field over HOME→B→C→D→HOME, no webcam needed
python evaluation/compare_paths.py --plot       # success rate, path length, time, collisions per planner
python evaluation/analyze_pose_log.py           # odometry vs. EKF error against ground truth (newest run)
python evaluation/plot_map.py                   # occupancy map with landmarks, goals, driven paths and A* plans
```
Every controller run writes `data/runs/<timestamp>/`, which holds:
- `pose_log.csv`: ground truth, odometry and EKF pose, plus EKF σ, landmarks used and commands, every step
- `plans.jsonl`: every A* plan
- `occupancy.npz`: the map
- `events.log`

Each finished GOAL trial is appended to `data/path_trials.csv`. Plots go to `evaluation/plots/`. All of these are gitignored, so rerun to regenerate them.

Gesture classifier:
```powershell
python gesture_app/collect_gestures.py   # 1-5 pick the label, R toggles recording, Q quits
python gesture_app/train_classifier.py   # prints rule-based vs. Random Forest accuracy, saves the model and confusion matrix
```
The gesture app uses `models/gesture_rf.joblib` automatically once it exists (`--classifier rules` forces the rules). Predictions below 0.6 confidence become STOP.

## Tests
```powershell
python -m pytest
```
The tests cover kinematics, odometry, the EKF, landmark detection, mapping, planning, the UDP protocol and the gesture features. They also include closed-loop navigation in a simulated arena and a check that the Webots world matches `config.yaml` (start pose, landmark positions, everything resting on the floor, goals reachable).

## Project layout
```text
config.yaml                     shared settings: network, robot, EKF, mapping, navigation, goals
webots/worlds/gesture_arena.wbt arena, landmark posts, e-puck with a LiDAR in its turret slot
webots/controllers/robot_controller/robot_controller.py   per-step loop: sensors → estimation → behaviour → motors
robot_core/                     Webots-independent algorithms
  kinematics.py odometry.py ekf.py landmarks.py occupancy_grid.py planning.py
  avoidance.py behaviors.py navigation.py commands.py datalog.py config.py
gesture_app/                    gesture_app.py, features.py, collect_gestures.py, train_classifier.py
evaluation/                     run_trials.py, analyze_pose_log.py, plot_map.py, compare_paths.py
tests/                          pytest suite (scan_sim.py is an analytic LiDAR for the tests)
docs/                           architecture.md (design and equations), demo_script.md
```

## Troubleshooting
- **Controller exits with an ImportError**: Webots is using a Python without the requirements. Fix the Python command preference (see Setup).
- **"cannot listen on 127.0.0.1:5005"**: another simulation or controller is still running. Close it, or change `network.port`.
- **The app shows "no status"**: the simulation is paused or not running, or the ports differ.
- **"no landmark matched … the pose estimate may be lost"**: reset the simulation. This happens after dragging the robot, or after driving it into a wall in TELEOP, where the wheels spin in place.
- **Webots stalls on first load**: it is downloading PROTO assets. Check the internet connection, restart Webots and reopen the world.

## Limitations
1. This is simulation only. The LiDAR is an addition to the e-puck model (the real e-puck's IR sensors reach only about 7 cm), and the landmark posts are part of the known map.
2. Landmarks are associated with the posts by prediction from the current estimate, starting from a known pose. Large unmodelled motion (dragging the robot, wheels spinning against a wall) can lose the estimate, and there is no global re-localization ("kidnapped robot").
3. Odometry alone drifts by metres within minutes. That's intended: `wheel_radius_m` is the nominal 20.5 mm while the simulated wheel is 20.0 mm, and turns slip by 10–30%. The EKF's process noise is tuned for this.
4. A* treats unknown cells as free and replans as the map fills in. The potential field can get stuck in local minima; it is the baseline that shows why planning matters.
5. "Collisions" in the trial log count moments when the LiDAR clearance drops below the body radius + 5 mm. They are a proxy, not physics contacts.
6. Gesture rules are sensitive to hand orientation and lighting. The Random Forest is only as good as the samples you record with your own camera.
7. Ground truth from the Supervisor is used only for logging and scoring, never by the estimator or the planner.
