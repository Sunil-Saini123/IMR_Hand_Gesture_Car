# Architecture and design rationale

```text
Laptop webcam ─► OpenCV capture ─► MediaPipe 21 landmarks ─► rules / Random Forest ─► 4-frame debounce
                                                                                            │
                                     UDP 127.0.0.1:5005  ◄── keyboard fallback ◄────────────┘
                                            │                       ▲ status replies (~5 Hz)
Webots controller (one loop per 32 ms step) ▼                       │
  wheel encoders ──► odometry increment ──► EKF predict ──┐
  LiDAR scan ──► landmark post detection ──► EKF update ◄─┘──► pose estimate
  LiDAR scan + pose ──► log-odds occupancy grid
  newest command + watchdog ──► mode arbitration
        TELEOP:   gesture → (v, ω)                       (baseline, no safety)
        ASSISTED: gesture intent + potential field → (v, ω), clearance stop
        GOAL:     A* on the grid → carrot follower, or potential field → (v, ω), clearance stop
  IR last-resort stop ──► inverse kinematics ──► wheel motors
  Supervisor ground truth ──► logs and trial scoring only
```

All algorithms live in `robot_core/` and do not import Webots, so they are unit-tested on their own
(`tests/`). The controller only wires devices to them.

## Syllabus mapping
- **Locomotion and kinematics**: e-puck differential drive; forward and inverse kinematics (`kinematics.py`).
- **Perception and uncertainty**: 360° LiDAR with noise; post detection with width and contrast tests (`landmarks.py`); IR proximity sensors.
- **Localization**: dead reckoning (`odometry.py`) and EKF localization against a known landmark map (`ekf.py`).
- **Mapping**: log-odds occupancy grid (`occupancy_grid.py`).
- **Planning and navigation**: A* with configuration-space inflation (`planning.py`), path following and goal handling (`navigation.py`), potential fields (`avoidance.py`).
- **Navigation architecture**: a gesture layer, mode arbitration, a deliberative planner, a reactive safety layer and the motion controller.

## Kinematics
Wheel linear speeds are `v_L = r·φ̇_L` and `v_R = r·φ̇_R`, the wheel separation is `b`, and the heading is `θ`:
- forward: `v = (v_R + v_L)/2`, `ω = (v_R − v_L)/b`, `ẋ = v cos θ`, `ẏ = v sin θ`, `θ̇ = ω`
- inverse (`wheel_speeds`): `φ̇_L = (v − ωb/2)/r`, `φ̇_R = (v + ωb/2)/r`. If either wheel exceeds the motor limit, both are scaled by the same factor, which keeps the turning radius.

The ideal robot has zero lateral velocity (the non-holonomic constraint), so the controllers steer by heading.

## Odometry
For each encoder increment, `Δs_{L,R} = r·Δφ_{L,R}`, `d = (Δs_R + Δs_L)/2` and `Δθ = (Δs_R − Δs_L)/b`. The update uses the
midpoint heading: `x += d cos(θ + Δθ/2)`, `y += d sin(θ + Δθ/2)`, `θ += Δθ`.

## EKF localization
State `x = [x, y, θ]ᵀ`, with covariance `P` starting at `robot.start_pose`.

**Prediction.** It uses the same midpoint motion model, with Jacobians

```text
F = [[1, 0, −d sin θm], [0, 1, d cos θm], [0, 0, 1]]        θm = θ + Δθ/2
V = [[cos θm, −½ d sin θm], [sin θm, ½ d cos θm], [0, 1]]
P ← F P Fᵀ + V M Vᵀ,   M = diag(a1²|d| + a2²|Δθ|,  a3²|d| + a4²|Δθ|)
```

The noise variance grows linearly with wheel travel (Siegwart's odometry error model in random-walk form),
not with time, so a parked robot doesn't become less certain. The rotation term `a4` is large because the
simulated e-puck slips by 10–30% when it turns (measured against ground truth: 158° of encoder heading
for 141° of real rotation).

**Measurement.** For a post at known `(l_x, l_y)`, with `Δx = l_x − x`, `Δy = l_y − y` and `q = Δx² + Δy²`:

```text
h(x) = [√q,  atan2(Δy, Δx) − θ]
H    = [[−Δx/√q, −Δy/√q, 0], [Δy/q, −Δx/q, −1]]
ν = z − h(x)  (bearing wrapped to ±π),   S = H P Hᵀ + R
accept if νᵀ S⁻¹ ν ≤ 9.21  (χ², 2 DOF, 99%),   K = P Hᵀ S⁻¹,   x ← x + Kν,   P ← (I−KH) P (I−KH)ᵀ + K R Kᵀ
```

**Data association.** It is done by prediction. Around each post's predicted bearing, the closest return near the
predicted range seeds a cluster of rays that hit the same post. The cluster counts only if it is no wider than a post
of that radius would appear at that range, and if the background on both sides is at least 0.1 m farther. This
rejects walls and box faces. The measured range is `r_min + radius` and the bearing is the cluster's mean.

**Safeguards.**
- Corrections are skipped while the robot is parked, because repeated scans from one spot aren't independent and would make `P` overconfident.
- If posts are in view but every match is gated out 10 times, `P` is widened so the filter can re-converge.
- After 10 s of driving with no match, the controller warns that the estimate may be lost.

## Occupancy grid
Each cell stores log-odds `l = log(p/(1−p))`, starting at 0 (p = 0.5, unknown). Along every LiDAR ray (sampled at half a
cell), cells before the return get `l += l_miss` (−0.4), and the return cell gets `l += l_hit` (+0.85). Values are clamped to
±5, and each cell is updated at most once per scan. Rays without a return only clear free space. The grid covers
6 × 6 m at 5 cm per cell, with the world origin at its centre.

## A* planning and path following
- Cells with `p ≥ 0.65` are obstacles. They are inflated by `inflation_m` (body radius plus margin) to give the
  configuration space. Unknown cells are free (optimistic planning).
- A* runs on the 8-connected grid with a Euclidean heuristic, and diagonal moves may not cut obstacle corners. If the robot has
  ended up inside the inflated zone, the planner first takes the shortest route out of it over non-occupied cells.
- The robot replans every second, and right away when the front clearance stops it, so new map information is used.
- The carrot follower aims at the first path point at least 0.12 m ahead, with `α = atan2(Δy, Δx) − θ`.
  If `|α| > 0.8 rad` it turns on the spot. Otherwise `v = v_max cos α` (slower near the goal) and `ω = k·α`.
  Both are clamped (`ω ≤ 1.2 rad/s`, because faster turns slip more).

## Potential field
The attractive unit vector points at the goal (GOAL mode) or along the user's command (ASSISTED mode). Each LiDAR return
at distance `d < ρ0` adds the repulsion `−k (1/d − 1/ρ0)/d² · d̂`. `k` is given per radian of scan, so it doesn't depend on
the LiDAR resolution. The robot follows the resulting heading. It has no memory, so it gets trapped in local minima, for
example behind a box that faces the goal. Stuck detection (less than 5 cm of progress in 15 s) ends such trials.

## Mode arbitration and safety
- **Watchdog**: if no packet arrives for 0.5 s (wall clock), the robot stops and any GOAL navigation is cancelled.
- **TELEOP** maps gestures straight to `(v, ω)`. It is the no-safety baseline.
- **ASSISTED** blends the user's intent with the potential field, scales speed by front clearance, and stops at 7 cm.
- **GOAL**: FORWARD starts or resumes, STOP pauses, and no hand keeps going. Changing the goal while driving switches to it.
- The **IR last-resort stop** applies outside TELEOP: forward motion stops when a front IR sensor reads above `proximity_stop_threshold` (about 3.5 cm).

## Evaluation discipline
- Supervisor ground truth is read only for logging and scoring, never by the estimator or the planner.
- Each run logs, per step: time, mode, command, navigation state, ground-truth, odometry and EKF pose, EKF σ, landmarks used, minimum range and `(v, ω)`.
- Each trial logs: method, goal, start, success and outcome, time, ground-truth path length, collisions (LiDAR clearance < body radius + 5 mm), final ground-truth error, and mean EKF error.
- `evaluation/run_trials.py` drives every planner over the same legs. When a leg fails it first brings the robot to that leg's goal with A*, so the comparison stays paired. Report repeated trials, means and failures.
