# Architecture and design rationale

```text
Laptop webcam
   -> OpenCV capture
   -> MediaPipe 21 landmarks
   -> rule-based recognition / optional Random Forest
   -> UDP localhost:5005
   -> Webots controller
   -> mode arbitration + stale-packet watchdog
   -> proximity safety override
   -> wheel motors
```

## Syllabus mapping
- Locomotion: e-puck differential-drive platform.
- Kinematics: `robot_core/kinematics.py`.
- Perception and uncertainty: proximity readings, noise model to add, EKF covariance.
- Localization: dead reckoning and EKF building block.
- Mapping/planning: occupancy grid, A*, potential-field helper.
- Navigation architectures: gesture layer, arbitration, planner, reactive safety, motion controller.

## Differential-drive equations
Let wheel linear speeds be `v_L` and `v_R`, wheel separation `b`, and heading `theta`.
- `v = (v_R + v_L)/2`
- `omega = (v_R - v_L)/b`
- `x_dot = v cos(theta)`, `y_dot = v sin(theta)`, `theta_dot = omega`

The ideal robot has zero lateral body-frame velocity (non-holonomic constraint).

## Evaluation discipline
Use Supervisor ground truth for scoring only, never as estimator input. Log timestamps, ground-truth pose, odometry pose, EKF pose, command, collision count and sensor readings. Compare algorithms on the same paths and report repeated trials, mean values and failures.
