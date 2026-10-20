# Demo video script (3–4 minutes)

1. **Arena (15 s).** Show the Webots world: a 4 × 4 m arena, three boxes, four yellow landmark posts, and the e-puck with a LiDAR in its turret. Say that it is a simulation.
2. **Pipeline (20 s).** Show the gesture window with the landmark overlay, and explain the path: MediaPipe landmarks → gesture → UDP → controller. Point out the status line, which shows the robot's state, its EKF pose and the landmarks matched.
3. **TELEOP (30 s).** Drive with pointing gestures: forward, left, right and stop (open palm). Show the keyboard fallback and that the robot stops within 0.5 s when the gesture app is closed (the watchdog).
4. **ASSISTED (30 s).** Press `M`, point forward towards a box, and show the robot slowing, steering away and stopping before contact. Compare this with TELEOP, which drives straight into it.
5. **GOAL with A* (45 s).** Press `M` again and select goal C (behind the green box) with `4`, or by pointing right. Point up to go. Show the robot replanning as the map fills in, then show a palm to pause and point up to resume.
6. **Potential-field baseline (20 s).** Press `P`, select goal B, and show the robot getting stuck against the blue box (a local minimum).
7. **Results (40 s).** Show the plots generated after `evaluation/run_trials.py`:
   - `pose_error.png`: odometry drift vs. the EKF, with the 2σ bound
   - `map.png`: the occupancy map with the driven paths and A* plans
   - `path_comparison.png`: A* vs. potential field (success rate, length, time)
   - `gesture_confusion.png`: rules vs. Random Forest accuracy on your own samples
8. **Wrap-up (20 s).** State what works, the measured numbers, and the limitations: it is simulation only, the LiDAR is an addition to the e-puck, the start pose must be known, and there is no global re-localization.

Present only results you have generated from your own runs.
