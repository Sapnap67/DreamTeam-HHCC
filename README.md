# CrossGuard - DreamTeam HHCC 2026

CrossGuard is a transparent prototype for warning pedestrians about the blind zone of a large truck turning right at a crosswalk.

## Run the demo

Open `index.html` in a modern browser. No installation, model download, or camera is required.

The 14-second replay demonstrates one clear incident:

1. A signal-pole camera identifies a large vehicle.
2. The system recognizes a right-turn trajectory.
3. A conservative blind-zone area is projected.
4. A pedestrian enters that area.
5. Roadside and driver warnings activate.

Use the timeline or **跳至预警** button during the pitch. A team-recorded MP4 can also be loaded locally; the file is not uploaded anywhere.

## Honest prototype boundary

The built-in scenario uses simulated detection coordinates. It demonstrates the product interaction and risk-decision flow, not a production computer-vision system. A real pilot would require an object-detection/tracking model, per-intersection calibration, weather and night testing, and validation with traffic authorities.

## Team review checklist

- Every presenter can explain the four decision steps.
- Replace or adjust copy based on the team's final hardware design.
- Test the full-screen page on the presentation laptop.
- Do not describe replay mode as a live camera feed.
- Record any team modifications in `PROJECT_LOG.md`.

