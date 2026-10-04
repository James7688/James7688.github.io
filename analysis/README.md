# DodgeTracker video-analysis staging

This branch is isolated from the website's main branch.

## Input
Place the fight recording at:

`analysis/input.mp4`

Then push it to the `dodgetracker-analysis` branch. GitHub Actions will automatically run **DodgeTracker video analysis**.

## Output
The workflow uploads an artifact named `dodgetracker-video-analysis` containing:

- whole-fight contact sheets at 1-second intervals
- 0.1-second bottom-left keystroke contact sheets
- high-motion candidate event sheets at 0.15-second intervals
- FFprobe metadata
- `report.json` with timing candidates

The rapid-change detector intentionally does not claim that every visual change is a key press. The generated images are meant for human/AI review.
