# SLEAP AI Prediction Visualizer

A comprehensive Python script for visualizing SLEAP AI predictions and analysis data. This tool provides various visualization options for animal tracking data including trajectories, confidence scores, movement patterns, and keypoint analysis.

## Features

- **Trajectory Visualization**: Plot movement paths of specific keypoints over time
- **Confidence Score Analysis**: Analyze prediction confidence distributions
- **Position Heatmaps**: Create 2D heatmaps showing where keypoints are most frequently detected
- **Movement Analysis**: Track speed and cumulative distance traveled
- **Frame-by-Frame Analysis**: Visualize all keypoints for specific frames
- **Summary Reports**: Generate comprehensive data analysis reports

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

The script can be used from the command line with various options:

```bash
# Create all visualizations
python sleap_visualization.py --csv "labels.v001.000_Video 1 test 061924.analysis.csv" --all

# Create specific visualizations
python sleap_visualization.py --csv "labels.v001.000_Video 1 test 061924.analysis.csv" --keypoint nose

# Specify output directory
python sleap_visualization.py --csv "labels.v001.000_Video 1 test 061924.analysis.csv" --output_dir ./my_plots --all
```

### Command Line Options

- `--csv`: Path to the CSV analysis file
- `--predictions`: Path to SLEAP predictions file (optional)
- `--output_dir`: Output directory for plots (default: ./plots)
- `--keypoint`: Specific keypoint to analyze (default: nose)
- `--frame`: Frame index to visualize (default: 0)
- `--all`: Create all available visualizations

### Python API

You can also use the visualizer programmatically:

```python
from sleap_visualization import SLEAPVisualizer
from pathlib import Path

# Initialize visualizer
visualizer = SLEAPVisualizer(
    csv_path="labels.v001.000_Video 1 test 061924.analysis.csv",
    predictions_path="predictions/labels.v001.slp.250622_112731.predictions.slp"
)

# Set output directory
visualizer.output_dir = Path("./my_plots")

# Generate summary report
visualizer.generate_summary_report()

# Create specific visualizations
visualizer.plot_trajectory("nose")
visualizer.plot_confidence_scores()
visualizer.plot_keypoint_heatmap("nose")
visualizer.plot_movement_analysis()
visualizer.plot_all_keypoints_frame(0)

# Or create all visualizations at once
visualizer.create_all_visualizations()
```

### Example Usage

Run the example script to see all visualizations in action:

```bash
python example_usage.py
```

## Data Format

The script expects CSV files with the following column format:
- `frame_idx`: Frame index
- `{keypoint}.x`: X coordinate for each keypoint
- `{keypoint}.y`: Y coordinate for each keypoint  
- `{keypoint}.score`: Confidence score for each keypoint (optional)

Example keypoints: nose, neck, l_ear, r_ear, l_frpaw, r_frpaw, l_bcpaw, r_bcpaw, t_base, t_mid, t_end

## Output

The script generates various visualization files:

- `trajectory_{keypoint}.png`: Movement trajectory plots
- `confidence_scores.png`: Confidence score distributions
- `heatmap_{keypoint}.png`: Position heatmaps
- `movement_analysis.png`: Movement speed and distance analysis
- `keypoints_frame_{frame}.png`: All keypoints for specific frames

## Supported Keypoints

Based on your data, the following keypoints are supported:
- nose
- neck
- l_ear (left ear)
- r_ear (right ear)
- l_frpaw (left front paw)
- r_frpaw (right front paw)
- l_bcpaw (left back paw)
- r_bcpaw (right back paw)
- t_base (tail base)
- t_mid (tail middle)
- t_end (tail end)

## Requirements

- Python 3.7+
- pandas >= 1.5.0
- numpy >= 1.21.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- h5py >= 3.7.0

## Notes

- The script currently focuses on CSV analysis files. SLEAP .slp files require the SLEAP library for full functionality.
- All plots are saved as high-resolution PNG files (300 DPI).
- The script automatically handles missing data and provides informative error messages.
- Y-axis is inverted for image coordinates to match video frame orientation.


## EOF Sleap issue

- Made a issue document for the same, addressing the EOF corrupted frames problem. 
- The script (rewrite_to_mjpg_avi.py) converts the mkv file format into mjpg format frame-by-frame, keeping constant frame rate with a uniform video.
