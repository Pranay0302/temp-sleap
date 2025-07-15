#!/usr/bin/env python3
"""
SLEAP AI Prediction Visualization Script

This script provides various visualization options for SLEAP AI predictions:
- Trajectory plots showing animal movement over time
- Confidence score analysis
- Keypoint tracking visualization
- Heatmaps of movement patterns
- Frame-by-frame analysis

Usage:
    python sleap_visualization.py --csv data.csv --predictions predictions.slp --output_dir ./plots

Author: Pranay Andra
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import h5py
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SLEAPVisualizer:
    """Class to handle SLEAP AI prediction visualizations."""
    
    def __init__(self, csv_path: Optional[str] = None, predictions_path: Optional[str] = None):
        """
        Initialize the visualizer with data paths.
        
        Args:
            csv_path: Path to the CSV analysis file
            predictions_path: Path to the SLEAP predictions file
        """
        self.csv_data = None
        self.predictions_data = None
        self.output_dir = Path("./plots")
        self.output_dir.mkdir(exist_ok=True)
        
        if csv_path:
            self.load_csv_data(csv_path)
        if predictions_path:
            self.load_predictions_data(predictions_path)
    
    def load_csv_data(self, csv_path: str):
        """Load CSV analysis data."""
        try:
            self.csv_data = pd.read_csv(csv_path)
            print(f"Loaded CSV data with {len(self.csv_data)} frames")
            print(f"Columns: {list(self.csv_data.columns)}")
        except Exception as e:
            print(f"Error loading CSV data: {e}")
    
    def load_predictions_data(self, predictions_path: str):
        """Load SLEAP predictions data (placeholder for .slp file handling)."""
        # Note: .slp files are binary and require SLEAP library
        # This is a placeholder for demonstration
        print(f"Predictions file detected: {predictions_path}")
        print("Note: .slp file processing requires SLEAP library")
        self.predictions_path = predictions_path
    
    def plot_trajectory(self, keypoint: str = "nose", save: bool = True):
        """Plot trajectory of a specific keypoint over time."""
        if self.csv_data is None:
            print("No CSV data loaded")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Extract coordinates
        x_col = f"{keypoint}.x"
        y_col = f"{keypoint}.y"
        score_col = f"{keypoint}.score"
        
        if x_col not in self.csv_data.columns or y_col not in self.csv_data.columns:
            print(f"Keypoint {keypoint} not found in data")
            return
        
        x_coords = self.csv_data[x_col].dropna()
        y_coords = self.csv_data[y_col].dropna()
        scores = self.csv_data[score_col].dropna() if score_col in self.csv_data.columns else None
        
        # Plot trajectory
        ax1.plot(x_coords, y_coords, 'b-', alpha=0.7, linewidth=1)
        ax1.scatter(x_coords.iloc[0], y_coords.iloc[0], c='green', s=100, label='Start', zorder=5)
        ax1.scatter(x_coords.iloc[-1], y_coords.iloc[-1], c='red', s=100, label='End', zorder=5)
        ax1.set_xlabel('X Coordinate')
        ax1.set_ylabel('Y Coordinate')
        ax1.set_title(f'{keypoint.title()} Trajectory')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot trajectory over time
        frame_indices = self.csv_data['frame_idx'].dropna()
        ax2.plot(frame_indices, x_coords, 'b-', label='X', alpha=0.7)
        ax2.plot(frame_indices, y_coords, 'r-', label='Y', alpha=0.7)
        ax2.set_xlabel('Frame Index')
        ax2.set_ylabel('Coordinate')
        ax2.set_title(f'{keypoint.title()} Position Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / f'trajectory_{keypoint}.png', dpi=300, bbox_inches='tight')
            print(f"Saved trajectory plot: {self.output_dir / f'trajectory_{keypoint}.png'}")
        
        plt.show()
    
    # def plot_confidence_scores(self, save: bool = True):
    #     """Plot confidence scores for all keypoints."""
    #     if self.csv_data is None:
    #         print("No CSV data loaded")
    #         return
        
    #     # Find all score columns
    #     score_columns = [col for col in self.csv_data.columns if col.endswith('.score')]
        
    #     if not score_columns:
    #         print("No confidence scores found in data")
    #         return
        
    #     fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    #     axes = axes.flatten()
        
    #     for i, score_col in enumerate(score_columns[:4]):  # Plot first 4 keypoints
    #         keypoint = score_col.replace('.score', '')
    #         scores = self.csv_data[score_col].dropna()
            
    #         if len(scores) > 0:
    #             axes[i].hist(scores, bins=30, alpha=0.7, edgecolor='black')
    #             axes[i].set_xlabel('Confidence Score')
    #             axes[i].set_ylabel('Frequency')
    #             axes[i].set_title(f'{keypoint.title()} Confidence Distribution')
    #             axes[i].grid(True, alpha=0.3)
                
    #             # Add statistics
    #             mean_score = scores.mean()
    #             axes[i].axvline(mean_score, color='red', linestyle='--', 
    #                            label=f'Mean: {mean_score:.3f}')
    #             axes[i].legend()
        
    #     plt.tight_layout()
        
    #     if save:
    #         plt.savefig(self.output_dir / 'confidence_scores.png', dpi=300, bbox_inches='tight')
    #         print(f"Saved confidence plot: {self.output_dir / 'confidence_scores.png'}")
        
    #     plt.show()
    
    def plot_keypoint_heatmap(self, keypoint: str = "nose", save: bool = True):
        """Create a heatmap of keypoint positions."""
        if self.csv_data is None:
            print("No CSV data loaded")
            return
        
        x_col = f"{keypoint}.x"
        y_col = f"{keypoint}.y"
        
        if x_col not in self.csv_data.columns or y_col not in self.csv_data.columns:
            print(f"Keypoint {keypoint} not found in data")
            return
        
        x_coords = self.csv_data[x_col].dropna()
        y_coords = self.csv_data[y_col].dropna()
        
        plt.figure(figsize=(10, 8))
        
        # Create 2D histogram
        plt.hist2d(x_coords, y_coords, bins=50, cmap='hot')
        plt.colorbar(label='Frequency')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title(f'{keypoint.title()} Position Heatmap')
        plt.grid(True, alpha=0.3)
        
        if save:
            plt.savefig(self.output_dir / f'heatmap_{keypoint}.png', dpi=300, bbox_inches='tight')
            print(f"Saved heatmap: {self.output_dir / f'heatmap_{keypoint}.png'}")
        
        plt.show()
    
    def plot_movement_analysis(self, save: bool = True):
        """Analyze and plot movement patterns."""
        if self.csv_data is None:
            print("No CSV data loaded")
            return
        
        # Calculate movement between frames for nose keypoint
        x_col = "nose.x"
        y_col = "nose.y"
        
        if x_col not in self.csv_data.columns or y_col not in self.csv_data.columns:
            print("Nose keypoint not found for movement analysis")
            return
        
        x_coords = self.csv_data[x_col].dropna()
        y_coords = self.csv_data[y_col].dropna()
        
        # Calculate distances between consecutive frames
        distances = np.sqrt(np.diff(x_coords)**2 + np.diff(y_coords)**2)
        frame_indices = self.csv_data['frame_idx'].dropna()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot movement speed over time
        ax1.plot(frame_indices[1:], distances, 'b-', alpha=0.7)
        ax1.set_xlabel('Frame Index')
        ax1.set_ylabel('Distance (pixels)')
        ax1.set_title('Movement Speed Over Time')
        ax1.grid(True, alpha=0.3)
        
        # Plot cumulative distance
        cumulative_distance = np.cumsum(distances)
        ax2.plot(frame_indices[1:], cumulative_distance, 'r-', alpha=0.7)
        ax2.set_xlabel('Frame Index')
        ax2.set_ylabel('Cumulative Distance (pixels)')
        ax2.set_title('Cumulative Distance Traveled')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'movement_analysis.png', dpi=300, bbox_inches='tight')
            print(f"Saved movement analysis: {self.output_dir / 'movement_analysis.png'}")
        
        plt.show()
    
    def plot_all_keypoints_frame(self, frame_idx: int = 0, save: bool = True):
        """Plot all keypoints for a specific frame."""
        if self.csv_data is None:
            print("No CSV data loaded")
            return
        
        # Get frame data
        frame_data = self.csv_data[self.csv_data['frame_idx'] == frame_idx]
        
        if frame_data.empty:
            print(f"Frame {frame_idx} not found in data")
            return
        
        # Find all keypoints
        keypoints = []
        for col in frame_data.columns:
            if col.endswith('.x'):
                keypoint = col.replace('.x', '')
                if f"{keypoint}.y" in frame_data.columns:
                    keypoints.append(keypoint)
        
        plt.figure(figsize=(12, 10))
        
        # Plot each keypoint
        for keypoint in keypoints:
            x_col = f"{keypoint}.x"
            y_col = f"{keypoint}.y"
            score_col = f"{keypoint}.score"
            
            x = frame_data[x_col].iloc[0]
            y = frame_data[y_col].iloc[0]
            score = frame_data[score_col].iloc[0] if score_col in frame_data.columns else 1.0
            
            # Color based on confidence
            color = plt.cm.viridis(score if not pd.isna(score) else 0.5)
            
            plt.scatter(x, y, c=[color], s=100, label=keypoint, alpha=0.8)
            plt.annotate(keypoint, (x, y), xytext=(5, 5), textcoords='offset points')
        
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title(f'All Keypoints - Frame {frame_idx}')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.gca().invert_yaxis()  # Invert Y axis for image coordinates
        
        if save:
            plt.savefig(self.output_dir / f'keypoints_frame_{frame_idx}.png', 
                       dpi=300, bbox_inches='tight')
            print(f"Saved keypoints plot: {self.output_dir / f'keypoints_frame_{frame_idx}.png'}")
        
        plt.show()
    
    def generate_summary_report(self):
        """Generate a summary report of the data."""
        if self.csv_data is None:
            print("No CSV data loaded")
            return
        
        print("\n" + "="*50)
        print("SLEAP AI PREDICTION SUMMARY REPORT")
        print("="*50)
        
        print(f"Total frames analyzed: {len(self.csv_data)}")
        print(f"Frame range: {self.csv_data['frame_idx'].min()} - {self.csv_data['frame_idx'].max()}")
        
        # Keypoint statistics
        keypoints = []
        for col in self.csv_data.columns:
            if col.endswith('.x'):
                keypoint = col.replace('.x', '')
                if f"{keypoint}.y" in self.csv_data.columns:
                    keypoints.append(keypoint)
        
        print(f"\nKeypoints detected: {len(keypoints)}")
        for keypoint in keypoints:
            x_col = f"{keypoint}.x"
            y_col = f"{keypoint}.y"
            score_col = f"{keypoint}.score"
            
            x_data = self.csv_data[x_col].dropna()
            y_data = self.csv_data[y_col].dropna()
            
            if len(x_data) > 0:
                print(f"  {keypoint}: {len(x_data)} detections")
                if score_col in self.csv_data.columns:
                    scores = self.csv_data[score_col].dropna()
                    if len(scores) > 0:
                        print(f"    Average confidence: {scores.mean():.3f}")
        
        # Movement analysis
        if "nose.x" in self.csv_data.columns and "nose.y" in self.csv_data.columns:
            x_coords = self.csv_data["nose.x"].dropna()
            y_coords = self.csv_data["nose.y"].dropna()
            
            if len(x_coords) > 1:
                distances = np.sqrt(np.diff(x_coords)**2 + np.diff(y_coords)**2)
                total_distance = np.sum(distances)
                avg_speed = np.mean(distances)
                
                print(f"\nMovement Analysis (nose keypoint):")
                print(f"  Total distance traveled: {total_distance:.2f} pixels")
                print(f"  Average movement per frame: {avg_speed:.2f} pixels")
                print(f"  Maximum movement in one frame: {np.max(distances):.2f} pixels")
        
        print("\n" + "="*50)
    
    def create_all_visualizations(self):
        """Create all available visualizations."""
        print("Creating all visualizations...")
        
        # Generate summary report
        self.generate_summary_report()
        
        # Create individual plots
        keypoints = ["nose", "neck", "l_ear", "r_ear", "t_base","t_mid","t_end"]
        
        for keypoint in keypoints:
            try:
                self.plot_trajectory(keypoint)
                self.plot_keypoint_heatmap(keypoint)
            except Exception as e:
                print(f"Error creating plots for {keypoint}: {e}")
        
        # Create general analysis plots
        try:
            # self.plot_confidence_scores()
            self.plot_movement_analysis()
            
            # Plot keypoints for first frame
            first_frame = self.csv_data['frame_idx'].iloc[0] if self.csv_data is not None else 0
            self.plot_all_keypoints_frame(first_frame)
        except Exception as e:
            print(f"Error creating analysis plots: {e}")
        
        print(f"\nAll visualizations saved to: {self.output_dir}")


def main():
    """Main function to run the visualizer."""
    parser = argparse.ArgumentParser(description='SLEAP AI Prediction Visualizer')
    parser.add_argument('--csv', type=str, help='Path to CSV analysis file')
    parser.add_argument('--predictions', type=str, help='Path to SLEAP predictions file')
    parser.add_argument('--output_dir', type=str, default='./plots', help='Output directory for plots')
    parser.add_argument('--keypoint', type=str, default='nose', help='Keypoint to analyze')
    parser.add_argument('--frame', type=int, default=0, help='Frame index to visualize')
    parser.add_argument('--all', action='store_true', help='Create all visualizations')
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = SLEAPVisualizer(args.csv, args.predictions)
    visualizer.output_dir = Path(args.output_dir)
    visualizer.output_dir.mkdir(exist_ok=True)
    
    if args.all:
        visualizer.create_all_visualizations()
    else:
        # Create specific visualizations
        if args.csv:
            visualizer.plot_trajectory(args.keypoint)
            # visualizer.plot_confidence_scores()
            visualizer.plot_keypoint_heatmap(args.keypoint)
            visualizer.plot_movement_analysis()
            visualizer.plot_all_keypoints_frame(args.frame)
            visualizer.generate_summary_report()


if __name__ == "__main__":
    main() 
