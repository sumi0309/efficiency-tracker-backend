# run_analytics.py
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from construction_analytics import generate_dashboard_data, predict_efficiency_with_ml
from video_analysis import update_dashboard_with_video_analysis

def main():
    print("Starting construction site analytics...")
    
    try:
        # Load data files
        print("\nLoading CSV files...")
        dfs = {
            'construction': pd.read_csv('construction_project_data.csv'),
            'equipment': pd.read_csv('equipment_tracking.csv'),
            'iot': pd.read_csv('iot_sensor_data.csv'),
            'material': pd.read_csv('material_management_data.csv'),
            'mobile': pd.read_csv('mobile_activity_data.csv'),
            'project': pd.read_csv('project_management_data.csv')
        }
        print("✓ Data files loaded successfully")

        # Convert timestamps
        print("\nProcessing timestamps...")
        for df_name, df in dfs.items():
            timestamp_cols = [col for col in df.columns if 'time' in col.lower()]
            for col in timestamp_cols:
                dfs[df_name][col] = pd.to_datetime(dfs[df_name][col])
        print("✓ Timestamps processed")

        # Generate predictions and dashboard data
        print("\nGenerating ML predictions and dashboard data...")
        dashboard_data = generate_dashboard_data(dfs)
        print("✓ Dashboard data generated")

        # Process surveillance videos if available
        video_folder = "Surveillance Camera Video"
        if os.path.exists(video_folder) and os.path.isdir(video_folder):
            print("\nProcessing surveillance videos...")
            try:
                dashboard_data = update_dashboard_with_video_analysis(dashboard_data, video_folder)
                print("✓ Video analysis completed")
            except Exception as video_error:
                print(f"⚠️ Video analysis skipped: {str(video_error)}")
        else:
            print("\n⚠️ Surveillance video folder not found, skipping video analysis")

        # Save output to file
        print("\nSaving results...")
        with open('dashboard_output.json', 'w') as f:
            json.dump(dashboard_data, f, indent=2)
        print("✓ Results saved to 'dashboard_output.json'")

        # Print sample of predictions
        print("\nSample of efficiency predictions:")
        predictions = dashboard_data['efficiency_predictions']
        for i, (time, hist, pred) in enumerate(zip(
            predictions['timestamps'][:5],
            predictions['historical_values'][:5],
            predictions['predicted_values'][:5]
        )):
            print(f"{time}: Historical={hist:.1f}, Predicted={pred:.1f}")
        print("...")

        # Print video analysis summary if available
        if 'video_inference' in dashboard_data:
            print("\nVideo Analysis Summary:")
            metrics = dashboard_data['video_inference']['overall_metrics']
            print(f"Total Workers Detected: {metrics['total_workers_detected']}")
            print(f"Total Videos Analyzed: {metrics['total_videos_analyzed']}")
            print(f"Average Productivity Rate: {metrics['average_productivity_rate']:.1f}%")
            print(f"Total Analysis Duration: {metrics['total_analysis_duration']:.1f} seconds")

        print("\nAnalysis completed successfully!")
        return dashboard_data

    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()