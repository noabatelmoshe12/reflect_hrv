"""
ECG Batch Processor - Summary Table Generator with Graphs
==========================================================
Processes ECG JSON files from folder structure:
    C:\Data Org\{Participant}\{Session}\{files}.json

Output saved to same structure:
    C:\Data Org\{Participant}\{Session}\hrv_results\

Features:
    - Per-participant timeline graphs
    - Combined all-participants graph
    - Runtime logging

Usage:
    1. Run the script
    2. Results saved per session + summary CSVs + graphs
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
import pandas as pd
import traceback
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# Add parent directory to path to import ECGProcessor
sys.path.insert(0, str(Path(__file__).parent))

# Import the ECGProcessor class
from ecg_hrv_processor_v2 import ECGProcessor

# ============================================================
# CONFIGURATION
# ============================================================
DATA_DIR = r"C:\Data Org"
OUTPUT_DIR = r"C:\Data Org\hrv_batch_results"
# ============================================================


def find_ecg_files(data_dir: str) -> list:
    """
    Find all ECG JSON files following the structure:
    data_dir / participant_id / session_id / *.json
    
    Returns list of tuples: (participant_id, session_id, filepath)
    """
    ecg_files = []
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"ERROR: Directory not found: {data_dir}")
        return []
    
    # Iterate through participant folders
    for participant_folder in sorted(data_path.iterdir()):
        if not participant_folder.is_dir():
            continue
        
        # Skip output folder
        if "hrv" in participant_folder.name.lower() or "result" in participant_folder.name.lower():
            continue
            
        participant_id = participant_folder.name
        
        # Iterate through session folders
        for session_folder in sorted(participant_folder.iterdir()):
            if not session_folder.is_dir():
                continue
                
            session_id = session_folder.name
            
            # Find JSON files in session folder
            for json_file in sorted(session_folder.glob("*.json")):
                # Skip result files
                if "hrv_result" in json_file.name.lower():
                    continue
                if "processing" in json_file.name.lower():
                    continue
                    
                ecg_files.append((participant_id, session_id, json_file))
    
    return ecg_files


def extract_timestamp_from_file(filepath: Path) -> datetime:
    """
    Extract timestamp from JSON file's Metadata field.
    
    Expected format:
    "Metadata": {
        "Year": 25,
        "Month": 4,
        "Day": 24,
        "Hour": 20,
        "Minute": 35,
        "Second": 6,
        ...
    }
    
    Returns None if metadata not found or invalid.
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            
            if "Metadata" in data:
                meta = data["Metadata"]
                
                # Extract date components
                year = meta.get("Year")
                month = meta.get("Month")
                day = meta.get("Day")
                hour = meta.get("Hour", 0)
                minute = meta.get("Minute", 0)
                second = meta.get("Second", 0)
                
                if year is not None and month is not None and day is not None:
                    # Handle 2-digit year (25 -> 2025)
                    if year < 100:
                        year = 2000 + year
                    
                    return datetime(year, month, day, hour, minute, second)
    except Exception as e:
        print(f"    Warning: Could not extract timestamp from {filepath.name}: {e}")
    
    return None


def save_session_results(participant_id: str, session_id: str, filepath: Path, 
                         status: str, hrv_metrics: dict, fail_reason: str, 
                         file_timestamp: datetime, output_dir: str):
    """
    Save results in the same folder structure as input files.
    Structure: output_dir / participant_id / session_id / {filename}_hrv_result.json
    """
    # Create output folder: output_dir / participant_id / session_id
    session_output_dir = Path(output_dir) / participant_id / session_id
    session_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create result filename based on original file
    result_filename = filepath.stem + "_hrv_result.json"
    result_path = session_output_dir / result_filename
    
    # Prepare result data
    result_data = {
        "source_file": filepath.name,
        "source_path": str(filepath),
        "participant_id": participant_id,
        "session_id": session_id,
        "file_timestamp": file_timestamp.isoformat() if file_timestamp else None,
        "processing_timestamp": datetime.now().isoformat(),
        "status": status,
        "fail_reason": fail_reason if status == "FAIL" else None,
        "hrv_metrics": hrv_metrics if status == "PASS" else None
    }
    
    # Save result
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    
    return result_path


def process_all_files(data_dir: str, output_dir: str) -> tuple:
    """
    Process all ECG files and track pass/fail status.
    
    Returns:
        tuple: (participant_summary_df, file_details_df)
    """
    # Initialize processor
    processor = ECGProcessor(base_output_dir=output_dir)
    
    # Find all ECG files
    ecg_files = find_ecg_files(data_dir)
    
    if not ecg_files:
        print(f"No ECG JSON files found in: {data_dir}")
        return None, None
    
    print(f"\nFound {len(ecg_files)} ECG files to process\n")
    print("=" * 80)
    
    # Track results
    results = []
    participant_stats = defaultdict(lambda: {"total": 0, "passed": 0, "failed": 0, "sessions": set()})
    
    # Process each file
    for i, (participant_id, session_id, filepath) in enumerate(ecg_files, 1):
        filename = filepath.name
        
        print(f"[{i}/{len(ecg_files)}] {participant_id}/{session_id}/{filename[:40]}...")
        
        # Track this file for participant
        participant_stats[participant_id]["total"] += 1
        participant_stats[participant_id]["sessions"].add(session_id)
        
        # Extract timestamp from file metadata
        file_timestamp = extract_timestamp_from_file(filepath)
        if file_timestamp is None:
            print(f"    Warning: No valid timestamp in metadata, file will be excluded from time graphs")
        
        # Initialize variables
        hrv_metrics = {}
        fail_reason = ""
        status = "FAIL"
        
        # Try to process
        try:
            hrv_metrics = processor.process_session(str(filepath))
            
            if hrv_metrics and len(hrv_metrics) > 0:
                # SUCCESS
                status = "PASS"
                participant_stats[participant_id]["passed"] += 1
                fail_reason = ""
                rmssd = hrv_metrics.get('HRV_RMSSD', 'N/A')
                hr = hrv_metrics.get('mean_hr_bpm', 'N/A')
                print(f"    ✓ PASS | HR={hr:.1f} BPM, RMSSD={rmssd:.1f} ms" if isinstance(rmssd, (int, float)) else f"    ✓ PASS")
            else:
                # FAILED - returned empty dict
                status = "FAIL"
                participant_stats[participant_id]["failed"] += 1
                fail_reason = "Poor signal quality or insufficient R-peaks"
                print(f"    ✗ FAIL | {fail_reason}")
                
        except FileNotFoundError as e:
            status = "FAIL"
            participant_stats[participant_id]["failed"] += 1
            fail_reason = f"File not found"
            print(f"    ✗ FAIL | {fail_reason}")
            
        except json.JSONDecodeError as e:
            status = "FAIL"
            participant_stats[participant_id]["failed"] += 1
            fail_reason = f"Invalid JSON format"
            print(f"    ✗ FAIL | {fail_reason}")
            
        except ValueError as e:
            status = "FAIL"
            participant_stats[participant_id]["failed"] += 1
            fail_reason = f"Data validation: {str(e)[:50]}"
            print(f"    ✗ FAIL | {fail_reason}")
            
        except Exception as e:
            status = "FAIL"
            participant_stats[participant_id]["failed"] += 1
            fail_reason = f"{type(e).__name__}: {str(e)[:50]}"
            print(f"    ✗ FAIL | {fail_reason}")
        
        # Save results in output folder with same structure
        result_path = save_session_results(
            participant_id, session_id, filepath, 
            status, hrv_metrics, fail_reason, file_timestamp, OUTPUT_DIR
        )
        
        # Record file result
        results.append({
            "Participant": participant_id,
            "Session": session_id,
            "Filename": filename,
            "Full_Path": str(filepath),
            "Result_Path": str(result_path),
            "Status": status,
            "Fail_Reason": fail_reason,
            "Timestamp": file_timestamp,
            "HRV_RMSSD": hrv_metrics.get('HRV_RMSSD') if status == "PASS" else None,
            "Mean_HR": hrv_metrics.get('mean_hr_bpm') if status == "PASS" else None
        })
    
    print("\n" + "=" * 80)
    print("Processing complete!\n")
    
    # Create DataFrames
    file_details_df = pd.DataFrame(results)
    
    # Create participant summary
    participant_summary = []
    for participant_id in sorted(participant_stats.keys()):
        stats = participant_stats[participant_id]
        participant_summary.append({
            "Participant": participant_id,
            "Sessions": len(stats["sessions"]),
            "Original_Files": stats["total"],
            "Passed": stats["passed"],
            "Failed": stats["failed"],
            "Pass_Rate": f"{stats['passed']/stats['total']*100:.1f}%" if stats["total"] > 0 else "N/A"
        })
    
    participant_summary_df = pd.DataFrame(participant_summary)
    
    # Add totals row
    if len(participant_summary_df) > 0:
        totals = {
            "Participant": "TOTAL",
            "Sessions": participant_summary_df["Sessions"].sum(),
            "Original_Files": participant_summary_df["Original_Files"].sum(),
            "Passed": participant_summary_df["Passed"].sum(),
            "Failed": participant_summary_df["Failed"].sum(),
        }
        totals["Pass_Rate"] = f"{totals['Passed']/totals['Original_Files']*100:.1f}%" if totals["Original_Files"] > 0 else "N/A"
        
        participant_summary_df = pd.concat([
            participant_summary_df, 
            pd.DataFrame([totals])
        ], ignore_index=True)
    
    return participant_summary_df, file_details_df


def create_participant_timeline_graph(details_df: pd.DataFrame, participant_id: str, output_dir: str):
    """
    Create timeline graph for a single participant showing passed files over time.
    Only includes files with valid timestamps from metadata.
    """
    # Filter for this participant's passed files with valid timestamps
    participant_data = details_df[
        (details_df["Participant"] == participant_id) & 
        (details_df["Status"] == "PASS") &
        (details_df["Timestamp"].notna())
    ].copy()
    
    if len(participant_data) == 0:
        print(f"    No passed files with valid timestamps for {participant_id}, skipping graph")
        return None
    
    # Sort by timestamp
    participant_data = participant_data.sort_values("Timestamp")
    
    # Calculate cumulative count
    participant_data["Cumulative_Count"] = range(1, len(participant_data) + 1)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot cumulative passed files
    ax.plot(participant_data["Timestamp"], participant_data["Cumulative_Count"], 
            marker='o', linewidth=2, markersize=6, color='#2E86AB')
    ax.fill_between(participant_data["Timestamp"], participant_data["Cumulative_Count"], 
                    alpha=0.3, color='#2E86AB')
    
    # Formatting
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Cumulative Passed Files", fontsize=12)
    ax.set_title(f"Participant {participant_id} - Passed ECG Files Over Time", fontsize=14, fontweight='bold')
    
    # Format x-axis based on date range
    date_range = (participant_data["Timestamp"].max() - participant_data["Timestamp"].min()).days
    if date_range <= 7:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator())
    elif date_range <= 30:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    
    plt.xticks(rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    # Add stats annotation
    stats_text = f"Total Passed: {len(participant_data)}"
    ax.annotate(stats_text, xy=(0.02, 0.98), xycoords='axes fraction',
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save to participant's folder in output directory
    participant_output_dir = Path(output_dir) / participant_id
    participant_output_dir.mkdir(parents=True, exist_ok=True)
    
    graph_path = participant_output_dir / f"{participant_id}_timeline.png"
    plt.savefig(graph_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return graph_path


def create_combined_timeline_graph(details_df: pd.DataFrame, output_dir: str):
    """
    Create combined timeline graph showing all participants' passed files.
    X-axis: weeks, Y-axis: cumulative passed files per participant (different colors)
    Only includes files with valid timestamps from metadata.
    """
    # Filter for passed files with valid timestamps only
    passed_df = details_df[
        (details_df["Status"] == "PASS") & 
        (details_df["Timestamp"].notna())
    ].copy()
    
    if len(passed_df) == 0:
        print("No passed files to graph")
        return None
    
    # Get unique participants
    participants = sorted(passed_df["Participant"].unique())
    
    # Create color map
    colors = plt.cm.tab10(np.linspace(0, 1, len(participants)))
    color_map = dict(zip(participants, colors))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot each participant
    for participant_id in participants:
        participant_data = passed_df[passed_df["Participant"] == participant_id].copy()
        participant_data = participant_data.sort_values("Timestamp")
        participant_data["Cumulative_Count"] = range(1, len(participant_data) + 1)
        
        ax.plot(participant_data["Timestamp"], participant_data["Cumulative_Count"],
                marker='o', linewidth=2, markersize=5, 
                color=color_map[participant_id], label=participant_id)
    
    # Formatting
    ax.set_xlabel("Date (Weekly)", fontsize=12)
    ax.set_ylabel("Cumulative Passed Files", fontsize=12)
    ax.set_title("All Participants - Passed ECG Files Over Time", fontsize=14, fontweight='bold')
    
    # Format x-axis for weekly view
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))  # Every Monday
    
    plt.xticks(rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    # Add legend
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0, title="Participants")
    
    # Add total stats
    total_passed = len(passed_df)
    date_range = passed_df["Timestamp"].max() - passed_df["Timestamp"].min()
    weeks = max(1, date_range.days // 7)
    stats_text = f"Total Passed: {total_passed}\nParticipants: {len(participants)}\nTime Span: {weeks} weeks"
    ax.annotate(stats_text, xy=(0.02, 0.98), xycoords='axes fraction',
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save to main output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    graph_path = output_path / f"all_participants_timeline_{timestamp}.png"
    plt.savefig(graph_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return graph_path


def create_weekly_bar_graph(details_df: pd.DataFrame, output_dir: str):
    """
    Create a stacked bar graph showing passed files per week, colored by participant.
    Only includes files with valid timestamps from metadata.
    """
    # Filter for passed files with valid timestamps only
    passed_df = details_df[
        (details_df["Status"] == "PASS") & 
        (details_df["Timestamp"].notna())
    ].copy()
    
    if len(passed_df) == 0:
        print("No passed files to graph")
        return None
    
    # Add week column
    passed_df["Week"] = passed_df["Timestamp"].dt.to_period('W').apply(lambda r: r.start_time)
    
    # Get unique participants and weeks
    participants = sorted(passed_df["Participant"].unique())
    weeks = sorted(passed_df["Week"].unique())
    
    # Create pivot table
    pivot_data = passed_df.groupby(["Week", "Participant"]).size().unstack(fill_value=0)
    
    # Ensure all participants are in columns
    for p in participants:
        if p not in pivot_data.columns:
            pivot_data[p] = 0
    pivot_data = pivot_data[participants]  # Reorder columns
    
    # Create color map
    colors = plt.cm.tab10(np.linspace(0, 1, len(participants)))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create stacked bar chart
    bottom = np.zeros(len(pivot_data))
    for i, participant_id in enumerate(participants):
        ax.bar(range(len(pivot_data)), pivot_data[participant_id], 
               bottom=bottom, label=participant_id, color=colors[i])
        bottom += pivot_data[participant_id].values
    
    # Formatting
    ax.set_xlabel("Week Starting", fontsize=12)
    ax.set_ylabel("Passed Files Count", fontsize=12)
    ax.set_title("Weekly Passed ECG Files by Participant", fontsize=14, fontweight='bold')
    
    # Set x-tick labels
    week_labels = [w.strftime('%Y-%m-%d') for w in pivot_data.index]
    ax.set_xticks(range(len(week_labels)))
    ax.set_xticklabels(week_labels, rotation=45, ha='right')
    
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(bottom=0)
    
    # Add legend
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0, title="Participants")
    
    plt.tight_layout()
    
    # Save to main output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    graph_path = output_path / f"weekly_stacked_bar_{timestamp}.png"
    plt.savefig(graph_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return graph_path


def print_tables(summary_df: pd.DataFrame, details_df: pd.DataFrame):
    """
    Print formatted tables to console.
    """
    print("\n" + "=" * 80)
    print("TABLE 1: PARTICIPANT SUMMARY")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    
    print("\n" + "=" * 80)
    print("TABLE 2: PASSED FILES")
    print("=" * 80)
    passed_files = details_df[details_df["Status"] == "PASS"][["Participant", "Session", "Filename"]]
    if len(passed_files) > 0:
        print(passed_files.to_string(index=False))
    else:
        print("No files passed processing.")
    
    print("\n" + "=" * 80)
    print("TABLE 3: FAILED FILES")
    print("=" * 80)
    failed_files = details_df[details_df["Status"] == "FAIL"][["Participant", "Session", "Filename", "Fail_Reason"]]
    if len(failed_files) > 0:
        # Truncate filename for display
        failed_display = failed_files.copy()
        failed_display["Filename"] = failed_display["Filename"].str[:35] + "..."
        print(failed_display.to_string(index=False))
    else:
        print("No files failed processing.")


def save_tables(summary_df: pd.DataFrame, details_df: pd.DataFrame, output_dir: str):
    """
    Save tables to CSV files.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save summary table
    summary_file = output_path / f"participant_summary_{timestamp}.csv"
    summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
    print(f"\n✓ Saved: {summary_file}")
    
    # Save detailed file list
    details_file = output_path / f"file_details_{timestamp}.csv"
    details_df.to_csv(details_file, index=False, encoding='utf-8-sig')
    print(f"✓ Saved: {details_file}")
    
    # Save passed files list
    passed_file = output_path / f"passed_files_{timestamp}.csv"
    details_df[details_df["Status"] == "PASS"].to_csv(passed_file, index=False, encoding='utf-8-sig')
    print(f"✓ Saved: {passed_file}")
    
    # Save failed files list
    failed_file = output_path / f"failed_files_{timestamp}.csv"
    details_df[details_df["Status"] == "FAIL"].to_csv(failed_file, index=False, encoding='utf-8-sig')
    print(f"✓ Saved: {failed_file}")


def create_all_graphs(details_df: pd.DataFrame, output_dir: str):
    """
    Create all graphs: per-participant timelines and combined graph.
    """
    print("\n" + "=" * 80)
    print("GENERATING GRAPHS")
    print("=" * 80)
    
    # Check how many files have valid timestamps
    valid_timestamps = details_df["Timestamp"].notna().sum()
    total_passed = (details_df["Status"] == "PASS").sum()
    passed_with_timestamps = ((details_df["Status"] == "PASS") & (details_df["Timestamp"].notna())).sum()
    
    print(f"\nFiles with valid metadata timestamps: {valid_timestamps}/{len(details_df)}")
    print(f"Passed files with timestamps: {passed_with_timestamps}/{total_passed}")
    
    if passed_with_timestamps == 0:
        print("\nNo passed files with valid timestamps - skipping all graphs")
        return
    
    # Get unique participants (exclude TOTAL row if present)
    participants = details_df["Participant"].unique()
    
    # Create per-participant timeline graphs
    print("\nCreating per-participant timeline graphs...")
    for participant_id in sorted(participants):
        graph_path = create_participant_timeline_graph(details_df, participant_id, output_dir)
        if graph_path:
            print(f"  ✓ Saved: {graph_path}")
    
    # Create combined timeline graph
    print("\nCreating combined timeline graph...")
    combined_path = create_combined_timeline_graph(details_df, output_dir)
    if combined_path:
        print(f"  ✓ Saved: {combined_path}")
    
    # Create weekly stacked bar graph
    print("\nCreating weekly stacked bar graph...")
    weekly_path = create_weekly_bar_graph(details_df, output_dir)
    if weekly_path:
        print(f"  ✓ Saved: {weekly_path}")


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    """
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes} min {secs:.1f} sec"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours} hr {minutes} min {secs:.1f} sec"


def main():
    """
    Main entry point.
    """
    # Start timer
    start_time = time.time()
    
    print("\n" + "=" * 80)
    print("ECG BATCH PROCESSOR - SUMMARY TABLE GENERATOR")
    print("=" * 80)
    print(f"Start time:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data directory:   {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Expected structure: {{Participant}}/{{Session}}/{{files}}.json")
    
    # Check if data directory exists
    if not Path(DATA_DIR).exists():
        print(f"\n❌ ERROR: Data directory not found: {DATA_DIR}")
        return
    
    # Process all files
    process_start = time.time()
    summary_df, details_df = process_all_files(DATA_DIR, OUTPUT_DIR)
    process_duration = time.time() - process_start
    
    if summary_df is None or len(summary_df) == 0:
        print("No files processed.")
        return
    
    # Print tables to console
    print_tables(summary_df, details_df)
    
    # Save tables to CSV
    save_tables(summary_df, details_df, OUTPUT_DIR)
    
    # Create graphs
    graphs_start = time.time()
    create_all_graphs(details_df, OUTPUT_DIR)
    graphs_duration = time.time() - graphs_start
    
    # Calculate total duration
    total_duration = time.time() - start_time
    
    # Print runtime log
    print("\n" + "=" * 80)
    print("RUNTIME LOG")
    print("=" * 80)
    print(f"Start time:           {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End time:             {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Processing duration:  {format_duration(process_duration)}")
    print(f"Graph generation:     {format_duration(graphs_duration)}")
    print(f"Total runtime:        {format_duration(total_duration)}")
    
    # Files processed stats
    total_files = len(details_df)
    if total_files > 0:
        print(f"Files processed:      {total_files}")
        print(f"Avg time per file:    {format_duration(process_duration / total_files)}")
    
    # Save runtime log to file
    log_path = Path(OUTPUT_DIR) / f"runtime_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(log_path, 'w') as f:
        f.write("ECG BATCH PROCESSOR - RUNTIME LOG\n")
        f.write("=" * 50 + "\n")
        f.write(f"Start time:           {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"End time:             {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Processing duration:  {format_duration(process_duration)}\n")
        f.write(f"Graph generation:     {format_duration(graphs_duration)}\n")
        f.write(f"Total runtime:        {format_duration(total_duration)}\n")
        f.write(f"Files processed:      {total_files}\n")
        if total_files > 0:
            f.write(f"Avg time per file:    {format_duration(process_duration / total_files)}\n")
    print(f"\n✓ Runtime log saved: {log_path}")
    
    print("\n" + "=" * 80)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
