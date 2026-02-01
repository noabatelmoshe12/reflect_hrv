"""
ECG Data Processing and HRV Analysis Script
===========================================
This script processes raw ECG data from a JSON file containing ADC values,
converts them to millivolts, and performs comprehensive HRV analysis using NeuroKit2.

Hardware specifications:
- ADC Resolution: 12-bit
- Reference Voltage: 3.6V
- Amplifier Gain: 2
- Sampling Rate: 200 Hz (5ms intervals)

Author: ECG Processing Pipeline
Date: 2025

Required Dependencies:
- json
- logging
- sys
- pathlib
- datetime
- numpy
- pandas
- neurokit2
- matplotlib
"""

import json
import logging
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import neurokit2 as nk
import matplotlib
matplotlib.use("Agg")   # ← אומר למטפל להפיק קבצים בלבד, בלי חלונות GUI
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import math

class ECGProcessor:
    """
    A class to process ECG data from JSON files and extract HRV metrics.
    
    This processor handles:
    1. JSON data loading and validation
    2. ADC to millivolt conversion
    3. ECG signal processing and peak detection
    4. Comprehensive HRV analysis
    """
    
    def __init__(self, adc_bits: int = 12, vref: float = 3.6, gain: float = 2.0, sampling_rate: int = 200, 
                 base_output_dir: str = "hrv_results"):
        """
        Initialize the ECG processor with hardware specifications and output directory.
        
        Parameters:
        -----------
        adc_bits : int
            ADC resolution in bits (default: 12)
        vref : float
            Reference voltage in volts (default: 3.6V)
        gain : float
            Amplifier gain (default: 2.0)
        sampling_rate : int
            Sampling frequency in Hz (default: 200 Hz)
        base_output_dir : str
            Base directory for output files (default: "hrv_results")
        """
        self.adc_bits = adc_bits
        self.vref = vref
        self.gain = gain
        self.sampling_rate = sampling_rate
        self.adc_max = 2**adc_bits - 1
        self.base_output_dir = base_output_dir
        
        # Logging configuration will be set up dynamically in process_session method
        self.logger = None
    
    def _setup_logging(self, log_file_path: Path):
        """
        Set up logging configuration with file and stream handlers.
        
        Parameters:
        -----------
        log_file_path : Path
            Path to the log file
        """
        # Create logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # Create formatters
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s')
        
        # File handler
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(file_formatter)
        
        # Stream handler
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(file_formatter)
        
        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)
    
    def load_json_data(self, filepath: str) -> Dict:
        """
        Load and validate JSON data from file.
        
        Parameters:
        -----------
        filepath : str
            Path to the JSON file
            
        Returns:
        --------
        Dict: Loaded JSON data
        """
        self.logger.info(f"Loading JSON data from: {filepath}")
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Validate required fields
            if 'Data' not in data:
                raise ValueError("JSON file missing 'Data' field")
            
            self.logger.info(f"Successfully loaded JSON with {len(data['Data'])} samples")
            self.logger.info(f"Session metadata: {data.get('Metadata', {})}")
            
            return data
            
        except FileNotFoundError:
            self.logger.error(f"File not found: {filepath}")
            raise
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON format: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading JSON: {e}")
            raise
    
    def extract_ecg_data(self, json_data: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract ECG values and timestamps from JSON data.
        
        Parameters:
        -----------
        json_data : Dict
            Loaded JSON data
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]: ECG values (ADC counts) and timestamps (microseconds)
        """
        self.logger.info("Extracting ECG data from JSON")
        
        data_points = json_data['Data']
        ecg_values = []
        timestamps = []
        
        for i, point in enumerate(data_points):
            try:
                ecg_val = point.get('ECG', 0)
                timestamp = point.get('us since start', i * 5000)  # Default to 5ms intervals
                
                ecg_values.append(ecg_val)
                timestamps.append(timestamp)
                
            except Exception as e:
                self.logger.warning(f"Error processing data point {i}: {e}")
                continue
        
        ecg_array = np.array(ecg_values, dtype=np.float64)
        time_array = np.array(timestamps, dtype=np.float64)
        
        self.logger.info(f"Extracted {len(ecg_array)} ECG samples")
        self.logger.info(f"ECG range: [{ecg_array.min():.0f}, {ecg_array.max():.0f}] ADC counts")
        self.logger.info(f"Time range: {time_array[0]/1e6:.2f}s to {time_array[-1]/1e6:.2f}s")
        
        return ecg_array, time_array
    
    def convert_adc_to_millivolts(self, adc_values: np.ndarray) -> np.ndarray:
        """
        Convert ADC counts to millivolts.
        
        Parameters:
        -----------
        adc_values : np.ndarray
            Array of ADC count values
            
        Returns:
        --------
        np.ndarray: ECG values in millivolts
        """
        self.logger.info("Converting ADC values to millivolts")

        # Artifact removal BEFORE conversion
        invalid_mask = (adc_values < 1) | (adc_values > 5000)
        num_invalid = np.sum(invalid_mask)
        if num_invalid > 0:
            self.logger.warning(f"Detected {num_invalid} invalid ADC samples "
                           f"({num_invalid/len(adc_values)*100:.2f}%). Replacing via interpolation.")
            adc_values = np.where(invalid_mask, np.nan, adc_values)
            # Replace missing values to maintain time continuity
            adc_values = pd.Series(adc_values).interpolate(limit_direction='both').to_numpy()
        
        # Convert ADC counts to millivolts
        ecg_mv = (adc_values / self.adc_max) * self.vref / self.gain * 1000
        
        self.logger.info(f"Conversion complete: ADC range [{adc_values.min():.0f}, {adc_values.max():.0f}]")
        self.logger.info(f"Millivolt range: [{ecg_mv.min():.3f}, {ecg_mv.max():.3f}] mV")
        self.logger.info(f"Mean ECG amplitude: {ecg_mv.mean():.3f} mV")
        
        return ecg_mv
    
    def preprocess_ecg_signal(self, ecg_signal: np.ndarray) -> np.ndarray:
        """
        Preprocess ECG signal for R-peak detection.
        
        Parameters:
        -----------
        ecg_signal : np.ndarray
            Raw ECG signal in millivolts
            
        Returns:
        --------
        np.ndarray: Preprocessed ECG signal
        """
        self.logger.info("Starting ECG signal preprocessing")
        
        try:
            # Remove DC offset
            ecg_centered = ecg_signal - np.mean(ecg_signal)
            self.logger.info(f"Removed DC offset, new mean: {np.mean(ecg_centered):.6f}")
            
            # Bandpass filter and baseline correction
            ecg_cleaned = nk.ecg_clean(ecg_centered, sampling_rate=self.sampling_rate, method="neurokit")
            self.logger.info("Applied NeuroKit2 cleaning (bandpass filter and baseline correction)")
            
            # Detect and handle outliers (clipping)
            outlier_threshold = 5 * np.std(ecg_cleaned)
            outliers = np.abs(ecg_cleaned) > outlier_threshold
            outlier_count = np.sum(outliers)
            if outlier_count > 0:
                self.logger.warning(f"Found {outlier_count} outliers ({outlier_count/len(ecg_cleaned)*100:.2f}%)")
                # Clip outliers instead of removing them
                ecg_cleaned = np.clip(ecg_cleaned, -outlier_threshold, outlier_threshold)
                
            # Interpolation (fill missing/clipped values)
            ecg_cleaned = pd.Series(ecg_cleaned).interpolate(limit_direction='both').to_numpy()
            self.logger.info("Applied interpolation to fill missing/clipped values and maintain time continuity")

            self.logger.info(f"Preprocessed signal range: [{ecg_cleaned.min():.3f}, {ecg_cleaned.max():.3f}] mV")
            return ecg_cleaned
            
        except Exception as e:
            self.logger.error(f"Error during preprocessing: {e}")
            self.logger.warning("Returning original signal")
            return ecg_signal
    
    def check_signal_quality(self, ecg_signal: np.ndarray) -> str:
        """
        Evaluates ECG signal quality after preprocessing.
        
        Parameters:
        -----------
        ecg_signal : np.ndarray
            Preprocessed ECG signal
            
        Returns:
        --------
        str: Signal quality flag ('GOOD' or 'BAD')
        """
        self.logger.info("Checking ECG signal quality")

        # Compute threshold for noise detection
        threshold = 5 * np.std(ecg_signal)
        clipped = np.abs(ecg_signal) > threshold
        clipped_ratio = np.sum(clipped) / len(ecg_signal)

        # Detect long continuous corrupted segment (>10 s = 2000 samples at 200 Hz)
        window = np.ones(2000, dtype=int)
        bad_segments = np.convolve(clipped.astype(int), window, mode='valid')
        has_long_bad = np.any(bad_segments >= 2000)

        # Determine signal quality
        if clipped_ratio > 0.5 or has_long_bad:
            self.logger.warning(f"Poor signal quality detected! "
                        f"Clipped samples: {clipped_ratio*100:.2f}% | "
                        f"Continuous bad segment >10 s: {has_long_bad}")
            quality_flag = "BAD"
        else:
            self.logger.info(f"Signal quality GOOD. Clipped {clipped_ratio*100:.2f}% of samples.")
            quality_flag = "GOOD"

        return quality_flag
    
    def detect_r_peaks(self, ecg_signal: np.ndarray) -> np.ndarray:
        """
        Detect R-peaks in the ECG signal using Robust Logic.
        Includes correct_artifacts=True to fix high RMSSD.
        """
        self.logger.info("Starting R-peak detection (Robust Mode)")
        
        try:
            _, rpeaks = nk.ecg_peaks(ecg_signal, 
                                   sampling_rate=self.sampling_rate, 
                                   correct_artifacts=True)
            
            peak_indices = rpeaks['ECG_R_Peaks']

            # -----------------------------------------------------------------
            # מכאן ומטה - הלוגיקה המקורית שלך לסינון רעשים ידני (מצוין להשאיר אותה)
            # -----------------------------------------------------------------
            if len(peak_indices) > 2:
                rr_intervals = np.diff(peak_indices)
                mean_rr = np.mean(rr_intervals)
                
                # Identify implausible intervals: too short or too long
                min_rr = mean_rr * 0.5   # faster than 120 BPM
                max_rr = mean_rr * 1.8   # slower than 33 BPM
                valid_mask = (rr_intervals > min_rr) & (rr_intervals < max_rr) 
            
                # Keep only peaks that produce valid RR intervals
                valid_peaks = np.insert(peak_indices[1:][valid_mask], 0, peak_indices[0])
                
                if len(valid_peaks) < len(peak_indices):
                    diff = len(peak_indices) - len(valid_peaks)
                    self.logger.warning(f"Removed {diff} suspect R-peaks (Manual Filter).")
                
                peak_indices = valid_peaks

            # Remove edge artifacts (start/end of recording)
            if len(peak_indices) > 2:
                edge_margin = int(2 * self.sampling_rate)  # הקטנתי ל-2 שניות כדי לא לאבד מידע
                valid_peaks = [p for p in peak_indices if edge_margin < p < (len(ecg_signal) - edge_margin)]
                
                if len(valid_peaks) < len(peak_indices):
                    self.logger.warning(f"Removed {len(peak_indices) - len(valid_peaks)} edge artifact peaks.")
                
                peak_indices = np.array(valid_peaks)

            self.logger.info(f"Detected {len(peak_indices)} R-peaks")
            
            # Compute RR intervals for logging
            if len(peak_indices) > 1:
                rr_intervals = np.diff(peak_indices) / self.sampling_rate * 1000  # in ms
                mean_rr = np.mean(rr_intervals)
                mean_hr = 60000 / mean_rr  # BPM
                
                self.logger.info(f"Mean heart rate: {mean_hr:.1f} BPM")
            
            return peak_indices
            
        except Exception as e:
            self.logger.error(f"Error during R-peak detection: {e}")
            raise
    
    def calculate_hrv_metrics(self, peak_indices: np.ndarray) -> Dict:
        """
        Calculate comprehensive HRV metrics from R-peak indices.
        
        Parameters:
        -----------
        peak_indices : np.ndarray
            Indices of detected R-peaks
            
        Returns:
        --------
        Dict: Dictionary containing all HRV metrics
        """
        self.logger.info("Calculating HRV metrics")
        
        if len(peak_indices) < 3:
            self.logger.error("Insufficient R-peaks for HRV analysis (need at least 3)")
            return {}
        
        try:
            # Convert peak indices to RR intervals in milliseconds
            rr_intervals = np.diff(peak_indices) / self.sampling_rate * 1000
            
            # Use NeuroKit2 for comprehensive HRV analysis
            hrv_metrics = nk.hrv(peak_indices, sampling_rate=self.sampling_rate, show=False)
            
            # Log key metrics
            self.logger.info("=== HRV METRICS CALCULATED ===")
            
            # Time domain metrics
            if 'HRV_RMSSD' in hrv_metrics.columns:
                self.logger.info(f"RMSSD: {hrv_metrics['HRV_RMSSD'].values[0]:.2f} ms")
            if 'HRV_SDNN' in hrv_metrics.columns:
                self.logger.info(f"SDNN: {hrv_metrics['HRV_SDNN'].values[0]:.2f} ms")
            if 'HRV_pNN50' in hrv_metrics.columns:
                self.logger.info(f"pNN50: {hrv_metrics['HRV_pNN50'].values[0]:.2f}%")
            
            # Frequency domain metrics
            if 'HRV_LF' in hrv_metrics.columns:
                self.logger.info(f"LF Power: {hrv_metrics['HRV_LF'].values[0]:.2f} ms²")
            if 'HRV_HF' in hrv_metrics.columns:
                self.logger.info(f"HF Power: {hrv_metrics['HRV_HF'].values[0]:.2f} ms²")
            if 'HRV_LFHF' in hrv_metrics.columns:
                self.logger.info(f"LF/HF Ratio: {hrv_metrics['HRV_LFHF'].values[0]:.2f}")
            
            # Convert DataFrame to dictionary for easier access
            hrv_dict = hrv_metrics.to_dict('records')[0] if not hrv_metrics.empty else {}
            
            # Add custom metrics
            hrv_dict['num_peaks'] = len(peak_indices)
            hrv_dict['mean_rr_ms'] = np.mean(rr_intervals)
            hrv_dict['std_rr_ms'] = np.std(rr_intervals)
            hrv_dict['mean_hr_bpm'] = 60000 / np.mean(rr_intervals)
            hrv_dict['recording_duration_s'] = (peak_indices[-1] - peak_indices[0]) / self.sampling_rate
            
            self.logger.info(f"Total metrics calculated: {len(hrv_dict)}")
            
            return hrv_dict
            
        except Exception as e:
            self.logger.error(f"Error calculating HRV metrics: {e}")
            return {}
    
    def visualize_results(self, ecg_signal: np.ndarray, peak_indices: np.ndarray, 
                          save_path: Optional[str] = None):
        """
        Create visualization of ECG signal and detected peaks.
        
        Parameters:
        -----------
        ecg_signal : np.ndarray
            Preprocessed ECG signal
        peak_indices : np.ndarray
            Indices of detected R-peaks
        save_path : Optional[str]
            Path to save the figure (if provided)
        """
        self.logger.info("Creating ECG visualization")
        
        try:
            # Create time axis in seconds
            time_axis = np.arange(len(ecg_signal)) / self.sampling_rate
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 1, figsize=(12, 8))
            
            # Plot 1: Full ECG signal with peaks
            axes[0].plot(time_axis, ecg_signal, 'b-', linewidth=0.5, label='ECG Signal')
            axes[0].plot(time_axis[peak_indices], ecg_signal[peak_indices], 'ro', 
                        markersize=5, label=f'R-peaks (n={len(peak_indices)})')
            axes[0].set_xlabel('Time (s)')
            axes[0].set_ylabel('Amplitude (mV)')
            axes[0].set_title('ECG Signal with Detected R-peaks')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Plot 2: RR intervals
            if len(peak_indices) > 1:
                rr_intervals = np.diff(peak_indices) / self.sampling_rate * 1000
                peak_times = time_axis[peak_indices[1:]]
                axes[1].plot(peak_times, rr_intervals, 'g-', linewidth=1.5)
                axes[1].set_xlabel('Time (s)')
                axes[1].set_ylabel('RR Interval (ms)')
                axes[1].set_title('RR Interval Tachogram')
                axes[1].grid(True, alpha=0.3)
                
                # Add mean line
                mean_rr = np.mean(rr_intervals)
                axes[1].axhline(y=mean_rr, color='r', linestyle='--', 
                              label=f'Mean: {mean_rr:.1f} ms')
                axes[1].legend()
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                self.logger.info(f"Visualization saved to: {save_path}")

            plt.close()

            #plt.show()
            

            
        except Exception as e:
            self.logger.error(f"Error creating visualization: {e}")
    
    def clean_nan_values(self, data: any, replacement=None) -> any:
        """
        Recursively clean NaN values from data structure.
        
        Parameters:
        -----------
        data : any
            Data structure to clean (dict, list, or value)
        replacement : any
            Value to replace NaN with (default: None, which becomes null in JSON)
            
        Returns:
        --------
        Cleaned data structure
        """
        if isinstance(data, dict):
            return {k: self.clean_nan_values(v, replacement) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.clean_nan_values(item, replacement) for item in data]
        elif isinstance(data, float):
            if math.isnan(data) or math.isinf(data):
                return replacement
            return data
        elif isinstance(data, np.floating):
            if np.isnan(data) or np.isinf(data):
                return replacement
            return float(data)
        elif isinstance(data, np.integer):
            return int(data)
        elif isinstance(data, np.ndarray):
            return data.tolist()
        else:
            return data
    
    def save_results(self, hrv_metrics: Dict, output_path: str, nan_replacement=None):
        """
        Save HRV metrics to a JSON file with NaN handling.
        
        Parameters:
        -----------
        hrv_metrics : Dict
            Dictionary containing HRV metrics
        output_path : str
            Path for output JSON file
        nan_replacement : any
            Value to replace NaN with (None for null, 0 for zero, etc.)
        """
        self.logger.info(f"Saving results to: {output_path}")
        
        try:
            # Clean NaN values from metrics
            cleaned_metrics = self.clean_nan_values(hrv_metrics, nan_replacement)
            
            # Count how many NaN values were replaced
            nan_count = sum(1 for k, v in hrv_metrics.items() 
                          if isinstance(v, (float, np.floating)) and 
                          (np.isnan(v) or np.isinf(v)))
            
            if nan_count > 0:
                self.logger.warning(f"Replaced {nan_count} NaN/Inf values with {nan_replacement}")
            
            # Add metadata
            results = {
                'timestamp': datetime.now().isoformat(),
                'processor_settings': {
                    'adc_bits': self.adc_bits,
                    'vref': self.vref,
                    'gain': self.gain,
                    'sampling_rate': self.sampling_rate
                },
                'hrv_metrics': cleaned_metrics,
                'processing_notes': {
                    'nan_values_replaced': nan_count,
                    'nan_replacement_value': nan_replacement
                }
            }
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            self.logger.info("Results successfully saved as valid JSON")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
    
    def process_session(self, json_filepath: str, base_output_dir: str = None) -> Dict:
        """
        Implements all steps (1–8) according to Yonatan's guidelines:
        1. Load JSON data
        2. Extract ECG values
        3. Convert ADC to millivolts (includes artifact removal)
        4. Preprocess ECG (filtering, clipping, interpolation)
        5. Check signal quality
        6. Detect R-peaks
        7. Calculate HRV metrics
        8. Save and visualize results
        
        Complete processing pipeline for a single ECG session.
        
        Parameters:
        -----------
        json_filepath : str
            Path to input JSON file
            
        Returns:
        --------
        Dict: HRV metrics dictionary
        """
        # Create output directory specific to this input file
        input_filename = Path(json_filepath).stem
        output_path = Path(self.base_output_dir) / input_filename
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Setup logging to file in this specific directory
        log_file = output_path / f"{input_filename}_processing.log"
        self._setup_logging(log_file)
        
        self.logger.info("=" * 50)
        self.logger.info("Starting ECG session processing")
        self.logger.info("=" * 50)
        
        try:
            # Step 1: Load JSON data
            json_data = self.load_json_data(json_filepath)
            
            # Step 2: Extract ECG data
            ecg_adc, timestamps = self.extract_ecg_data(json_data)
            
            # Step 3: Convert to millivolts (with artifact removal)
            ecg_mv = self.convert_adc_to_millivolts(ecg_adc)

          
            # --- תחילת בדיקת אבחון (הוספה ידנית) ---
            # חישוב עוצמת האות במיליוולט (מקסימום פחות מינימום)
            # Step 3: Convert to millivolts (with artifact removal)

            # ========================================================
            # 🛑 DIAGNOSTIC CHECK (בדיקת אבחון זמנית)
            # ========================================================
            signal_amp = np.max(ecg_mv) - np.min(ecg_mv)
            
            self.logger.info(f" DIAGNOSTIC CHECK:")
            self.logger.info(f"   Signal Amplitude: {signal_amp:.2f} mV")
            
            if signal_amp > 100:
                self.logger.warning(f"  WARNING: Amplitude is HUGE ({signal_amp:.2f} mV).")
                self.logger.warning(f"    This means Gain=2 is incorrect for this formula.")
                self.logger.warning(f"    Try changing GAIN to 2000 or dividing result by 1000.")
            elif signal_amp < 0.1:
                self.logger.warning(f"  WARNING: Signal is flat/dead.")
            else:
                self.logger.info(f"  Amplitude looks physiological (1-3 mV). Gain is correct.")
            # ========================================================

           
        
            # Step 4: Preprocess signal... (המשך הקוד המקורי שלך)-  

            # Step 4: Preprocess signal (filtering, clipping, interpolation)
            ecg_processed = self.preprocess_ecg_signal(ecg_mv)
            
            # Step 5: Signal Quality check
            quality = self.check_signal_quality(ecg_processed)
            if quality == "BAD":
                self.logger.error("Session rejected due to poor ECG signal quality.")
                return {}

            
            # Step 6: Detect R-peaks
            # ---------------------------------------------------------
            # 1. ניקוי האות 
            ecg_cleaned = nk.ecg_clean(ecg_processed, sampling_rate=self.sampling_rate, method="neurokit")
            
            # 2. זיהוי פעימות עם תיקון ארטיפקטים 
            # correct_artifacts=True הוא המפתח לתיקון ה-RMSSD הגבוה
            signals, info = nk.ecg_peaks(ecg_cleaned, 
                                       sampling_rate=self.sampling_rate, 
                                       method="neurokit", 
                                       correct_artifacts=True)
            
            # המרת התוצאה לפורמט שהקוד  מכיר (רשימת אינדקסים)
            peak_indices = info["ECG_R_Peaks"]
            # ---------------------------------------------------------

            # Step 7: Calculate HRV metrics
            hrv_metrics = self.calculate_hrv_metrics(peak_indices)
            
            if hrv_metrics:
                # Step 8: Save & visualize
                # 8.1: Save results
                output_file = output_path / f"{input_filename}_hrv_results.json"
                plot_file = output_path / f"{input_filename}_ecg_plot.png"
                
                # Save results with NaN replacement
                self.save_results(hrv_metrics, str(output_file), nan_replacement=0)
                
                # Create visualization
                self.visualize_results(ecg_processed, peak_indices, str(plot_file))
                
                self.logger.info("=" * 50)
                self.logger.info("Session processing completed successfully!")
                self.logger.info("=" * 50)
                
                return hrv_metrics
            else:
                self.logger.error("Failed to calculate HRV metrics")
                return {}
                
        except Exception as e:
            self.logger.error(f"Error during session processing: {e}")
            raise
