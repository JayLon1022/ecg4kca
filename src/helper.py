"""
helper functions for the project

drafted by Juntang Wang at May 25th 2025
"""

import os
import wfdb
import numpy as np
import warnings
from scipy.signal import butter, filtfilt, sosfiltfilt, resample



class HeaParser:
    def __init__(self, base_dir):
        self.base_dir = base_dir

    def _get_hea_path(self, subject_id, study_id):
        return os.path.join(
            self.base_dir, 
            "p" + subject_id[:4], 
            "p" + subject_id, 
            "s" + study_id, 
            study_id + ".hea"
        )
        
    def _check_heapath(self, hea_path):
        if not os.path.exists(hea_path): 
            raise FileNotFoundError(f"Not found: {hea_path}")
        return hea_path.replace(".hea", "")

    def _plot_record(self, record, subject_id, study_id):
        wfdb.plot_wfdb(
            record=record,
            title=f'ECG - Subject {subject_id} (Study {study_id})\n \
            Record Time: {record.base_date} {record.base_time}, \
            Sampling Rate: {record.fs} Hz',
            ecg_grids='all',
            figsize=(10, 1.3*record.n_sig),
        )
        
    def _notch_filter(self, signal, fs, freq=60.0, q=30.0):
        """Remove mains interference."""
        from scipy.signal import iirnotch
        nyq = 0.5 * fs
        w0 = freq / nyq
        b, a = iirnotch(w0, q)
        return filtfilt(b, a, signal, axis=0)    
    
    def _butter_filter(self, signal, fs, lowcut=1, highcut=40.0):
        """Bandpass filter using Butterworth."""
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(4, [low, high], btype='band', output='sos')
        return sosfiltfilt(sos, signal, axis=0)
    
    def _normalize(self, signal):
        """Per-lead z-score normalization."""
        mean = np.mean(signal, axis=0, keepdims=True)
        std = np.std(signal, axis=0, keepdims=True)
        return (signal - mean) / (std + 1e-8)

    def _resample(self, signal, old_fs, new_fs):
        """Resample signal to new sampling frequency."""
        n_samples = int(signal.shape[0] * new_fs / old_fs)
        return resample(signal, n_samples, axis=0)

    def _align_to_peak(self, record, target_duration=7.0):
        """Align record to 7 seconds with peak at 0.42 second mark."""
        from scipy.signal import find_peaks
        
        fs = record.fs
        peak_position_samples = int(fs * 0.42) # ROI stars
        _end = int(fs * (0.42 + 1.5)) # 1.5 seconds after ROI
        target_samples = int(fs * target_duration)
        signal_ch2 = record.p_signal[peak_position_samples:_end, 1]
        peaks, _ = find_peaks(signal_ch2, height=2.5)
        
        if len(peaks) == 0:
            warnings.warn("No peak > 2.5 found, using any peak")
            peaks, _ = find_peaks(signal_ch2)
        
        if len(peaks) == 0:
            warnings.warn("No peaks found, aligning to chosen anchor")
            peak_idx = peak_position_samples
        else:
            peak_idx = peak_position_samples + peaks[0]
        
        start_idx = peak_idx - peak_position_samples
        end_idx = start_idx + target_samples
        
        record.p_signal = record.p_signal[start_idx:end_idx]
        return record

    def parse(
        self,
        subject_id,
        study_id,
        butter=True,
        notch=True,
        normalize=True,
        resample_fs=128,
        align_peak=True,
        plot=False,
    ):
        hea_path = self._get_hea_path(subject_id, study_id)
        record_name = self._check_heapath(hea_path)
        record = wfdb.rdrecord(record_name, sampfrom=250, sampto=4750)
        
        # preprocess
        fs = record.fs
        if notch:
            record.p_signal = self._notch_filter(record.p_signal, fs)
        if butter:
            record.p_signal = self._butter_filter(record.p_signal, fs)
        if normalize:
            record.p_signal = self._normalize(record.p_signal)
        if resample_fs and fs != resample_fs:
            record.p_signal = self._resample(record.p_signal, fs, resample_fs)
            record.fs = resample_fs
        if align_peak:
            record = self._align_to_peak(record)
        if normalize:
            record.p_signal = self._normalize(record.p_signal)

        if plot:
            self._plot_record(record, subject_id, study_id)
        return record
