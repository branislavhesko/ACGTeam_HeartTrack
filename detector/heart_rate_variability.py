import numpy as np
from scipy.stats import zscore
import glob


def calculate_hrv_from_times(peak_times):
    if len(peak_times) < 3:
        return 0.0

    # Calculate RR intervals
    rr_intervals = np.diff(peak_times)
    
    # Remove physiologically impossible intervals (< 0.2s or > 2.0s)
    rr_intervals = rr_intervals[(rr_intervals >= 0.2) & (rr_intervals <= 2.0)]
    
    if len(rr_intervals) < 2:
        return 0.0

    hrv = np.std(rr_intervals * 1000)
    
    return hrv


if __name__ == "__main__":
    for i, locs in enumerate(glob.glob('C:/Users/vojta/Downloads/detections/detections/*.detector')):
        signal = np.loadtxt(locs)
        hrv = calculate_hrv_from_times(np.array(signal))

        if i == 4:
            hrv = calculate_hrv_from_times(np.array(signal))

        print(locs)
        print(hrv)
