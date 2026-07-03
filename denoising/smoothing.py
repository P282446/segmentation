import numpy as np
from scipy.signal import savgol_filter


# Suppress warnings
def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn


def smooth_with_savgol(data, window_length=5, polyorder=3):
    """ 
    Savitsky-Golay method to reduce noise in the data
    """
    data = np.array(data)
    smoothed_data = np.zeros_like(data)

    if len(data.shape) == 1:
        smoothed_data = savgol_filter(data, window_length, polyorder)
    else:
        if len(data) >= window_length:
            reshaped_data = data.reshape(data.shape[0], -1)
            smoothed_values = savgol_filter(reshaped_data,
                                            window_length,
                                            polyorder,
                                            axis=0)
            smoothed_data = smoothed_values.reshape(data.shape)
        else:
            smoothed_data = data

    return smoothed_data


def smooth_defined_data_with_savgol(data, window_length=5, polyorder=3):
    """
    Uses Savitsky-Golay filter everywhere possible, as there can be missing values in the data
    """
    smoothed_data = [[] for i in range(len(data))]
    good_indexes = [i for i in range(len(data)) if len(data[i]) > 0]
    contiguous_segments = find_contiguous_segments(good_indexes)
    for contiguous_segment in contiguous_segments:
        contiguous_data = [data[i] for i in contiguous_segment]
        smoothed_contiguous_data = smooth_with_savgol(contiguous_data,
                                                      window_length, polyorder)
        for i in contiguous_segment:
            smoothed_data[i] = smoothed_contiguous_data[i -
                                                        contiguous_segment[0]]
    return smoothed_data


def interpolate_missing_values(indexes, values, width):
    """
    Linear interpolation of missing values
    """
    sorted_data = sorted(zip(indexes, values))
    sorted_indexes, sorted_values = zip(*sorted_data)

    interpolated_values = [0] * width

    if sorted_indexes[0] != 0:
        interpolated_values[:sorted_indexes[0]] = [sorted_values[0]
                                                   ] * sorted_indexes[0]
    if sorted_indexes[-1] != width - 1:
        interpolated_values[sorted_indexes[-1] +
                            1:] = [sorted_values[-1]
                                   ] * (width - sorted_indexes[-1] - 1)

    for i in range(len(sorted_indexes) - 1):
        start_index, start_value = sorted_indexes[i], sorted_values[i]
        end_index, end_value = sorted_indexes[i + 1], sorted_values[i + 1]

        if start_index != end_index:
            slope = (end_value - start_value) / (end_index - start_index)
        else:
            slope = 0

        for j in range(start_index, end_index + 1):
            interpolated_values[j] = start_value + slope * (j - start_index)

    return interpolated_values


def interpolate_close_missing_values(values, max_distance=10):
    """
    Interpolates linearly missing values that are not part of a contiguous segment bigger than max_distance:
    (If we miss the landmarks for too much time, interpolating them will give us absurd results)
    """
    interpolated_values = [[] for i in range(len(values))]
    missing_indexes = [i for i in range(len(values)) if values[i] == []]
    if not missing_indexes:
        return values
    good_indexes = [i for i in range(len(values)) if i not in missing_indexes]
    for i in good_indexes:
        interpolated_values[i] = values[i]
    contiguous_segments = find_contiguous_segments(missing_indexes)
    for contiguous_segment in contiguous_segments:
        if len(contiguous_segment) < max_distance:
            if contiguous_segment[0] >= 1 and contiguous_segment[-1] < len(
                    values) - 1:
                slope = (-np.array(values[contiguous_segment[0] - 1]) +
                         values[contiguous_segment[-1] + 1]) / (
                             len(contiguous_segment) + 1)
                contiguous_interpolated_values = np.array([
                    values[contiguous_segment[0] - 1] + slope * (i + 1)
                    for i in range(len(contiguous_segment))
                ])
                for i in contiguous_segment:
                    interpolated_values[i] = contiguous_interpolated_values[
                        i - contiguous_segment[0]]
    return interpolated_values


def find_contiguous_segments(indexes):
    """
    Finds contiguous segments inside a list of indexes:
    Example
    [1,2,3,5,6,8,9] -> [[1,2,3],[5,6],[8,9]]
    """
    contiguous_segments = []
    last_index = indexes[0]
    current_contiguous_segment = [last_index]
    for index in indexes[1:]:
        if index - last_index == 1:
            current_contiguous_segment.append(index)
        else:
            contiguous_segments.append(current_contiguous_segment)
            current_contiguous_segment = [index]
        last_index = index
    contiguous_segments.append(current_contiguous_segment)
    return contiguous_segments
