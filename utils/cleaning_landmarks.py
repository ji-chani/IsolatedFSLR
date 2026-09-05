import numpy as np
from scipy.interpolate import interp1d
import copy

def filter_detected(data:np.ndarray) -> np.ndarray:
    """
    Remove frames where both hands are not detected. 
    :param data: n_frames x n_landmarks x 3 - dimensional array
    :return: filtered_data
    """
    hands_idx = list(range(42))
    matrix_checker = np.zeros((42,3))
    keep_frame = []

    # check each frame
    for frame in range(data.shape[0]):
        if (data[frame, hands_idx] == matrix_checker).sum() < 42*3:  # filter frames that detects at least 1 hand
            keep_frame.append(frame)

    return data[keep_frame]


def interpolate_missing_data(data:np.ndarray):
    """
    If frame has zero (missing) values for the hand, then prev and next
    frame is checked. If both frames have nonzero values, interpolation
    is implemented to estimate the missing values.

    :param data: n_frames x n_landmarks x 3 - dimensional array
    :return: interpolated_data
    """

    data = copy.deepcopy(data)
    hand_idx = [(0, 21), (21, 42)]
    matrix_checker = np.zeros((21,3))
    for hand in hand_idx:
        
        # filter frames where hand is missing
        zero_frames = []
        for frame in range(data.shape[0]):  # iterates through all frames
            if (data[frame, hand[0]:hand[1]] == matrix_checker).sum() == 21*3:
                zero_frames.append(frame)

        # cluster missing frames if they are consecutive
        if len(zero_frames) != 0:
            clustered_zero_frames, new_cluster = [], []
            for frame in zero_frames:
                if len(new_cluster) == 0:
                    new_cluster.append(frame)
                    continue
                
                # if new frame is the next number
                if new_cluster[-1]+1 == frame:
                    new_cluster.append(frame)
                else:
                    clustered_zero_frames.append(new_cluster)
                    new_cluster = [frame]
            clustered_zero_frames.append(new_cluster)

        else:
            continue
        
        # remove first and last frame so interpolation can be applied properly
        # remove first frame if it is in first clustered zero frame
        if len(clustered_zero_frames) != 0:
            if 0 in clustered_zero_frames[0]:
                clustered_zero_frames.pop(0)

        # remove last frame if it is in last clustered zero frame
        if len(clustered_zero_frames) != 0:
            if data.shape[0]-1 in clustered_zero_frames[-1]:
                clustered_zero_frames.pop(-1)

        # interpolate over missing values
        interp_data = data[:, hand[0]:hand[1]].reshape(data.shape[0], 21*3)  # flatten coordinates (21x3 -> 63)
        for cluster in clustered_zero_frames:
            lframe, rframe = cluster[0]-1, cluster[-1]+1
            for i in range(21*3):
                interpfunc = interp1d([lframe, rframe], 
                                      [interp_data[lframe,i], interp_data[rframe,i]],
                                      kind='linear')
                interp_data[cluster,i] = interpfunc(cluster)

        data[:, hand[0]:hand[1]] = interp_data.reshape(data.shape[0], 21, 3)
    
    return data


