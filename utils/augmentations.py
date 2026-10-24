from collections import defaultdict, Counter
import numpy as np
import random
import math

from .cleaning_landmarks import filter_detected, interpolate_missing_data


# list index of left & right arm landmarks (keys: chain, pose_hand, hand)
ARMS = {
    "left": dict(chain=[42 + i for i in (11,13,15)],
                 pose_hand=[42 + i for i in (17, 19, 21)],
                 hand=list(range(0,21))
    ),
    "right": dict(chain=[42 + i for i in (12,14,16)],
                 pose_hand=[42 + i for i in (18, 20, 22)],
                 hand=list(range(21,42))
    )
}

JOINT_INDEX = {'shoulder': 0, 'elbow': 1, 'wrist': 2}




def augment_joint_rotate(landmarks: np.ndarray, angle_range:tuple, num_new_landmarks:int,
                         arms:dict=ARMS, joints:tuple=("shoulder", "elbow", "wrist")):
    """
    Introduces small angle rotation to downstream arm joints in this order: shoulder -> elbow -> wrist
    Example:
        (1) joint = shoulder -> elbow, wrist, and hand landmarks are rotated by some randomly generated angle (a) relative to shoulder
        (2) joint = elbow -> wrist and hand lamdmarks are rotated by some randomly genenerated angle (b) relative to elbow
        (3) joint = wrist -> hand landmarks are rotated by some randomly generated angle (c) relative to wrist

    Each pivot is read after the earlier rotations have moved it, so the chain
    stays connected. Angles are drawn independently per joint, per arm, per variant,
    and are constant across all frames of a sequence. Undetected landmarks (0,0)
    are left unchanged, and a rotation is skipped in frames where its pivot is undetected.
    """
    
    out_list, angles_list = [], []
    valid = ~np.all(landmarks[:,:,:2] == 0, axis=2)     # (nframes, nlandmarks) False = MediaPipe 'undetected'

    for _ in range(num_new_landmarks):
        out = landmarks.copy()
        angles = {}
        for side, arm in arms.items():
            chain, tip = arm['chain'], arm['pose_hand'] + arm['hand']
            angles[side] = {}
            for name in joints:                 # order matters: shoulder -> elbow -> wrist
                k = JOINT_INDEX[name]
                angle = round(math.radians(random.uniform(*angle_range)), 3)
                angles[side][name] = angle
                c, s = math.cos(angle), math.sin(angle)
                matrix_rotate = np.array([[c,-s], [s,c]])

                downstream = chain[k+1:] + tip              # everything "after" this joint
                pivot = out[:, chain[k], :2][:, None, :]    # (nframes, nlandmarks, 3) -> (nframes, 2) -> (nframes, 1, 2) | CURRENT position (already moved by earlier joints)
                pts = out[:, downstream, :2]                # (nframes, ndown, 2)
                
                rotated = (pts - pivot) @ matrix_rotate.T + pivot       # rotate about pivot
                ok = valid[:, downstream] & valid[:, [chain[k]]]        # skip undetected points / undetected pivot
                out[:, downstream, :2] = np.where(ok[..., None], rotated, pts)

        out_list.append(out)
        angles_list.append(angles)

    return out_list, angles_list


def augment_rotate(landmarks:np.ndarray, origin:tuple, angle_range:tuple, num_new_landmarks:int) -> list:
    """
    IN-PLANE ROTATION. All the x,y joint coordinates in each frame are rotated by `num_new_landmarks` random angles between `angle_range[0]` to `angle_range[1]` inclusive. 
    The center of the frame (`origin`) is the point of rotation.
    
    :param landmarks: shape: (nframes, nlandmarks, 3)
    :param origin: x,y coordinates of point of origin for rotation
    :param angle_range: `(min(angle), max(angle))`
    :param num_new_landmarks: number of new independently-rotated variants to generate
    :return: list of new set of landmarks rotated relative to `landmarks` with length = `num_new_landmarks`
    """
    ox, oy = origin
    new_data, angles = [], []
    for _ in range(num_new_landmarks):
        angle = round(math.radians(random.uniform(*angle_range)),3)
        matrix_rotate = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle),  math.cos(angle)]])        # 2x2 matrix

        landmarks_2d = landmarks[:,:,:2].transpose(0,2,1).copy()        # (nframes x nlandmarks x 3) -> (nframes x nlandmarks x 2) -> (nframes x 2 x nlandmarks)
        landmarks_3 = landmarks[:,:,2:3]                                # (nframes x nlandmarks x 3) -> (nframes x nlandmarks x 1)

        # rotate(px-ox) + ox | rotate(py-oy) + oy
        landmarks_2d = landmarks_2d - np.array([ox,oy])[:,None]                     # pose_landmarks_2d[:,0,:] - ox | pose_landmarks_2d[:,1,:] - oy
        landmarks_2d_rotated = matrix_rotate @ landmarks_2d                         # (2x2) @ (nframes x 2 x nlandmarks) -> (nframes x 2 x nlandmarks)
        landmarks_2d_rotated = landmarks_2d_rotated.transpose(0,2,1)                # (nframes x 2 x nlandmarks) -> (nframes x nlandmarks x 2)
        landmarks_2d_rotated = landmarks_2d_rotated + np.array([ox, oy])            # landmarks_2d_rotated[:,:,0] + ox | landmarks_2d_rotated[:,:,1] + oy

        new_data.append(np.concatenate([landmarks_2d_rotated, landmarks_3], axis=2))  # (nframes, nlandmarks, 3)
        angles.append(angle)

    return new_data, angles

def augment_squeeze(landmarks:np.ndarray, squeeze_ratio:tuple, num_new_landmarks:int) -> list:
    """
    SQUEEZE. Each frame is squeezed horizontally: independently sampled proportions of the frame's
    width are cut from the left and right edges. y and z coordinates are left untouched. only POSE landmarks are squeezed.

    :param landmarks: shape (nframes, nlandmarks, 3)
    :param squeeze_ratio: (min, max) proportion of the width to randomly cut from each side
    :param num_new_landmarks: number of new independently-squeezed variants to generate
    :return: list of new landmark arrays, length = num_new_landmarks
    """

    new_data, ratios = [], []
    hand_landmarks, pose_landmarks = landmarks[:,:21*2,:], landmarks[:,21*2:,:]

    # (0,0,0) landmarks coords / "missing" joints are restored after the transformation
    is_zero = (pose_landmarks[:,:,0] == 0) & (pose_landmarks[:,:,1] == 0)     # (nframes, nlandmarks) | True/False

    
    for _ in range(num_new_landmarks):
        move_left = round(random.uniform(*squeeze_ratio),3)
        move_right = round(random.uniform(*squeeze_ratio),3)

        # x = 0 -> move_left, x =1 -> 1 - move_right; y & z unaffected
        scale_x = 1 - move_left - move_right
        matrix_squeeze = np.array([[scale_x, 0], [0,1]])        # 2x2 matrix
        translation = np.array([move_left, 0])                  # shift only x

        landmarks_2d = pose_landmarks[:,:,:2].transpose(0,2,1).copy()    # (nframes,2,nlandmarks)
        landmarks_3 = pose_landmarks[:,:,2:3]

        landmarks_2d_squeezed = matrix_squeeze @ landmarks_2d           # (2x2) @ (nframes,2,nlandmarks)
        landmarks_2d_squeezed = landmarks_2d_squeezed.transpose(0,2,1)  # -> (nframes, nlandmarks, 2)
        landmarks_2d_squeezed = landmarks_2d_squeezed + translation     # broadcast add [move_left, 0]

        squeezed = np.concatenate([landmarks_2d_squeezed, landmarks_3], axis=2)     # (nframes, nlandmarks, 3)

        # restore originally-missing (0,0) landmarks
        squeezed[:,:,0][is_zero] = 0
        squeezed[:,:,1][is_zero] = 0

        new_data.append(np.concatenate([hand_landmarks, squeezed], axis=1))
        ratios.append((move_left, move_right))

    return new_data, ratios


def generate_augmented_data(data:dict, labels:list, train_idx:list, target_class_size:int=30,
                            augment_methods:list[tuple]=None, weights:list[float]=None) -> tuple[dict, dict]:
    """
    Generate augmented data on training set using `augment_method` until `target_class_size` is met per class.
    
    augment_methods: list of (callable, kwargs_dict) pairs
    Each method must accept (landmarks, num_new_landmarks, ..., **kwargs) and return (new_landmarks, params) with len == num_new_landmarks.
    weights = optional sampling probability per method (uniform if None)
    """
    if not augment_methods:
        raise ValueError("Provide at least one (method, kwargs) pair.")

    # group training index by class
    grouped_train_idx = defaultdict(list)
    for idx, label in zip(train_idx, np.asarray(labels)[train_idx]):
        grouped_train_idx[int(label)].append(idx)
    grouped_train_idx = dict(sorted(grouped_train_idx.items()))

    augmented_data = {key: [] for key in grouped_train_idx}
    augmented_data_params = {key: [] for key in grouped_train_idx}
    method_ids = range(len(augment_methods))

    for key, idxs in grouped_train_idx.items():
        n_aug = target_class_size - len(idxs)
        if n_aug <= 0:      # class size = target_class_size
            continue
        base, extra = divmod(n_aug, len(idxs))

        # every data point gets `base` augmentations; `extra` random samples gets one more extra
        counts = dict().fromkeys(idxs, base)
        for i in random.sample(idxs, k=extra):
            counts[i] += 1

        for i, n_new in counts.items():
            if n_new == 0:
                continue        # skip preprocessing when nothing to generate

            # raw landmarks -> remove irrelevant frames -> interpolate misdetections
            raw_lm = data['data'][i].reshape(-1,75,3)
            ready_lm = interpolate_missing_data(filter_detected(raw_lm))

            # randomly assign each of the n_new augmentations to a method
            picks = Counter(random.choices(method_ids, weights=weights, k=n_new))

            for m_id, n_m in picks.items():
                method, method_kwargs = augment_methods[m_id]
                new_lm, params = method(ready_lm, num_new_landmarks=n_m, **method_kwargs)
                augmented_data[key].extend(new_lm)
                augmented_data_params[key].extend((method.__name__, p) for p in params)

    return augmented_data, augmented_data_params