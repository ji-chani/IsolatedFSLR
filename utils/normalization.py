import numpy as np
import math


def translate_pose_landmarks(pose_landmarks:np.ndarray) -> np.ndarray:
    """
    Creates a bounding box for `pose_landmarks` for each frame.
    A `head_unit` is defined as head height, estimated as half the distance between shoulders.
    Bounding box size: 6`head_units` wide x 8`head_units` tall
    Top edge of bounding box is 0.5`head_units` upright from left eye. Horizontal center is the nose.

    Origin is set as bottom left corner of the bounding box. Pose landmarks are shifted with respect to
    new origin.
    

    :param pose_landmarks: shape = nframes x 33 landmarks x 3 dimensions
    :return translated_pose_landmarks:
    """

    for f in range(pose_landmarks.shape[0]):
        # calculate head_unit: 1/2 dist(left -> right shoulder; x & y coords)
        head_unit = math.dist(pose_landmarks[f,11,:2], pose_landmarks[f,12,:2])/2
        
        # calculate new origin (bottom left corner of bbox)
        x0 = pose_landmarks[f,0,0] - 3*head_unit      # 6 head_units horizontal, centered at nose
        y0 = pose_landmarks[f,2,1] - 0.5*head_unit    # 0.5 head_units upward from eye

        # scale points
        pose_landmarks[f,:,0] = pose_landmarks[f,:,0] - x0
        pose_landmarks[f,:,1] = pose_landmarks[f,:,1] - y0

    return pose_landmarks


def translate_hand_landmarks(hand_landmarks:np.ndarray, padding:float=0.1) -> np.ndarray:
    """
    A smallest possible square is defined from the X and Y range of `hand_landmarks`.
    A `padding`% is added to each side which defines the edge of the bounding box.

    The origin is now defined as the center of the bounding box. Hand landmarks are shifted with respect to
    new origin. Shifted `hand_landmarks` coordinates are then each scaled to [-1,1].
    
    If hand is missing (all 0s), retain landmarks as 0s.

    :param hand_landmarks: shape = nframes x 21 landmarks x 3 dimensions
    :return translated_hand_landmarks:
    """

    for f in range(hand_landmarks.shape[0]):
        # define smallest possible square
        min_x, max_x = min(hand_landmarks[f,:,0]), max(hand_landmarks[f,:,0])
        min_y, max_y = min(hand_landmarks[f,:,1]), max(hand_landmarks[f,:,1])
        
        width, height = max_x - min_x, max_y - min_y
        # new width = width + 2*delta_x (pad left & right)
        # new height = height + 2*delta_y (pad top & bottom)
        if width > height:
            
            delta_x = padding*width
            delta_y = delta_x + ((width-height) / 2)
        else:
            delta_y = padding*height
            delta_x = delta_y + ((height-width) / 2)

        # define bounding box corner
        top_left = (min_x - delta_x, min_y - delta_y)
        bottom_right = (max_x + delta_x, max_y + delta_y)

        # define center of bounding box as origin
        center = ((top_left[0] + bottom_right[0])/2,
                (top_left[1] + bottom_right[1])/2)

        # set scale range of values to [-1,1]
        half_width = (bottom_right[0] - top_left[0])/2
        half_height = (bottom_right[1] - top_left[1])/2

        # scale landmarks
        hand_landmarks[f,:,0] = (hand_landmarks[f,:,0] - center[0])/half_width if half_width !=0 else hand_landmarks[f,:,0]
        hand_landmarks[f,:,1] = (hand_landmarks[f,:,1] - center[1])/half_height if half_height !=0 else hand_landmarks[f,:,1]

    return hand_landmarks        