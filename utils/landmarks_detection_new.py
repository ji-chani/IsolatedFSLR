import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HolisticLandmarker, HolisticLandmarkerOptions, RunningMode
import cv2
import numpy as np

# import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Video2Landmarks:
    def __init__(self, model_path:str, 
                 landmark_type:str='relative',
                 landmark_include_ind:int=0, 
                 display_vid:bool=False, 
                 remove_duplicate_landmarks:bool=True):
        """
        Args:
            model_path: path to the holistic_landmarker.task model file
            landmark_type: 'relative' (normalized image coordinates) or 'world' (real-world metrics)
            https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/1/holistic_landmarker.task
            landmark_include_ind: integer indicating which landmarks to include
                0 - hands only
                1 - pose only
                2 - face only
                3 - hands + pose
                4 - hands + pose + face
            display_vid: whether to display annotated video while processing
            remove_duplicate_landmarks: whether to remove pose landmarks that are redundant
        """

        self.model_path = model_path
        self.landmark_type = landmark_type
        self.landmark_include_ind = landmark_include_ind
        self.display_vid = display_vid
        self.remove_duplicate_landmarks = remove_duplicate_landmarks

    def transform(self, vid_path:str):
        self.path = vid_path
        complete_landmarks = []

        cap = cv2.VideoCapture(self.path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        options = HolisticLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=RunningMode.VIDEO,
            min_face_detection_confidence=0.5,
            min_face_landmarks_confidence=0.5,
            min_pose_detection_confidence=0.5,
            min_pose_landmarks_confidence=0.5,
            min_hand_landmarks_confidence=0.5
        )

        with HolisticLandmarker.create_from_options(options) as landmarker:
            for f in range(n_frames):
                ret, frame = cap.read()
                if not ret:
                    break

                # Tasks API require RGB; OpeCV gives BGR
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                # timestamp must be monotonically increasing (in ms)
                timestamp_ms = int((f/fps)*1000)

                # detection
                results = landmarker.detect_for_video(mp_image, timestamp_ms)

                # save landmarks
                complete_landmarks.append(extract_landmarks(results, self.landmark_type, self.landmark_include_ind, self.remove_duplicate_landmarks))

                if self.display_vid:
                    annotated_image = draw_styled_landmarks(frame, results, self.landmark_include_ind, self.remove_duplicate_landmarks)
                    cv2.imshow("Video with Mediapipe Landmarks", annotated_image)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        cap.release()
        cv2.destroyAllWindows()

        return np.array(complete_landmarks), results


############ HELPER FUNCTIONS AND VARIABLES #################
def extract_landmarks(results, landmark_type:str, landmark_include_ind:int, remove_duplicate_landmarks:bool) -> np.ndarray:
    """ Extract landmarks from a HolisticLandmarker result into a flat array.
    
    Relative landmarks (each is a list with 0-1 (normalized) entries):
      - results.pose_landmarks          → 33 landmarks (x, y, z)
      - results.face_landmarks          → 478 landmarks (x, y, z)
      - results.left_hand_landmarks     → 21 landmarks (x, y, z)
      - results.right_hand_landmarks    → 21 landmarks (x, y, z)
    
    World landmarks (list of metric distances relative to an origin):
      - results.pose_world_landmarks          → 33 landmarks (x, y, z)
      - results.left_hand_world_landmarks     → 21 landmarks (x, y, z)
      - results.right_hand_world_landmarks    → 21 landmarks (x, y, z)
    
    !!! Mediapipe Task has no face_world_landmarks feature !!!
      Returns a flat numpy array of shape (N*3,) for downstream use, or zeros if a component is not detected.
    """

    def landmarks_to_array(landmark_list, expected_count, flatten:bool=True):
        if landmark_list:
            return np.array([[lm.x, lm.y, lm.z] for lm in landmark_list]).flatten() if flatten else np.array([[lm.x, lm.y, lm.z] for lm in landmark_list])
        return np.zeros(expected_count*3)
    
    pose  = landmarks_to_array(results.pose_landmarks,        33) if landmark_type == 'relative' else landmarks_to_array(results.pose_world_landmarks,        33)
    face  = landmarks_to_array(results.face_landmarks,       478)
    lhand = landmarks_to_array(results.left_hand_landmarks,   21) if landmark_type == 'relative' else landmarks_to_array(results.left_hand_world_landmarks,        21)
    rhand = landmarks_to_array(results.right_hand_landmarks,  21) if landmark_type == 'relative' else landmarks_to_array(results.right_hand_world_landmarks,        21)

    if landmark_include_ind == 0:                         # hands only
        return np.concatenate([lhand, rhand])
    elif landmark_include_ind == 1:                       # pose only
        return np.concatenate([pose[:15]]) if remove_duplicate_landmarks else np.concatenate([pose])
    elif landmark_include_ind == 2:                       # face only
        return np.concatenate([face])
    elif landmark_include_ind == 3:                       # hands + pose
        return np.concatenate([lhand, rhand, pose[:15]]) if remove_duplicate_landmarks else np.concatenate([lhand, rhand, pose])
    else:                                                 # hands + pose + face
        return np.concatenate([lhand, rhand, pose[11:15], face])  if remove_duplicate_landmarks else np.concatenate([lhand, rhand, pose, face])

# Pose: 33 landmarks (BlazePose)
POSE_CONNECTIONS = frozenset([
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
    (9,10),(11,12),(11,13),(13,15),(15,17),(15,19),(15,21),(17,19),
    (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
    (11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28),
    (27,29),(28,30),(29,31),(30,32),(27,31),(28,32)
])

# Hand: 21 landmarks
HAND_CONNECTIONS = frozenset([
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
])

# Face: 478 landmarks (FaceMesh contours subset)
FACEMESH_CONTOURS = frozenset([
    (10,338),(338,297),(297,332),(332,284),(284,251),(251,389),(389,356),
    (356,454),(454,323),(323,361),(361,288),(288,397),(397,365),(365,379),
    (379,378),(378,400),(400,377),(377,152),(152,148),(148,176),(176,149),
    (149,150),(150,136),(136,172),(172,58),(58,132),(132,93),(93,234),
    (234,127),(127,162),(162,21),(21,54),(54,103),(103,67),(67,109),(109,10),
    (33,7),(7,163),(163,144),(144,145),(145,153),(153,154),(154,155),
    (155,133),(133,173),(173,157),(157,158),(158,159),(159,160),(160,161),
    (161,246),(246,33),(362,382),(382,381),(381,380),(380,374),(374,373),
    (373,390),(390,249),(249,263),(263,466),(466,388),(388,387),(387,386),
    (386,385),(385,384),(384,398),(398,362)
])

def draw_styled_landmarks(bgr_frame:np.ndarray, results, landmark_include_ind:int, remove_duplicate_landmarks:bool) -> np.ndarray:
    image = bgr_frame.copy()

    def draw(landmarks, connections, dot_color, line_color):
        h, w = image.shape[:2]
        if connections:
            for a,b in connections:
                if a < len(landmarks) and b < len(landmarks):
                    x1, y1 = int(landmarks[a].x * w), int(landmarks[a].y * h)
                    x2, y2 = int(landmarks[b].x * w), int(landmarks[b].y * h)
                    cv2.line(image, (x1, y1), (x2, y2), line_color, 1)
        for lm in landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(image, (cx,cy), 2, dot_color, 2)
    
    if landmark_include_ind == 0 or landmark_include_ind >= 3:
        if results.right_hand_landmarks:
            draw(results.right_hand_landmarks, HAND_CONNECTIONS, (121, 22, 76),  (121, 44, 250))
        if results.left_hand_landmarks:
            draw(results.left_hand_landmarks, HAND_CONNECTIONS, (121, 22, 76),  (121, 44, 250))

    if landmark_include_ind == 1 or landmark_include_ind >= 3:
        if results.pose_landmarks:
            if remove_duplicate_landmarks:
                draw(results.pose_landmarks[11:15], frozenset([(0,1), (0,2), (1,3)]),
                     (80, 22, 10),   (80, 44, 121))
            else:
                draw(results.pose_landmarks,       POSE_CONNECTIONS,     (80, 22, 10),   (80, 44, 121))
    
    if landmark_include_ind == 2 or landmark_include_ind == 4:
        if results.face_landmarks:
            draw(results.face_landmarks,       FACEMESH_CONTOURS,    (80, 110, 10),  (80, 256, 121))
    
    return image


if __name__ == "__main__":
    vid2landmarks = Video2Landmarks(model_path="holistic_landmarker.task", 
                                    landmark_include_ind=4,
                                    display_vid=True,
                                    remove_duplicate_landmarks=True)
    landmarks, results = vid2landmarks.transform('clips/4/7.MOV')

    print(len(landmarks[0]))