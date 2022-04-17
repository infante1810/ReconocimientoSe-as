import cv2
import mediapipe

import Operations
from Recorder import Recorder
from Camera import Camera


if __name__ == "__main__":
    # Create dataset of the videos where landmarks have not been extracted yet
    videos = Operations.load_dataset()

    # Create a DataFrame of reference signs (name: str, model: GestureModel, distance: int)
    reference_signs = Operations.load_reference_signs(videos)

    # Object that stores mediapipe results and computes sign similarities
    sign_recorder = Recorder(reference_signs)

    # Object that draws keypoints & displays results
    webcam_manager = Camera()

    # Turn on the webcam
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # Set up the Mediapipe environment
    with mediapipe.solutions.holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as holistic:
        while cap.isOpened():

            # Read feed
            ret, frame = cap.read()

            # Make detections
            image, results = Operations.mediapipe_detection(frame, holistic)

            # Process results
            sign_detected, is_recording = sign_recorder.process_results(results)

            # Update the frame (draw landmarks & display result)
            webcam_manager.update(frame, results, sign_detected, is_recording)

            pressedKey = cv2.waitKey(1) & 0xFF
            if pressedKey == ord("r"):  # Record pressing r
                sign_recorder.record()
            elif pressedKey == ord("q"):  # Break pressing q
                break

        cap.release()
        cv2.destroyAllWindows()
