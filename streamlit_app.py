# import streamlit as st # type: ignore
# import cv2

# import mediapipe as mp

# import math
# from bicep_curl import calculate_angle as calculate_bicep_angle
#   # Importing from bicep curl file
# from shoulder_press import calculate_angle as calculate_shoulder_angle  # Importing from shoulder press file

# # Function to initialize pose and video capture
# def init_video_capture():
#     # mp_pose = mp.solutions.pose
#     # pose = mp_pose.Pose()
#     # cap = cv2.VideoCapture(0)
#     # return cap, pose
#     def init_video_capture():
#         cap = cv2.VideoCapture(0)  # Open the webcam
#         mp_pose = mp.solutions.pose  # Initialize Mediapipe pose
#         pose = mp_pose.Pose()  # Create a pose object
#         return cap, pose


# # Function to display exercise frame and count reps for bicep curls
# def bicep_curl_exercise(cap, pose):
#     left_rep_count = 0
#     right_rep_count = 0
#     left_in_rep = False
#     right_in_rep = False
#     left_set_count = 1
#     right_set_count = 1

#     angle_threshold_lower = 50
#     angle_threshold_upper = 160
#     reps_per_set = 13

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = pose.process(frame_rgb)

#         if results.pose_landmarks:
#             left_shoulder = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * frame.shape[1],
#                              results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * frame.shape[0])
#             left_elbow = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x * frame.shape[1],
#                           results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y * frame.shape[0])
#             left_wrist = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * frame.shape[1],
#                           results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * frame.shape[0])

#             right_shoulder = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * frame.shape[1],
#                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * frame.shape[0])
#             right_elbow = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x * frame.shape[1],
#                            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y * frame.shape[0])
#             right_wrist = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x * frame.shape[1],
#                            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * frame.shape[0])

#             left_angle = calculate_bicep_angle(left_shoulder, left_elbow, left_wrist)
#             right_angle = calculate_bicep_angle(right_shoulder, right_elbow, right_wrist)

#             # Rep counting logic for left arm
#             if left_angle > angle_threshold_upper and not left_in_rep:
#                 left_in_rep = True
#             elif left_angle < angle_threshold_lower and left_in_rep:
#                 left_rep_count += 1
#                 left_in_rep = False
#                 if left_rep_count == reps_per_set:
#                     left_rep_count = 0
#                     left_set_count += 1

#             # Rep counting logic for right arm
#             if right_angle > angle_threshold_upper and not right_in_rep:
#                 right_in_rep = True
#             elif right_angle < angle_threshold_lower and right_in_rep:
#                 right_rep_count += 1
#                 right_in_rep = False
#                 if right_rep_count == reps_per_set:
#                     right_rep_count = 0
#                     right_set_count += 1

#         # Display rep count and set count for bicep curls
#         cv2.putText(frame, f'Left Reps: {left_rep_count} / Set {left_set_count}',
#                     (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
#         cv2.putText(frame, f'Right Reps: {right_rep_count} / Set {right_set_count}',
#                     (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

#         # Show frame in Streamlit
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         st.image(frame_rgb, channels="RGB", use_column_width=True)

#         if cv2.waitKey(10) & 0xFF == ord('q'):
#             break

#     cap.release()

# # Function to display exercise frame and count reps for shoulder press
# def shoulder_press_exercise(cap, pose):
#     left_shoulder_press_rep_count = 0
#     right_shoulder_press_rep_count = 0
#     left_shoulder_press_in_rep = False
#     right_shoulder_press_in_rep = False
#     shoulder_press_set_count = 1
#     angle_threshold_lower = 160
#     angle_threshold_upper = 30
#     reps_per_set = 15

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = pose.process(frame_rgb)

#         if results.pose_landmarks:
#             left_shoulder = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * frame.shape[1],
#                              results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * frame.shape[0])
#             left_elbow = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x * frame.shape[1],
#                           results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y * frame.shape[0])
#             left_wrist = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * frame.shape[1],
#                           results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * frame.shape[0])

#             right_shoulder = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * frame.shape[1],
#                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * frame.shape[0])
#             right_elbow = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x * frame.shape[1],
#                            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y * frame.shape[0])
#             right_wrist = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x * frame.shape[1],
#                            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * frame.shape[0])

#             left_angle = calculate_shoulder_angle(left_shoulder, left_elbow, left_wrist)
#             right_angle = calculate_shoulder_angle(right_shoulder, right_elbow, right_wrist)

#             # Rep counting logic for shoulder press
#             if left_angle < angle_threshold_upper and not left_shoulder_press_in_rep:
#                 left_shoulder_press_in_rep = True
#             elif left_angle > angle_threshold_lower and left_shoulder_press_in_rep:
#                 left_shoulder_press_rep_count += 1
#                 left_shoulder_press_in_rep = False
#                 if left_shoulder_press_rep_count == reps_per_set:
#                     left_shoulder_press_rep_count = 0
#                     shoulder_press_set_count += 1

#             if right_angle < angle_threshold_upper and not right_shoulder_press_in_rep:
#                 right_shoulder_press_in_rep = True
#             elif right_angle > angle_threshold_lower and right_shoulder_press_in_rep:
#                 right_shoulder_press_rep_count += 1
#                 right_shoulder_press_in_rep = False
#                 if right_shoulder_press_rep_count == reps_per_set:
#                     right_shoulder_press_rep_count = 0
#                     shoulder_press_set_count += 1

#         # Display rep count and set count for shoulder press
#         cv2.putText(frame, f'Left Shoulder Press Reps: {left_shoulder_press_rep_count} / Set {shoulder_press_set_count}',
#                     (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
#         cv2.putText(frame, f'Right Shoulder Press Reps: {right_shoulder_press_rep_count} / Set {shoulder_press_set_count}',
#                     (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

#         # Show frame in Streamlit
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         st.image(frame_rgb, channels="RGB", use_column_width=True)

#         if cv2.waitKey(10) & 0xFF == ord('q'):
#             break

#     cap.release()

# # Streamlit UI
# st.title("Exercise Rep Counter")
# exercise_choice = st.selectbox("Choose Exercise", ["Bicep Curl", "Shoulder Press"])

# cap, pose = init_video_capture()

# if exercise_choice == "Bicep Curl":
#     bicep_curl_exercise(cap, pose)
# elif exercise_choice == "Shoulder Press":
#     shoulder_press_exercise(cap, pose)

# import streamlit as st
# import cv2
# import mediapipe as mp
# from bicep_curl import calculate_angle as calculate_bicep_angle  # Import from bicep_curl.py
# from shoulder_press import calculate_angle as calculate_shoulder_angle  # Import from shoulder_press.py

# # Initialize MediaPipe pose model
# mp_pose = mp.solutions.pose

# def init_video_capture():
#     cap = cv2.VideoCapture(0)  # Open the webcam
#     pose = mp_pose.Pose()  # Create a pose object
#     return cap, pose

# # Bicep curl exercise logic
# def bicep_curl_exercise(cap, pose):
#     left_rep_count = 0
#     right_rep_count = 0
#     left_in_rep = False
#     right_in_rep = False
#     angle_threshold_lower = 50
#     angle_threshold_upper = 160

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = pose.process(frame_rgb)

#         if results.pose_landmarks:
#             left_shoulder = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * frame.shape[1],
#                              results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * frame.shape[0])
#             left_elbow = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x * frame.shape[1],
#                           results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y * frame.shape[0])
#             left_wrist = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * frame.shape[1],
#                           results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * frame.shape[0])

#             right_shoulder = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * frame.shape[1],
#                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * frame.shape[0])
#             right_elbow = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x * frame.shape[1],
#                            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y * frame.shape[0])
#             right_wrist = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x * frame.shape[1],
#                            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * frame.shape[0])

#             left_angle = calculate_bicep_angle(left_shoulder, left_elbow, left_wrist)
#             right_angle = calculate_bicep_angle(right_shoulder, right_elbow, right_wrist)

#             # Rep counting logic for left arm
#             if left_angle > angle_threshold_upper and not left_in_rep:
#                 left_in_rep = True
#             elif left_angle < angle_threshold_lower and left_in_rep:
#                 left_rep_count += 1
#                 left_in_rep = False

#             # Rep counting logic for right arm
#             if right_angle > angle_threshold_upper and not right_in_rep:
#                 right_in_rep = True
#             elif right_angle < angle_threshold_lower and right_in_rep:
#                 right_rep_count += 1
#                 right_in_rep = False

#         # Display the count on the frame
#         cv2.putText(frame, f'Left Reps: {left_rep_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
#         cv2.putText(frame, f'Right Reps: {right_rep_count}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

#         # Show frame in Streamlit
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         st.image(frame_rgb, channels="RGB", use_column_width=True)

#         # Break loop if 'q' is pressed
#         if cv2.waitKey(10) & 0xFF == ord('q'):
#             break

#     cap.release()

# # Shoulder press exercise logic (Similar to bicep_curl_exercise)
# def shoulder_press_exercise(cap, pose):
#     left_rep_count = 0
#     right_rep_count = 0
#     left_in_rep = False
#     right_in_rep = False
#     angle_threshold_lower = 50
#     angle_threshold_upper = 160

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = pose.process(frame_rgb)

#         if results.pose_landmarks:
#             left_shoulder = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * frame.shape[1],
#                              results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * frame.shape[0])
#             left_elbow = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x * frame.shape[1],
#                           results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y * frame.shape[0])
#             left_wrist = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * frame.shape[1],
#                           results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * frame.shape[0])

#             right_shoulder = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * frame.shape[1],
#                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * frame.shape[0])
#             right_elbow = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x * frame.shape[1],
#                            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y * frame.shape[0])
#             right_wrist = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x * frame.shape[1],
#                            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * frame.shape[0])

#             left_angle = calculate_shoulder_angle(left_shoulder, left_elbow, left_wrist)
#             right_angle = calculate_shoulder_angle(right_shoulder, right_elbow, right_wrist)

#             # Rep counting logic for left arm
#             if left_angle > angle_threshold_upper and not left_in_rep:
#                 left_in_rep = True
#             elif left_angle < angle_threshold_lower and left_in_rep:
#                 left_rep_count += 1
#                 left_in_rep = False

#             # Rep counting logic for right arm
#             if right_angle > angle_threshold_upper and not right_in_rep:
#                 right_in_rep = True
#             elif right_angle < angle_threshold_lower and right_in_rep:
#                 right_rep_count += 1
#                 right_in_rep = False

#         # Display the count on the frame
#         cv2.putText(frame, f'Left Reps: {left_rep_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
#         cv2.putText(frame, f'Right Reps: {right_rep_count}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

#         # Show frame in Streamlit
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         st.image(frame_rgb, channels="RGB", use_column_width=True)

#         # Break loop if 'q' is pressed
#         if cv2.waitKey(10) & 0xFF == ord('q'):
#             break

#     cap.release()

# # Streamlit UI
# st.title("Exercise Rep Counter")

# # Step 1: Main page with exercise selection
# exercise_choice = st.selectbox("Choose Exercise", ["Select Exercise", "Bicep Curl", "Shoulder Press"])

# if exercise_choice != "Select Exercise":
#     # Initialize video capture and pose model
#     cap, pose = init_video_capture()

#     # Step 2: Handle exercise selection and start video capture
#     if exercise_choice == "Bicep Curl":
#         st.subheader("Bicep Curl Rep Counter")
#         bicep_curl_exercise(cap, pose)  # Start bicep curl exercise
#     elif exercise_choice == "Shoulder Press":
#         st.subheader("Shoulder Press Rep Counter")
#         shoulder_press_exercise(cap, pose)  # Start shoulder press exercise

#     # Release video capture after exercise is done
#     cap.release()
# else:
#     st.info("Please select an exercise to begin.")
# import streamlit as st
# from bicep_curl import bicep_curl_exercise
# from shoulder_press import shoulder_press_exercise

# def main():
#     st.title("Exercise Counter")

#     exercise = st.selectbox("Select Exercise", ("Bicep Curl", "Shoulder Press"))

#     if exercise == "Bicep Curl":
#         if st.button("Start Bicep Curl Exercise"):
#             bicep_curl_exercise()
#     elif exercise == "Shoulder Press":
#         if st.button("Start Shoulder Press Exercise"):
#             shoulder_press_exercise()

# if __name__ == "__main__":
#     main()
# import streamlit as st
# from bicep_curl import bicep_curl_exercise  # Import function for bicep curls
# from shoulder_press import shoulder_press_exercise  # Import function for shoulder press

# def main():
#     st.title("Exercise Repetition Counter")
#     st.write("Select an exercise below to start counting repetitions:")

#     # Drop-down menu to choose between exercises
#     exercise = st.selectbox("Choose an Exercise", ["", "Bicep Curl", "Shoulder Press"])

#     # Run the corresponding exercise function based on user selection
#     if exercise == "Bicep Curl":
#         if st.button("Start Bicep Curl"):
#             bicep_curl_exercise()
#     elif exercise == "Shoulder Press":
#         if st.button("Start Shoulder Press"):
#             shoulder_press_exercise()
#     else:
#         st.info("Please select an exercise to begin.")

# if __name__ == "__main__":
#     main()

# import streamlit as st
# import os
# from bicep_curl import bicep_curl_exercise
# from shoulder_press import shoulder_press_exercise

# def main():
#     st.title("Exercise Repetition Counter")
#     st.write("Select an exercise below to start counting repetitions:")

#     exercise = st.selectbox("Choose an Exercise", ["", "Bicep Curl", "Shoulder Press"])

#     if "STREAMLIT_ENV" in os.environ:
#         st.warning("Camera access is disabled in this environment, so exercise counting is unavailable.")
#     else:
#         if exercise == "Bicep Curl" and st.button("Start Bicep Curl"):
#             bicep_curl_exercise()
#         elif exercise == "Shoulder Press" and st.button("Start Shoulder Press"):
#             shoulder_press_exercise()
#         else:
#             st.info("Please select an exercise to begin.")

# if __name__ == "__main__":
#     main()


# import streamlit as st
# import os
# from bicep_curl import bicep_curl_exercise
# from shoulder_press import shoulder_press_exercise

# def main():
#     st.title("Exercise Repetition Counter")
#     st.write("Select an exercise below to start counting repetitions:")

#     exercise = st.selectbox("Choose an Exercise", ["", "Bicep Curl", "Shoulder Press"])

#     if "STREAMLIT_ENV" in os.environ:
#         st.warning("Camera access is disabled in this environment, so exercise counting is unavailable.")
#     else:
#         if exercise == "Bicep Curl" and st.button("Start Bicep Curl"):
#             bicep_curl_exercise()
#         elif exercise == "Shoulder Press" and st.button("Start Shoulder Press"):
#             shoulder_press_exercise()
#         else:
#             st.info("Please select an exercise to begin.")

# if __name__ == "__main__":
#     main()

# import streamlit as st

# def main():
#     st.title("Exercise Repetition Counter")
#     st.write("Select an exercise below to start counting repetitions:")

#     exercise = st.selectbox("Choose an Exercise", ["", "Bicep Curl", "Shoulder Press"])

#     if "STREAMLIT_ENV" in os.environ:
#         st.warning("Camera access is disabled in this environment, so exercise counting is unavailable.")
#     else:
#         if exercise == "Bicep Curl" and st.button("Start Bicep Curl"):
#             bicep_curl_exercise()
#         elif exercise == "Shoulder Press" and st.button("Start Shoulder Press"):
#             shoulder_press_exercise()
#         else:
#             st.info("Please select an exercise to begin.")

# def bicep_curl_exercise():
#     st.camera_input("Capture your exercise here")

# if __name__ == "__main__":
#     main()

import streamlit as st
import os
from bicep_curl import bicep_curl_exercise
from shoulder_press import shoulder_press_exercise
import streamlit_webrtc as webrtc

def main():
    st.title("Exercise Repetition Counter")
    st.write("Select an exercise below to start counting repetitions:")

    exercise = st.selectbox("Choose an Exercise", ["", "Bicep Curl", "Shoulder Press"])

    if "STREAMLIT_ENV" in os.environ:
        st.warning("Camera access is disabled in this environment, so exercise counting is unavailable.")
    else:
        if exercise == "Bicep Curl" and st.button("Start Bicep Curl"):
            start_bicep_curl()
        elif exercise == "Shoulder Press" and st.button("Start Shoulder Press"):
            start_shoulder_press()
        else:
            st.info("Please select an exercise to begin.")

def start_bicep_curl():
    # Here, we use WebRTC to access the webcam for the exercise counting
    webrtc_stream = webrtc.WebRtcMode.SENDRECV
    webrtc.config = webrtc.VideoEncoderConfig(
        codec="vp8", 
        quality=8, 
        resolution="720p"
    )
    webrtc.run_video_streaming(bicep_curl_exercise)  # Pass the exercise function to WebRTC

def start_shoulder_press():
    # Similar to the bicep curl, use WebRTC for shoulder press
    webrtc_stream = webrtc.WebRtcMode.SENDRECV
    webrtc.config = webrtc.VideoEncoderConfig(
        codec="vp8", 
        quality=8, 
        resolution="720p"
    )
    webrtc.run_video_streaming(shoulder_press_exercise)  # Pass the exercise function to WebRTC

if __name__ == "__main__":
    main()
