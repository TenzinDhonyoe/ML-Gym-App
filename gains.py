import tkinter as tk
import customtkinter as ck
import pandas as pd
import numpy as np
import pickle
import mediapipe as mp
import cv2
from PIL import Image, ImageTk
from landmarks import landmarks

# Set up the main window
window = tk.Tk()
window.geometry("480x700")
window.title("ML Gym Tracker")
ck.set_appearance_mode("dark")

# Create and place labels for displaying information
classLabel = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="white", padx=10, text='STAGE')
classLabel.place(x=10, y=1)

counterLabel = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="white", padx=10, text='REPS')
counterLabel.place(x=160, y=1)

probLabel = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="white", padx=10, text='PROB')
probLabel.place(x=300, y=1)

classBox = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue", text='0')
classBox.place(x=10, y=41)

counterBox = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue", text='0')
counterBox.place(x=160, y=41)

probBox = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue", text='0')
probBox.place(x=300, y=41)

# Function to reset the counter
def reset_counter():
    global counter
    counter = 0

# Create and place reset button
button = ck.CTkButton(window, text='RESET', command=reset_counter, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue")
button.place(x=10, y=600)

# Set up the frame for video display
frame = tk.Frame(height=480, width=480)
frame.place(x=10, y=90)
lmain = tk.Label(frame)
lmain.place(x=0, y=0)

# Initialize Mediapipe and model
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.PPose(min_tracking_confidence=0.5, min_detection_confidence=0.5)

# Load the pre-trained model
with open('deadlift.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialize variables for tracking exercise stages
cap = cv2.VideoCapture(3)
current_stage = ''
counter = 0
bodylang_prob = np.array([0, 0])
bodylang_class = ''

# Function to detect pose and update UI
def detect():
    global current_stage, counter, bodylang_class, bodylang_prob

    ret, frame = cap.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    
    # Draw landmarks on the image
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(106, 13, 173), thickness=4, circle_radius=5),
        mp_drawing.DrawingSpec(color=(255, 102, 0), thickness=5, circle_radius=10))

    try:
        # Extract pose landmarks and make predictions
        row = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten().tolist()
        X = pd.DataFrame([row], columns=landmarks)
        bodylang_prob = model.predict_proba(X)[0]
        bodylang_class = model.predict(X)[0]

        # Update the exercise stage and counter
        if bodylang_class == "down" and bodylang_prob[bodylang_prob.argmax()] > 0.7:
            current_stage = "down"
        elif current_stage == "down" and bodylang_class == "up" and bodylang_prob[bodylang_prob.argmax()] > 0.7:
            current_stage = "up"
            counter += 1

    except Exception as e:
        print(e)

    # Update the displayed image
    img = image[:, :460, :]
    imgarr = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(imgarr)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, detect)

    # Update the labels with the current values
    counterBox.configure(text=counter)
    probBox.configure(text=f"{bodylang_prob[bodylang_prob.argmax()]:.2f}")
    classBox.configure(text=current_stage)

# Start the detection loop
detect()
window.mainloop()