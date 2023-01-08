# import necessary libraries
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.9)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
model = load_model('model')

# Load class names
classNames = ['Thumbs up', 'Thumbs down', 'Open Palm']

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read each frame from the webcam
    _, frame = cap.read()

    x, y, c = frame.shape

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmark prediction
    result = hands.process(framergb)

    # print(result)
    
    className = ''

    # post process the result
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                # print(id, lm)
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)
                landmarks.append([lmx, lmy])
            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            # Predict gesture
            prediction = model.predict([landmarks])
            pred = np.argmax(prediction)
            if prediction[0][pred]*100 >= 70.00:
                if pred == 2:
                    className = classNames[0]
                elif pred == 3:
                    className = classNames[1]
                elif pred == 5:
                    className = classNames[2]

    # show the prediction on the frame 
    cv2.putText(frame, className, (20, 60), cv2.FONT_HERSHEY_COMPLEX, 
                   2, (0,194,247), 3, cv2.LINE_AA)
    if className != '':
        print(className)
    # Show the final output
    cv2.imshow("Output", frame) 

    if cv2.waitKey(1) == ord('q'):
        break

# release the webcam and destroy all active windows
cap.release()

cv2.destroyAllWindows()