import cv2
import mediapipe as mp
import time
import numpy as np
import uuid
import os
import math

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

count = 0

cap = cv2.VideoCapture(0)
with mp_hands.Hands(min_detection_confidence = 0.8, min_tracking_confidence = 0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()

        # BGR 2 RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Flip on horizontal
        image = cv2.flip(image, 1)
        
        # Set flag
        image.flags.writeable = False
        
        # Detections
        results = hands.process(image)
        
        # Set flag to true
        image.flags.writeable = True
        
        # RGB 2 BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                # Detections
        print(results)
        
        # Rendering results
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(0, 0, 76), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(0, 0, 250), thickness=2, circle_radius=2),
                                         )
        
        #save images into file
        #imgDir=r"C:/Users/georg/OneDrive\Desktop/gang_recogs/Output Images"
        #os.chdir(imgDir)
        #filename='{}.jpg'.format(uuid.uuid1())
        #it might be worth using a stationary filename, or maybe 2 or 3 alternating names to save space
        #cv2.imwrite(filename, image)
       
        cv2.imshow('Hand tracker', image)


        #if halt_command == True or cv2.waitKey(10) & 0xFF == ord('q'):
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        try:
            for hand_landmarks in results.multi_hand_landmarks:
                Tx=hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x
                Ty=hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y
                print(
                    f'Thumb: (',
                    f'{Tx}, '
                    f'{Ty})'
                )
                Ix=hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
                Iy=hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
                print(
                    f'Index: (',
                    f'{Ix}, '
                    f'{Iy})'
                )
                Mx=hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x
                My=hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
                print(
                    f'Middle: (',
                    f'{Mx}, '
                    f'{My})'
                )
                Rx=hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x
                Ry=hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y
                print(
                    f'Ring: (',
                    f'{Rx}, '
                    f'{Ry})'
                )
                Px=hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x
                Py=hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y
                print(
                    f'Pinky: (',
                    f'{Px}, '
                    f'{Py})'
                )
                #If the middle-ring distance is small relative to the middle-index and ring-pinky distances, then a specific gesture or hand pose is detected.
                if 7 * math.sqrt((Mx-Rx)**2 + (My-Ry)**2) < math.sqrt((Mx-Ix)**2 + (My-Iy)**2) and 5 * math.sqrt((Mx-Rx)**2 + (My-Ry)**2) < math.sqrt((Rx-Px)**2 + (Ry-Py)**2) and count <3:
                    count +=1
                    print("True")
                elif 7 * math.sqrt((Mx-Rx)**2 + (My-Ry)**2) < math.sqrt((Mx-Ix)**2 + (My-Iy)**2) and 5 * math.sqrt((Mx-Rx)**2 + (My-Ry)**2) < math.sqrt((Rx-Px)**2 + (Ry-Py)**2) and count >=3:
                    print("Saved")
                    #saves to output images in same directory
                    imgDir = os.path.join(os.path.dirname(__file__), "Output Images")
                    os.chdir(imgDir)
                    filename='{}.jpg'.format(uuid.uuid1())
                    #filename='temp.jpg'
                    #it might be worth using a stationary filename, or maybe 2 or 3 alternating names to save space
                    cv2.imwrite(filename, image)
                    count = 0
                else:
                    count = 0
        except TypeError:
            continue
        #slow down the program so the pose has to be held
        time.sleep(0.25)

cap.release()
cv2.destroyAllWindows()


#thumb to pinky: 4,8,12,16,20, so while divisible by 4, greater than 0

#for landmark in mp_hands.HandLandmark:
    #print(landmark, landmark.value)