import cv2
import numpy as np
from keras.models import model_from_json
import face_recognition

# Declaring Classes
emotion_classes = {
    0: "Angry", 
    1: "Disgust", 
    2: "Fear", 
    3: "Happy", 
    4: "Neutral", 
    5: "Sad", 
    6: "Surprise"}

# Loading Trained Model:
json_file = open(r"model/model_v2.json", 'r')

# Loading model.json file into model
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# Loading Weights:
model.load_weights(r"model/new_model_v2.h5")

print("Model lodded scussesfully")

# Loading Face Cascade 
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


class EmotionDetector:
    def __init__(self):
        self.known_faces_encodings = []
        self.known_faces_names = []

        # Load known faces and their encodings
        image_of_person1 = face_recognition.load_image_file(r"C:\Users\bhara\Downloads\Facial_Detection_Project-main\Facial_Detection_Project-main\face_recog\ajith.png")
        encoding_of_person1 = face_recognition.face_encodings(image_of_person1)[0]
        image_of_person2 = face_recognition.load_image_file(r"C:\Users\bhara\Downloads\Facial_Detection_Project-main\Facial_Detection_Project-main\face_recog\bharath.png")
        encoding_of_person2 = face_recognition.face_encodings(image_of_person2)[0]

        self.known_faces_encodings = [encoding_of_person1, encoding_of_person2]
        self.known_faces_names = ["ajith", "bharath"]

    def process_frame(self, image_frame):
        # Decode the image frame and convert it to RGB
        rgb_img = cv2.imdecode(np.frombuffer(image_frame.read(), np.uint8), -1)
        # Check the number of channels in the image
        if len(rgb_img.shape) == 2:  # Grayscale image
            # Convert grayscale image to RGB
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_GRAY2RGB)

        # Run facial recognition
        facial_result = self.run_facial_recognition(rgb_img)

        # Run emotion detection
        emotion_result = self.run_emotion_detection(rgb_img)

        # Return the results
        return {'user_name': facial_result, 'emotion': emotion_result}

    def run_facial_recognition(self, rgb_img):
        # Run your facial recognition logic here using face_recognition library
        # Find all face locations and face encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_img)
        face_encodings = face_recognition.face_encodings(rgb_img, face_locations)

        # Loop through each face found in the current frame
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Check if the face matches any known faces
            matches = face_recognition.compare_faces(self.known_faces_encodings, face_encoding)

            username = "Unknown"
            print("checking if name prints b4hand",username)

            # If a match is found, use the name of the known face
            if True in matches:
                first_match_index = matches.index(True)
                username = self.known_faces_names[first_match_index]
                print(username)

        return(username)

    def run_emotion_detection(self, rgb_img):
        # Detect faces available on the camera:
        num_face = face_detector.detectMultiScale(rgb_img, scaleFactor=1.3, minNeighbors=5)

        # Take each face available on the camera and preprocess it:
        for (x, y, w, h) in num_face:
            cv2.rectangle(rgb_img, (x, y - 50), (x + w, y + h + 10), (0, 255, 0), 4)
            roi_gray_frame = rgb_img[y:y + h, x: x + w]

            # Modify this part of the code to convert the RGB image to grayscale
            roi_gray_frame = cv2.cvtColor(roi_gray_frame, cv2.COLOR_BGR2GRAY)

            # Modify this part to prepare the input for emotion detection model
            cropped_img = cv2.resize(roi_gray_frame, (48, 48), -1)
            cropped_img = np.expand_dims(cropped_img, 0)
            cropped_img = np.expand_dims(cropped_img, -1)  # Add the channel dimension

            # Predict the emotion:
            if np.sum([roi_gray_frame]) != 0:
                emotion_prediction = model.predict(cropped_img)
                maxindex = int(np.argmax(emotion_prediction))
                emotion_result = str(emotion_classes[maxindex])
                print(emotion_result)
            else:
                emotion_result = 'null'
                print('no faces')

        return emotion_result
