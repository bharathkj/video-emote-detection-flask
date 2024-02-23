from flask import Flask, request, jsonify
from emotion_detector import EmotionDetector
import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firebase with your credentials
cred = credentials.Certificate(r'C:\Users\bhara\Downloads\Facial_Detection_Project-main\Facial_Detection_Project-main\serviceAccountKey.json')
firebase_admin.initialize_app(cred)

app = Flask(__name__)
emotion_detector = EmotionDetector()

# Define a route to receive image frames
@app.route('/process_frame', methods=['POST'])
def process_frame():
    image_frame = request.files['image_frame']

    # Use the EmotionDetector to process the frame
    result = emotion_detector.process_frame(image_frame)
    user_name = result['user_name']
    emotion = result['emotion']


    # Reference to the "users" collection
    users_ref = firestore.client().collection("users")
    # Reference to the specific user document
    user_doc_ref = users_ref.document(user_name)
    # Reference to the "emotions" subcollection within the user document
    emotions_ref = user_doc_ref.collection("dummyemoteflask")
    # Emotion data to be stored
    emotion_data = {"emotion": emotion, "timestamp": firestore.SERVER_TIMESTAMP}
    # Add emotion data to the "emotions" subcollection
    emotions_ref.add(emotion_data)

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True,port=5001, host='0.0.0.0')
