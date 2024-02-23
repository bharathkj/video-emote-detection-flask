import requests
import json
from flask import jsonify

# Flask server URL
flask_url = 'http://127.0.0.1:5000/process_frame'  # Update with your actual Flask server URL

# image data to send for analysis
image_frame_file = r"C:\Users\bhara\Downloads\Facial_Detection_Project-main\Facial_Detection_Project-main\face_recog\BharathkumarJ_Photocopy.jpeg"
# Prepare the request payload
image_frame = {'image_frame': open(image_frame_file, 'rb')}

# Send the HTTP POST request
try:
    response = requests.post(flask_url, files=image_frame)

    if response.status_code == 200:
        # Successfully received sentiment analysis results
        data = response.json()
        print('Sentiment Analysis Results:', data)
    else:
        # Handle error
        print('Error:', response.status_code)

except requests.exceptions.RequestException as e:
    # Handle network errors
    print('Error:', e)



#Invoke-RestMethod -Uri "http://127.0.0.1:5000/process_frame" -Method Post -InFile "C:\Users\bhara\Downloads\Facial_Detection_Project-main\Facial_Detection_Project-main\face_recog\bharath.png" -ContentType "multipart/form-data"
