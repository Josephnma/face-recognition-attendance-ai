import base64
import os
import cv2
import numpy as np
import pandas as pd
from difflib import SequenceMatcher
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import pyttsx3

app = Flask(__name__)
CORS(app)

# Define the directory for student images and the Excel file path
STUDENT_IMAGE_FOLDER = 'student_images'
STUDENTS_FILE_PATH = 'students.xlsx'  # Replace with your actual file path

# Ensure the static directory exists
STATIC_DIR = 'static'
os.makedirs(STATIC_DIR, exist_ok=True)


# Function to preprocess an image for comparison
def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (224, 224))
    return img


# Function to load student names from the Excel/CSV file
def load_student_names(file_path):
    file_ext = os.path.splitext(file_path)[1].lower()
    if file_ext == '.xlsx':
        df = pd.read_excel(file_path)
    elif file_ext == '.csv':
        df = pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file format. Please use .xlsx or .csv")
    return df


# Function to update attendance in the Excel file
def update_attendance(file_path, recognized_name):
    df = load_student_names(file_path)
    if recognized_name in df['Names'].values:
        df.loc[df['Names'] == recognized_name, 'Attendance'] = 1
    else:
        print(f"Student {recognized_name} not found in the file.")
    df.to_excel(file_path, index=False) if file_path.endswith('.xlsx') else df.to_csv(file_path, index=False)


# Function to find the closest name match
def find_closest_name(filename, student_names):
    highest_similarity = 0
    best_match = "Unknown"
    for name in student_names:
        similarity = SequenceMatcher(None, filename, name).ratio()
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = name
    return best_match


# Function to get student image paths from a folder
def get_student_image_paths(folder_path):
    image_paths = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_paths.append(os.path.join(folder_path, filename))
    return image_paths


# Function to use text-to-speech to speak a message
def speak(message):
    try:
        # Initialize the pyttsx3 engine
        engine = pyttsx3.init()
        engine.say(message)
        engine.runAndWait()
    except Exception as e:
        print(f"Error in TTS: {e}")


# Function to recognize student from the captured image
def recognize_student(captured_img_path):
    # Load the face cascade classifier
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    captured_img = cv2.imread(captured_img_path)
    gray = cv2.cvtColor(captured_img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Check if any faces are detected
    if len(faces) > 0:
        # Extract the face region
        (x, y, w, h) = faces[0]
        face_img = captured_img[y:y + h, x:x + w]

        # Preprocess the face image
        preprocessed_captured_img = preprocess_image(face_img)

        student_found = None
        student_img_path = None

        df = load_student_names(STUDENTS_FILE_PATH)
        student_names = df['Names'].tolist()

        for image_path in get_student_image_paths(STUDENT_IMAGE_FOLDER):
            student_img = cv2.imread(image_path)
            preprocessed_student_img = preprocess_image(student_img)
            result = cv2.matchTemplate(preprocessed_captured_img, preprocessed_student_img, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)

            if max_val >= 0.3:  # Threshold
                filename = os.path.splitext(os.path.basename(image_path))[0]
                recognized_name = find_closest_name(filename, student_names)
                student_found = recognized_name
                # Ensure the path for the frontend includes 'static'
                student_img_path = os.path.join('static', os.path.relpath(image_path, STUDENT_IMAGE_FOLDER))
                update_attendance(STUDENTS_FILE_PATH, recognized_name)
                speak(f"Welcome {recognized_name}, please have a seat.")  # Speak the welcome message
                break

        return student_found, student_img_path

    else:
        print("No face detected in the image.")
        speak("No face detected in the camera.")
        student_found = None
        student_img_path = None
        return student_found, student_img_path

    # Route for the main page


@app.route('/')
def index():
    return render_template('index1.html')  # Assuming the template name


# Route to capture image and recognize student
@app.route('/capture', methods=['POST'])
def capture():
    img_data = request.form['image']
    img_data = base64.b64decode(img_data.split(',')[1])
    img_path = os.path.join(STATIC_DIR, 'captured_image.jpg')

    with open(img_path, 'wb') as f:
        f.write(img_data)

    student_name, student_img_path = recognize_student(img_path)
    if student_name:
        return jsonify({
            'student_name': student_name,
            'student_img': f'/{student_img_path}',
            'success': True
        })
    else:
        return jsonify({'student_name': None, 'student_img': None, 'success': False})


# Route to serve static files
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(STATIC_DIR, filename)


# Route to display attendance sheet
@app.route('/attendance')
def attendance():
    df = load_student_names(STUDENTS_FILE_PATH)
    return df.to_html()


if __name__ == '__main__':
    app.run(debug=True)
