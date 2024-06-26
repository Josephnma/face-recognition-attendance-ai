# import base64
# import os
# import cv2
# import numpy as np
# import pandas as pd
# from flask import Flask, request, jsonify, render_template, send_from_directory
# from flask_cors import CORS
# import pyttsx3
#
# app = Flask(__name__)
# CORS(app)
#
# # Define the directory for student images and the Excel file path
# STUDENT_IMAGE_FOLDER = 'student_images'
# STUDENTS_FILE_PATH = 'students.csv'  # Replace with your actual file path
#
# # Ensure the static directory exists
# STATIC_DIR = 'static'
# os.makedirs(STATIC_DIR, exist_ok=True)
#
# # Load OpenCV's pre-trained models for face detection
# face_detector_model = 'deploy.prototxt'  # Path to the face detection model's prototxt file
# face_detector_weights = 'res10_300x300_ssd_iter_140000_fp16.caffemodel'  # Path to the face detection model's weights file
#
#
# # Function to load student names from the Excel/CSV file
# def load_student_names(file_path):
#     file_ext = os.path.splitext(file_path)[1].lower()
#     if file_ext == '.xlsx':
#         df = pd.read_excel(file_path)
#     elif file_ext == '.csv':
#         df = pd.read_csv(file_path)
#     else:
#         raise ValueError("Unsupported file format. Please use .xlsx or .csv")
#     return df
#
#
# # Function to update attendance in the Excel file
# def update_attendance(file_path, recognized_name):
#     df = load_student_names(file_path)
#     if recognized_name in df['Names'].values:
#         df.loc[df['Names'] == recognized_name, 'Attendance'] = 1
#     else:
#         print(f"Student {recognized_name} not found in the file.")
#     df.to_excel(file_path, index=False) if file_path.endswith('.xlsx') else df.to_csv(file_path, index=False)
#
#
# # Function to find the closest name match based on cosine similarity
# def find_closest_name(face_encoding, known_face_encodings, student_names):
#     if known_face_encodings:
#         # Calculate cosine similarities
#         similarities = [np.dot(face_encoding, enc) / (np.linalg.norm(face_encoding) * np.linalg.norm(enc)) for enc in
#                         known_face_encodings]
#         best_match_index = np.argmax(similarities)
#         return student_names[best_match_index] if similarities[best_match_index] > 0.5 else "Unknown"
#     return "Unknown"
#
#
# # Function to get student image paths from a folder
# def get_student_image_paths(folder_path):
#     image_paths = []
#     for filename in os.listdir(folder_path):
#         if filename.endswith(".jpg") or filename.endswith(".png"):
#             image_paths.append(os.path.join(folder_path, filename))
#     return image_paths
#
#
# # Function to load known faces
# def load_known_faces():
#     known_face_encodings = []
#     student_names = []
#     for image_path in get_student_image_paths(STUDENT_IMAGE_FOLDER):
#         img = cv2.imread(image_path)
#         encoding = get_face_encoding(img)
#         if encoding is not None:
#             known_face_encodings.append(encoding)
#             student_names.append(os.path.splitext(os.path.basename(image_path))[0])
#     return known_face_encodings, student_names
#
#
# # Function to get face encoding using OpenCV's DNN module
# def get_face_encoding(image):
#     # Create a 4D blob from the image
#     blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)
#     net = cv2.dnn.readNetFromCaffe(face_detector_model, face_detector_weights)
#     net.setInput(blob)
#     detections = net.forward()
#
#     # Assume the largest detected face is the one we want
#     h, w = image.shape[:2]
#     for i in range(detections.shape[2]):
#         confidence = detections[0, 0, i, 2]
#         if confidence > 0.5:
#             box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#             (startX, startY, endX, endY) = box.astype("int")
#             face = image[startY:endY, startX:endX]
#             if face.size == 0:
#                 continue
#             face_blob = cv2.dnn.blobFromImage(face, 1.0, (300, 300), [104, 117, 123], False, False)
#             net.setInput(face_blob)
#             face_encoding = net.forward()
#             return face_encoding.flatten()
#     return None
#
#
# # Function to use text-to-speech to speak a message
# def speak(message):
#     try:
#         # Initialize the pyttsx3 engine
#         engine = pyttsx3.init()
#         engine.say(message)
#         engine.runAndWait()
#     except Exception as e:
#         print(f"Error in TTS: {e}")
#
#
# # Function to recognize student from the captured image
# def recognize_student(captured_img_path):
#     captured_img = cv2.imread(captured_img_path)
#     face_encoding = get_face_encoding(captured_img)
#
#     if face_encoding is None:
#         return None, None
#
#     known_face_encodings, student_names = load_known_faces()
#     recognized_name = find_closest_name(face_encoding, known_face_encodings, student_names)
#     if recognized_name != "Unknown":
#         student_img_path = os.path.join('static', os.path.relpath(captured_img_path, '.'))
#         update_attendance(STUDENTS_FILE_PATH, recognized_name)
#         speak(f"Welcome {recognized_name}, please have a seat.")
#         return recognized_name, student_img_path
#
#     return None, None
#
#
# # Route for the main page
# @app.route('/')
# def index():
#     return render_template('index1.html')
#
#
# # Route to capture image and recognize student
# @app.route('/capture', methods=['POST'])
# def capture():
#     img_data = request.form['image']
#     img_data = base64.b64decode(img_data.split(',')[1])
#     img_path = os.path.join(STATIC_DIR, 'captured_image.jpg')
#
#     with open(img_path, 'wb') as f:
#         f.write(img_data)
#
#     student_name, student_img_path = recognize_student(img_path)
#     if student_name:
#         return jsonify({
#             'student_name': student_name,
#             'student_img': f'/{student_img_path}',
#             'success': True
#         })
#     else:
#         return jsonify({'student_name': None, 'student_img': None, 'success': False})
#
#
# # Route to serve static files
# @app.route('/static/<path:filename>')
# def static_files(filename):
#     return send_from_directory(STATIC_DIR, filename)
#
#
# # Route to display attendance sheet
# @app.route('/attendance')
# def attendance():
#     df = load_student_names(STUDENTS_FILE_PATH)
#     return df.to_html()
#
#
# if __name__ == '__main__':
#     app.run(debug=True)
