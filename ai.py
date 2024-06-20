import cv2
import numpy as np
import os
import pandas as pd  # For reading and writing Excel/CSV files
from difflib import SequenceMatcher
import pyttsx3  # For text-to-speech


# Function to pre-process an image for comparison
def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for simpler comparison
    img = cv2.resize(img, (224, 224))  # Resize for efficiency (adjust if needed)
    return img


# Function to capture image from webcam, compare, and recognize student
def capture_compare_recognize(student_image_folder, students_file_path):
    # Open webcam video capture
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Display webcam feed
        cv2.imshow('Webcam Capture', frame)

        # Press 'c' to capture an image
        key = cv2.waitKey(1)
        if key % 256 == 27:  # ESC pressed
            print("Escape hit, closing...")
            break
        elif key % 256 == ord('c'):
            # Capture image, close webcam, and save image
            captured_image_path = 'captured_image.jpg'
            cv2.imwrite(captured_image_path, frame)
            cap.release()
            cv2.destroyAllWindows()

            # Preprocess captured image
            captured_img = cv2.imread(captured_image_path)
            preprocessed_captured_img = preprocess_image(captured_img)

            # Load student names from the Excel file
            student_names = load_student_names(students_file_path)

            # Loop through student images and compare
            for image_path in get_student_image_paths(student_image_folder):
                student_img = cv2.imread(image_path)
                preprocessed_student_img = preprocess_image(student_img)

                # Use template matching for comparison
                result = cv2.matchTemplate(preprocessed_captured_img, preprocessed_student_img, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

                # Set a threshold for considering a match
                threshold = 0.8  # You can experiment with this value

                if max_val >= threshold:
                    # Extract the filename (without extension)
                    filename = os.path.splitext(os.path.basename(image_path))[0]

                    # Find the matching student name
                    recognized_name = find_closest_name(filename, student_names)

                    print(f"Match found! Student recognized: {recognized_name}")

                    # Update attendance in the Excel file
                    update_attendance(students_file_path, recognized_name)

                    # Use text-to-speech to welcome the student
                    speak(f"Welcome {recognized_name}, please have a seat.")
                    break  # Exit loop after finding a match

            else:
                print("No match found for any student.")

            break
# Function to get student image paths from a folder
def get_student_image_paths(folder_path):
    image_paths = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_paths.append(os.path.join(folder_path, filename))
    return image_paths


# Function to find the closest name match
def find_closest_name(filename, student_names):
    highest_similarity = 0
    best_match = "Unknown"
    for name in student_names:
        # Use sequence matcher to find the similarity
        similarity = SequenceMatcher(None, filename, name).ratio()
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = name
    return best_match


# Function to load student names from Excel/CSV file
def load_student_names(file_path):
    # Determine file type
    file_ext = os.path.splitext(file_path)[1].lower()

    if file_ext == '.xlsx':
        try:
            df = pd.read_excel(file_path)
        except ImportError:
            raise ImportError("Missing optional dependency 'openpyxl'. Use pip or conda to install openpyxl.")
    elif file_ext == '.csv':
        df = pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file format. Please use .xlsx or .csv")

    return df['Names'].tolist()


# Function to update attendance in the Excel file
def update_attendance(file_path, recognized_name):
    # Determine file type
    file_ext = os.path.splitext(file_path)[1].lower()

    if file_ext == '.xlsx':
        try:
            df = pd.read_excel(file_path)
        except ImportError:
            raise ImportError("Missing optional dependency 'openpyxl'. Use pip or conda to install openpyxl.")
    elif file_ext == '.csv':
        df = pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file format. Please use .xlsx or .csv")

    # Update the attendance for the recognized student
    if recognized_name in df['Names'].values:
        df.loc[df['Names'] == recognized_name, 'Attendance'] = 1
    else:
        print(f"Student {recognized_name} not found in the file.")

    # Save the updated DataFrame back to the file
    if file_ext == '.xlsx':
        df.to_excel(file_path, index=False)
    elif file_ext == '.csv':
        df.to_csv(file_path, index=False)


# Function to use text-to-speech to speak a message
def speak(message):
    engine = pyttsx3.init()
    engine.say(message)
    engine.runAndWait()


# Replace these with your actual paths and names
student_image_folder = 'student_images'  # Folder containing student images
students_file_path = 'students.xlsx'  # Path to Excel or CSV file containing student names

if __name__ == '__main__':
    capture_compare_recognize(student_image_folder, students_file_path)
