import os
import cv2
import pandas as pd

# Define paths
STUDENT_IMAGE_FOLDER = 'student_images'
STUDENTS_FILE_PATH = 'students.csv'  # Replace with your actual file path

# Function to ensure the directory exists
os.makedirs(STUDENT_IMAGE_FOLDER, exist_ok=True)


# Function to add student to the CSV file
def add_student_to_csv(student_name):
    # Check if the file exists
    if not os.path.exists(STUDENTS_FILE_PATH):
        # Create a new DataFrame if the file does not exist
        df = pd.DataFrame(columns=['Names', 'Attendance'])
    else:
        # Load existing data
        df = pd.read_csv(STUDENTS_FILE_PATH)

    # Append the new student
    new_student = pd.DataFrame({'Names': [student_name], 'Attendance': [0]})
    df = pd.concat([df, new_student], ignore_index=True)

    # Save the updated DataFrame back to the CSV file
    df.to_csv(STUDENTS_FILE_PATH, index=False)


# Function to save student's image
def save_student_image(student_name, img_path):
    student_img_path = os.path.join(STUDENT_IMAGE_FOLDER, f"{student_name}.jpg")
    img = cv2.imread(img_path)
    if img is not None:
        cv2.imwrite(student_img_path, img)
    else:
        print("Error: Could not read the image.")


# Example usage
def register_student(student_name, img_path):
    add_student_to_csv(student_name)
    save_student_image(student_name, img_path)
    print(f"Registered {student_name} successfully!")

# Uncomment below lines to test this script directly
# register_student("John Doe", "path_to_image.jpg")

# def register_student(student_name, image_filename):
#     image_folder = 'student_images'
#     image_path = os.path.join(image_folder, image_filename)
#
#     # Save student details to CSV
#     add_student_to_csv(student_name)
