import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageFilter
from ultralytics import YOLO
import numpy as np
import face_recognition

# Create a Tkinter GUI window
window = tk.Tk()
window.title("Image Blurring")

# Set the window size
window.geometry("600x400")

# Load the YOLO model
model = YOLO('model.pt')

# Create a Tkinter variable to store the selected object(s)
selected_objects = tk.StringVar(value="Face,License Plate")


# Global variables
selected_faces = []  # List to store the selected face locations

# Mouse callback function
def mouse_callback(event, x, y, flags, param):
    global selected_faces
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if the clicked location is within any of the face locations
        for (top, right, bottom, left) in selected_faces:
            if left <= x <= right and top <= y <= bottom:
                if (top, right, bottom, left) in selected_faces:
                    selected_faces.remove((top, right, bottom, left))
                else:
                    selected_faces.append((top, right, bottom, left))
                break
# Function to blur the selected object(s)
def blur_selected_objects(image, selected_objects):
    if "Face" in selected_objects or "Both" in selected_objects:
        face_locations = face_recognition.face_locations(np.array(image))
        for (top, right, bottom, left) in face_locations:
            face_image = image.crop((left, top, right, bottom))
            blurred_face = face_image.filter(ImageFilter.GaussianBlur(radius=30))
            image.paste(blurred_face, (left, top, right, bottom))

    if "License Plate" in selected_objects or "Both" in selected_objects:
        image_array = np.array(image)
        results = model.predict(image_array, save=True)
        for bbox in results[0].boxes.data[:, :4]:
            x, y, w, h = bbox.cpu().numpy()
            roi = image_array[int(y):int(h), int(x):int(w)]
            blurred_roi = cv2.GaussianBlur(roi, (51, 51), 0)
            image_array[int(y):int(h), int(x):int(w)] = blurred_roi

        image = Image.fromarray(image_array)

    return image

# Function to open and process the image
def process_image():
    # Open a file dialog to select an image
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])

    # Load the image using PIL
    image = Image.open(file_path)

    # Get the selected objects
    selected_objects_list = selected_objects.get().split(",")

    # Blur the selected objects
    modified_image = blur_selected_objects(image, selected_objects_list)

    # Save and display the blurred image
    modified_image.save('image generated.png')

    modified_image = Image.open("image generated.png")
    modified_image.show()

# Function to open and process the live video stream
def process_live_stream():
    #cv2.namedWindow("Blurred Faces")

    # Open the camera
    video_capture = cv2.VideoCapture(0)  # 0 represents the default camera

    # Create a video writer to save the processed video
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    video_writer = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))
    # Create a window and set the mouse callback
    #cv2.namedWindow("Blurred Faces")
    #cv2.setMouseCallback("Blurred Faces", mouse_callback)
    while True:
        # Read a frame from the camera
        ret, frame = video_capture.read()
        if not ret:
            break

        # Convert the frame to PIL Image
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Get the selected objects
        selected_objects_list = selected_objects.get().split(",")

        # Blur the selected objects
        #cv2.setMouseCallback("Processed Frame", mouse_callback)
        modified_image = blur_selected_objects(image, selected_objects_list)

        # Convert the modified image back to OpenCV format
        modified_frame = cv2.cvtColor(np.array(modified_image), cv2.COLOR_RGB2BGR)

        # Write the processed frame to the video writer
        video_writer.write(modified_frame)
        # Display the processed frame
        cv2.imshow("Blurred Faces", modified_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and video writer
    video_capture.release()
    video_writer.release()
    cv2.destroyAllWindows()

# Function to open and process the video stream
def process_video_stream():
    
    # Open a file dialog to select a video file
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4")])

    # Open the video file using OpenCV
    video_capture = cv2.VideoCapture(file_path)

    # Create a video writer to save the processed video
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    video_writer = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))

    while True:
        # Read a frame from the video
        ret, frame = video_capture.read()
        if not ret:
            break

        # Convert the frame to PIL Image
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Get the selected objects
        selected_objects_list = selected_objects.get().split(",")

        # Blur the selected objects
        modified_image = blur_selected_objects(image, selected_objects_list)

        # Convert the modified image back to OpenCV format
        modified_frame = cv2.cvtColor(np.array(modified_image), cv2.COLOR_RGB2BGR)

        # Write the processed frame to the video writer
        video_writer.write(modified_frame)
        
        # Display the processed frame
        cv2.imshow("Processed Frame", modified_frame)
        cv2.setMouseCallback("Processed Frame", mouse_callback)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and video writer
    video_capture.release()
    video_writer.release()
    cv2.destroyAllWindows()

# Button to open and process an image
image_button = tk.Button(window, text="Open Image", command=process_image, font=("Arial", 12), padx=20, pady=10)
image_button.pack()

# Checkbox for Face
face_checkbox = tk.Checkbutton(window, text="Face", variable=selected_objects, onvalue="Face", font=("Arial", 12))

# Checkbox for License Plate
license_plate_checkbox = tk.Checkbutton(window, text="License Plate", variable=selected_objects, onvalue="License Plate", font=("Arial", 12))

# Checkbox for Both
both_checkbox = tk.Checkbutton(window, text="Both", variable=selected_objects, onvalue="Both", font=("Arial", 12))

face_checkbox.pack()
license_plate_checkbox.pack()
both_checkbox.pack()

# Button to open and process a live stream
video_button = tk.Button(window, text="Open Live Stream", command=process_live_stream, font=("Arial", 12), padx=20, pady=10)
video_button.pack()

# Button to open and process a video stream
video_button = tk.Button(window, text="Open Video Stream", command=process_video_stream, font=("Arial", 12), padx=20, pady=10)
video_button.pack()

# Start the Tkinter event loop
window.mainloop()
