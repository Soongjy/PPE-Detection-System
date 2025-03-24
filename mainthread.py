import threading
import time
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import torch
from ultralytics import YOLOv10
import logging
import numpy as np
import pygame

# Global variables
latest_frame = None
frame_lock = threading.Lock()
cap = None  # Declare cap globally to handle it properly
last_face_detected_time = 0 # Track the last time all toggled items were detected
alert_playing = False
# Initialize pygame mixer
pygame.mixer.init()
sound = pygame.mixer.Sound("alert_sound.mp3")
            
# Threaded function to capture frames
def capture_frames():
    global latest_frame
    while True:
        ret, frame = cap.read()
        if not ret:
            logging.warning("Failed to capture frame. Stopping capture thread.")
            break

        # Flip and store the latest frame with a lock
        with frame_lock:
            latest_frame = cv2.flip(frame, 1)

# Function to handle video capture and display
def process_frame(frame, model, toggle_states, class_colors):
    global last_face_detected_time, alert_playing
    # Get frame dimensions
    height, width, _ = frame.shape

    # Calculate center and crop the middle 640x640 region
    center_x, center_y = width // 2, height // 2
    crop_x1 = max(center_x - 320, 0)
    crop_y1 = max(center_y - 320, 0)
    crop_x2 = min(center_x + 320, width)
    crop_y2 = min(center_y + 320, height)

    cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]
  
    results = model(cropped_frame)[0]
    processed_frame = frame.copy()
    
    # Track detected classes
    detected_classes = set()
    face_detected = False

    for box in results.boxes:
        cls = int(box.cls[0].cpu().numpy())
        class_name = results.names[cls]
        conf = box.conf[0].cpu().numpy()

        if conf > 0.7:
            # More robust class name checking
            if (class_name in toggle_states and toggle_states[class_name].get()):  
                try:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    
                    # Adjust coordinates to map back to the original frame
                    x1, x2 = x1 + crop_x1, x2 + crop_x1
                    y1, y2 = y1 + crop_y1, y2 + crop_y1
                    
                    color = class_colors.get(class_name, (255, 255, 255))
                    
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
                    label = f'{class_name} {conf:.2f}'
                    cv2.putText(processed_frame, label, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Add detected class to the set
                    detected_classes.add(class_name)

                    # Track face detection timing
                    if class_name == 'Face':
                        face_detected = True
                        
                except Exception as e:
                    logging.error(f"Error processing detection: {e}")

    # Perform verification after processing all detections
    active_toggles = {key for key, value in toggle_states.items() if value.get()}

    if face_detected:
        if last_face_detected_time == 0:
            last_face_detected_time = time.time()  # Start the timer

        # Show toggled items are missing
        missing_classes = active_toggles - detected_classes

        if not missing_classes:  # If all toggled items are detected
            success_label.config(text="Success", fg="green")
            last_face_detected_time = time.time()
            alert_playing = False  # Stop alert if playing

        else:  # If there are missing items
            if time.time() - last_face_detected_time > 2:
                success_label.config(text="Missing: " + ", ".join(missing_classes), fg="red")
            else:
                success_label.config(text="Detecting PPE...", fg="blue")
            # Check if 3 seconds have elapsed since face detection
            if time.time() - last_face_detected_time > 3 and not alert_playing:
                alert_playing = True
                sound.play()

    else:
        # If no face is detected, reset timer and update label
        last_face_detected_time = 0
        alert_playing = False
        success_label.config(text="Waiting for Face Detection", fg="orange")

    return processed_frame


def show_frame():
    global latest_frame

    with frame_lock:
        if latest_frame is not None:
            frame = latest_frame.copy()
        else:
            frame = None

    if frame is not None:
        start_time = time.time()
        processed_frame = process_frame(frame, model, toggle_states, class_colors)

        frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        fps = 1 / (time.time() - start_time)
        cv2.putText(
            frame_rgb,
            f'FPS: {int(fps)}',
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (255, 0, 0),
            2,
        )

        img_pil = Image.fromarray(frame_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        video_frame.img_tk = img_tk
        video_frame.configure(image=img_tk)
    else:
        video_frame.config(text="No Camera Feed", fg="red")

    video_frame.after(10, show_frame)

def setup_camera(index=0, width=640, height=480):
    global cap
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        logging.error(f"Unable to open camera {index}")
        return False

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return True

def on_closing():
    global cap
    logging.info("Shutting down...")
    if cap:
        cap.release()
    root.quit()
    root.destroy()
    

# Create the main GUI window
root = tk.Tk()
root.title("PPE Detection System")
root.geometry("800x580")
root.maxsize(800, 580)
root.minsize(800, 580)

# Frame for video display
video_frame = tk.Label(root)
video_frame.grid(row=0, column=0, padx=10, pady=10)

# Toggle switches for PPE items
toggle_frame = tk.Frame(root)
toggle_frame.grid(row=0, column=1, padx=10, pady=10, sticky="n")

# Label to display success or missing items
success_label = tk.Label(root, text="", font=("Arial", 10))
success_label.grid(row=1, column=0, columnspan=2, pady=8)

# Toggle states and colors (unchanged)
toggle_states = {
    'Face': tk.BooleanVar(value=True),
    'No_Gloves': tk.BooleanVar(value=True),
    'Glasses': tk.BooleanVar(value=True),
    'Safety_Hat': tk.BooleanVar(value=False),
    'Lab_Coat': tk.BooleanVar(value=True),
    'Safety_Glasses': tk.BooleanVar(value=False),
    'Gloves': tk.BooleanVar(value=False),
    'Normal_Shoes': tk.BooleanVar(value=False),
    'Mask': tk.BooleanVar(value=False),
    'Sandal': tk.BooleanVar(value=False),
    'Earmuffs' : tk.BooleanVar(value=False),
}

class_colors = {
    'Face': (0, 255, 0),           # Green
    'No_Gloves': (255, 0, 0),      # Blue
    'Glasses': (0, 0, 255),        # Red
    'Safety_Hat': (255, 255, 0),   # Cyan
    'Lab_Coat': (255, 165, 0),     # Orange
    'Safety_Glasses': (128, 0, 128),# Purple
    'Gloves': (0, 255, 255),       # Yellow
    'Normal_Shoes': (75, 0, 130),  # Indigo
    'Mask': (75, 0, 130),          # Indigo
    'Sandal': (75, 0, 130),        # Indigo
    'Earmuffs': (0, 255, 255),
}

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s: %(message)s')

if not setup_camera():
    logging.error("Camera setup failed")
    exit()


# Load the YOLO model
model = YOLOv10('last.pt')
if model is None:
    print("Failed to load model. Exiting.")
    exit()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for ppe_item, var in toggle_states.items():
    toggle_button = ttk.Checkbutton(toggle_frame, text=ppe_item, variable=var)
    toggle_button.pack(anchor="w")

# Start video capture
capture_thread = threading.Thread(target=capture_frames, daemon=True)
capture_thread.start()

# Run the show_frame function
show_frame()

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()
