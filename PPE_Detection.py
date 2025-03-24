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

latest_frame = None
frame_lock = threading.Lock()
last_success_time = 0 # Track the last time all toggled items were detected
alert_playing = False

# Initialize pygame mixer
pygame.mixer.init()
sound = pygame.mixer.Sound("alert_sound.mp3")

# Capture frames in a separate thread
def capture_frames():
    global latest_frame
    global running
    while True:
        ret, frame = cap.read()
        if not ret:
            running = False
            break

        # Update the shared frame with the latest one
        with frame_lock:
            latest_frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            latest_frame = cv2.resize(latest_frame, (380, 640), interpolation=cv2.INTER_AREA)

def process_frame(frame, model, toggle_states, class_colors):
    global last_success_time, alert_playing
    results = model(frame)[0]
    processed_frame = frame.copy()
    
    # Track detected classes
    detected_classes = set()

    for box in results.boxes:
        cls = int(box.cls[0].cpu().numpy())
        class_name = results.names[cls]
        conf = box.conf[0].cpu().numpy()

        if conf > 0.5:
            # More robust class name checking
            if (class_name in toggle_states and toggle_states[class_name].get()):  
                try:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    
                    color = class_colors.get(class_name, (255, 255, 255))
                    
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
                    label = f'{class_name} {conf:.2f}'
                    cv2.putText(processed_frame, label, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Add detected class to the set
                    detected_classes.add(class_name)
                except Exception as e:
                    logging.error(f"Error processing detection: {e}")

    # Perform verification after processing all detections
    active_toggles = {key for key, value in toggle_states.items() if value.get()}

    if active_toggles.issubset(detected_classes):
        # All toggled objects are detected, show "Success" message
        success_label.config(text="Success", fg="green")
        last_success_time = time.time()  # Reset the timer
        alert_playing = False  # Reset the alert flag
    else:
        # Missing toggled objects
        missing_classes = active_toggles - detected_classes
        success_label.config(text="Missing: " + ", ".join(missing_classes), fg="red")
        if time.time() - last_success_time > 3 and not alert_playing and last_success_time!=0:
            alert_playing = True  # Set alert flag to avoid multiple plays
            sound.play()  # Play alert sound non-blocking
            print("alert")
            print("alert")
            print("alert")
            #root.bell()
    return processed_frame

def show_frame():
    global latest_frame
    global running

    # Check if the application is still running
    if not running:
        logging.info("Stopping video stream as the camera feed is unavailable.")
        success_label.config(text="Camera feed stopped", fg="red")
        return  # Stop updating frames
    
    start_time = time.time()
    
    # Access the most recent frame
    with frame_lock:
        frame = latest_frame.copy() if latest_frame is not None else None

    if frame is not None:
        frame = cv2.flip(frame, 1)
        # Process the frame
        processed_frame = process_frame(frame, model, toggle_states, class_colors)
        
        # Convert to RGB for display
        frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
    
        # Calculate and display FPS
        end_time = time.time()
        fps = 1/np.round(end_time - start_time, 2)
        cv2.putText(frame_rgb, f'FPS:{int(fps)}',(20,70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)
        
        # Convert to Tkinter-compatible image
        img_pil = Image.fromarray(frame_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        video_frame.img_tk = img_tk
        video_frame.configure(image=img_tk)
    
    # Schedule next frame
    video_frame.after(10, show_frame)

def setup_camera():
    """Setup IP camera for vertical orientation"""
    # Use IP Webcam URL from phone
    cap = cv2.VideoCapture("http://192.168.215.64:8080/video")
    
    if not cap.isOpened():
        logging.error("Unable to open camera")
        return None
    
    # # Log camera properties
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # logging.info(f"Camera resolution: {width}x{height}")

    return cap

def on_closing():
    """Clean up resources on exit"""
    logging.info("Shutting down...")
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    root.quit()
    root.destroy()

# Create vertical GUI window
root = tk.Tk()
root.title("Vertical PPE Detection")
root.geometry("700x500")  # Adjusted for vertical orientation

root.minsize(600, 700)


# Main container frames
main_frame = tk.Frame(root)
main_frame.pack(fill="both", expand=True)

video_frame_container = tk.Frame(main_frame)
video_frame_container.pack(side="left", fill="both",expand=True, padx=10, pady=10)

toggle_frame_container = tk.Frame(main_frame,width=200)
toggle_frame_container.pack(side="right", fill="y", padx=10, pady=10)

# Video frame (left)
video_frame = tk.Label(video_frame_container, bg="black")
video_frame.pack(fill="both", expand=True, padx=10, pady=10)

# Toggle frame (right)
toggle_frame = tk.Frame(toggle_frame_container)
toggle_frame.pack(fill="y", padx=10, pady=10)

# Success label (bottom)
success_label = tk.Label(root, text="", font=("Arial", 10), pady=3)
success_label.pack(side="bottom", fill="x")

# Toggle states and colors (unchanged)
toggle_states = {
    'Face': tk.BooleanVar(value=True),
    'No_Gloves': tk.BooleanVar(value=True),
    'Glasses': tk.BooleanVar(value=True),
    'Safety_Hat': tk.BooleanVar(value=True),
    'Lab_Coat': tk.BooleanVar(value=True),
    'Safety_Glasses': tk.BooleanVar(value=True),
    'Gloves': tk.BooleanVar(value=True),
    'Normal_Shoes': tk.BooleanVar(value=True),
    'Mask': tk.BooleanVar(value=True),
    'Sandal': tk.BooleanVar(value=True),
    'Earmuffs' : tk.BooleanVar(value=True),
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

# Logging setup
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s: %(message)s')

# Load YOLO model
model = YOLOv10('newbest.pt')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Create toggle buttons
for ppe_item, var in toggle_states.items():
    toggle_button = ttk.Checkbutton(toggle_frame, text=ppe_item, variable=var)
    toggle_button.pack(anchor="w",pady=2)

# Start camera capture
cap = setup_camera()
if cap is None:
    logging.error("Camera setup failed")
    exit()

running = True
capture_thread = threading.Thread(target=capture_frames, daemon=True)
capture_thread.start()

# Start video stream
show_frame()

# Setup closing protocol
root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()