import time
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import torch
from ultralytics import YOLOv10
import logging
import numpy as np



# Function to handle video capture and display
def process_frame(frame, model, toggle_states, class_colors):
    """Separate frame processing logic for better modularity"""
    results = model(frame)[0]
    processed_frame = frame.copy()
    
    # Track detected classes
    detected_classes = set()

    for box in results.boxes:
        cls = int(box.cls[0].cpu().numpy())
        class_name = results.names[cls]

        # More robust class name checking
        if (class_name in toggle_states and 
            toggle_states[class_name].get()):  
            try:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = box.conf[0].cpu().numpy()
                
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
    else:
        # Missing toggled objects
        missing_classes = active_toggles - detected_classes
        success_label.config(text="Missing: " + ", ".join(missing_classes), fg="red")

    return processed_frame


def show_frame():
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        logging.warning("Failed to capture frame")
        return

    frame = cv2.flip(frame, 1)
    processed_frame = process_frame(frame, model, toggle_states, class_colors)
    
    # Convert and display (similar to original code)
    frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
    end_time = time.time()
    fps = 1/np.round(end_time - start_time, 2)
    cv2.putText(frame_rgb, f'FPS:{int(fps)}',(20,70),cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0),2)
    img_pil = Image.fromarray(frame_rgb)
    img_tk = ImageTk.PhotoImage(image=img_pil)
    video_frame.img_tk = img_tk
    video_frame.configure(image=img_tk)
    video_frame.after(10, show_frame)

def setup_camera(index=0, width=640, height=480):
    """Enhanced camera setup with configuration options"""
    cap = cv2.VideoCapture(index)
    
    if not cap.isOpened():
        logging.error(f"Unable to open camera {index}")
        return None
    
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    return cap

def on_closing():
    """Improved resource management on exit"""
    logging.info("Shutting down...")
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
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

# Initialize toggles for each PPE item
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
}

# Define colors for each class
class_colors = {
    'Face': (0, 255, 0),           # Green
    'No_Gloves': (255, 0, 0),      # Blue
    'Glasses': (0, 0, 255),        # Red
    'Safety_Hat': (255, 255, 0),       # Cyan
    'Lab_Coat': (255, 165, 0),     # Orange
    'Safety_Glasses': (128, 0, 128),# Purple
    'Gloves': (0, 255, 255),       # Yellow
    'Normal_Shoes': (75, 0, 130), # Indigo
    'Mask': (75, 0, 130), # Indigo
    'Sandal': (75, 0, 130), # Indigo
}

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s: %(message)s')

# Camera configuration
CAMERA_INDEX = 0  # Allow easy configuration
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Load the YOLO model
model = YOLOv10('newbest.pt')
if model is None:
    print("Failed to load model. Exiting.")
    exit()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for ppe_item, var in toggle_states.items():
    toggle_button = ttk.Checkbutton(toggle_frame, text=ppe_item, variable=var)
    toggle_button.pack(anchor="w")

# Start video capture
cap = setup_camera(CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT)
if cap is None:
    logging.error("Camera setup failed")
    exit()

# Run the show_frame function
show_frame()

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()
