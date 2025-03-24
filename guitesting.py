import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import torch
from ultralytics import YOLOv10

# Load the YOLO model
model = YOLOv10('new10n.pt')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Create the main GUI window
root = tk.Tk()
root.title("PPE Detection System")
root.geometry("900x510")

# Initialize toggles for each PPE item
toggle_states = {
    'Face': tk.BooleanVar(value=True),
    'No_Gloves': tk.BooleanVar(value=True),
    'Glasses': tk.BooleanVar(value=True),
    'Helmet': tk.BooleanVar(value=True),
    'Lab Coat': tk.BooleanVar(value=True),
    'Goggles': tk.BooleanVar(value=True),
    'Gloves': tk.BooleanVar(value=True),
    'Covered Shoes': tk.BooleanVar(value=True)
}

# Frame for video display
video_frame = tk.Label(root)
video_frame.grid(row=0, column=0, padx=10, pady=10)


# Frame for toggle buttons and messages
controls_frame = tk.Frame(root)
controls_frame.grid(row=0, column=1, padx=10, pady=10, sticky="n")

# Function to handle video capture and display
def show_frame():
    ret, frame = cap.read()
    if not ret:
        return

    frame = cv2.flip(frame, 1)  # Mirror the camera
    results = model(frame)[0]
    
    # Maintain state of detected toggled objects
    detected_classes = set()

    # Apply toggles to selectively display detections
    for box in results.boxes:
        cls = int(box.cls[0].cpu().numpy())  # Class label index

        # Ensure the class exists in the model's class names
        if cls >= len(results.names):
            print(f"Warning: Unknown class detected with index {cls}. Skipping.")
            continue

        class_name = results.names[cls]

        # Ensure the class name exists in toggle_states (user-defined classes)
        if class_name not in toggle_states:
            print(f"Warning: Detected class '{class_name}' not in toggle_states. Skipping.")
            continue

        # Check if the toggle is active for the class
        if toggle_states[class_name].get():
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = box.conf[0].cpu().numpy()
            label = f'{class_name} {conf:.2f}'

            # Mark this class as detected
            detected_classes.add(class_name)

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Perform verification after processing all detections
    active_toggles = {key for key, value in toggle_states.items() if value.get()}  # Get all active toggles


    if active_toggles.issubset(detected_classes):
        # All toggled objects are detected, show "Success" message
        success_label.config(text="Success", fg="green")
    else:
        # Missing toggled objects
        missing_classes = active_toggles - detected_classes
        success_label.config(text="Missing: " + ", ".join(missing_classes), fg="red")
        
    # Convert frame to Tkinter format and display
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(frame_rgb)
    img_tk = ImageTk.PhotoImage(image=img_pil)
    video_frame.img_tk = img_tk
    video_frame.configure(image=img_tk)
    video_frame.after(10, show_frame)

# Create toggle switches for PPE items
for ppe_item, var in toggle_states.items():
    toggle_button = ttk.Checkbutton(controls_frame, text=ppe_item, variable=var)
    toggle_button.pack(anchor="w", padx=5)

# Label to display success or missing items
success_label = tk.Label(controls_frame, text="", font=("Arial", 12))
success_label.pack(pady=10)

# Start video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Unable to read camera feed")

# Run the show_frame function
show_frame()

# Close resources on exit
def on_closing():
    cap.release()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()
