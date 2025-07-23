import tkinter as tk
from tkinter import messagebox, simpledialog
import cv2
import os
import numpy as np
import face_recognition
import csv
from datetime import datetime, timedelta
import threading
import time
from PIL import Image, ImageTk

# Fix for embedded Python Tkinter
script_dir = os.path.dirname(os.path.abspath(__file__))
tcl_path = os.path.join(script_dir, "Lib", "site-packages", "tcl")
os.environ['TCL_LIBRARY'] = os.path.join(tcl_path, 'tcl8.6')
os.environ['TK_LIBRARY'] = os.path.join(tcl_path, 'tk8.6')

class FaceLoggerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition Entry Logger")
        self.root.configure(bg="#2c3e50")

        self.known_face_encodings = []
        self.known_face_names = []
        self.last_log_time = {}
        self.log_cooldown = timedelta(hours=1)
        self.load_known_faces()

        self.create_widgets()

    def load_known_faces(self):
        self.known_face_encodings = []
        self.known_face_names = []
        if not os.path.exists("faces"):
            os.makedirs("faces")

        for person_name in os.listdir("faces"):
            person_dir = os.path.join("faces", person_name)
            if os.path.isdir(person_dir):
                for filename in os.listdir(person_dir):
                    if filename.endswith(".npy"):
                        try:
                            encoding = np.load(os.path.join(person_dir, filename))
                            self.known_face_encodings.append(encoding)
                            self.known_face_names.append(person_name)
                        except Exception as e:
                            print(f"Error loading encoding for {person_name}: {e}")
        print(f"Loaded {len(self.known_face_names)} known faces.")

    def create_widgets(self):
        main_frame = tk.Frame(self.root, bg="#2c3e50")
        main_frame.pack(padx=20, pady=20, fill="both", expand=True)

        title_label = tk.Label(main_frame, text="Face Logger", font=("Arial", 24, "bold"), bg="#2c3e50", fg="white")
        title_label.pack(pady=(0, 20))

        button_style = {"font": ("Arial", 12), "bg": "#3498db", "fg": "white", "activebackground": "#2980b9", "relief": "flat", "width": 20, "height": 2}
        
        self.register_button = tk.Button(main_frame, text="Register New Face", command=self.open_registration_window, **button_style)
        self.register_button.pack(pady=10)

        self.log_button = tk.Button(main_frame, text="Start Logging", command=self.open_logging_window, **button_style)
        self.log_button.pack(pady=10)

    def open_registration_window(self):
        RegistrationWindow(self.root, self.load_known_faces)

    def open_logging_window(self):
        LoggingWindow(self.root, self.known_face_encodings, self.known_face_names, self.load_known_faces, self.last_log_time, self.log_cooldown)

class RegistrationWindow(tk.Toplevel):
    def __init__(self, master, callback_on_close):
        super().__init__(master)
        self.title("Register New Face")
        self.configure(bg="#2c3e50")
        self.geometry("700x600")
        self.callback_on_close = callback_on_close

        reg_frame = tk.Frame(self, bg="#2c3e50")
        reg_frame.pack(padx=20, pady=20, fill="both", expand=True)

        self.name_label = tk.Label(reg_frame, text="Enter Name:", font=("Arial", 12), bg="#2c3e50", fg="white")
        self.name_label.pack(pady=(0, 5))
        self.name_entry = tk.Entry(reg_frame, font=("Arial", 12))
        self.name_entry.pack(pady=5, fill="x")

        button_style = {"font": ("Arial", 12), "bg": "#3498db", "fg": "white", "activebackground": "#2980b9", "relief": "flat"}

        self.start_capture_button = tk.Button(reg_frame, text="Start Capture", command=self.start_capture, **button_style)
        self.start_capture_button.pack(pady=10)

        self.capture_button = tk.Button(reg_frame, text="Capture Image", command=self.capture_image, state=tk.DISABLED, **button_style)
        self.capture_button.pack(pady=5)

        self.message_label = tk.Label(reg_frame, text="", font=("Arial", 12), bg="#2c3e50", fg="white")
        self.message_label.pack(pady=10)

        self.video_frame = tk.Label(reg_frame, bg="black")
        self.video_frame.pack(fill="both", expand=True)

        self.cap = None
        self.is_capturing = False
        self.face_encodings_to_save = []
        self.num_images_captured = 0

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def start_capture(self):
        person_name = self.name_entry.get().strip()
        if not person_name:
            messagebox.showerror("Error", "Please enter a name.")
            return

        self.person_dir = os.path.join("faces", person_name)
        if not os.path.exists(self.person_dir):
            os.makedirs(self.person_dir)
        else:
            response = messagebox.askyesno("Name Exists", "This name already exists. Do you want to add more images to this person?")
            if not response:
                return

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open webcam.")
            return

        self.is_capturing = True
        self.start_capture_button.config(state=tk.DISABLED)
        self.capture_button.config(state=tk.NORMAL)
        self.message_label.config(text="Capturing... Look at the camera.")
        self.update_frame()

    def update_frame(self):
        if self.is_capturing and self.cap:
            ret, frame = self.cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                small_frame = cv2.resize(frame_rgb, (0, 0), fx=0.25, fy=0.25)
                face_locations = face_recognition.face_locations(small_frame)

                for (top, right, bottom, left) in face_locations:
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                img_tk = ImageTk.PhotoImage(image=img)
                self.video_frame.config(image=img_tk)
                self.video_frame.image = img_tk
            self.after(10, self.update_frame)

    def capture_image(self):
        if self.cap and self.is_capturing:
            ret, frame = self.cap.read()
            if ret:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame)
                if face_locations:
                    face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
                    self.face_encodings_to_save.append(face_encoding)
                    self.num_images_captured += 1
                    self.message_label.config(text=f"Captured {self.num_images_captured} images. Keep capturing or close.")
                    
                    file_path = os.path.join(self.person_dir, f"face_{self.num_images_captured}.npy")
                    np.save(file_path, face_encoding)
                else:
                    self.message_label.config(text="No face detected. Please try again.")
            else:
                self.message_label.config(text="Failed to capture image.")

    def on_close(self):
        if self.cap:
            self.cap.release()
        self.is_capturing = False
        self.callback_on_close()
        self.destroy()

class LoggingWindow(tk.Toplevel):
    def __init__(self, master, known_face_encodings, known_face_names, callback_on_close, last_log_time, log_cooldown):
        super().__init__(master)
        self.title("Face Recognition Logging")
        self.configure(bg="#2c3e50")
        self.geometry("900x700")
        
        self.known_face_encodings = known_face_encodings
        self.known_face_names = known_face_names
        self.callback_on_close = callback_on_close
        self.last_log_time = last_log_time
        self.log_cooldown = log_cooldown

        log_frame = tk.Frame(self, bg="#2c3e50")
        log_frame.pack(padx=20, pady=20, fill="both", expand=True)

        self.video_frame = tk.Label(log_frame, bg="black")
        self.video_frame.pack(fill="both", expand=True)

        self.status_label = tk.Label(log_frame, text="Detecting faces...", font=("Arial", 16, "bold"), bg="#2c3e50", fg="white")
        self.status_label.pack(pady=10)

        self.cap = None
        self.is_logging = False
        self.log_file = "logs.csv"
        self.init_log_file()

        self.start_logging()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def init_log_file(self):
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Name", "Status"])

    def start_logging(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open webcam.")
            self.on_close()
            return

        self.is_logging = True
        self.update_logging_frame()

    def update_logging_frame(self):
        if self.is_logging and self.cap:
            ret, frame = self.cap.read()
            if ret:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    name = "Unknown"
                    if self.known_face_encodings:
                        matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                        face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                        
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = self.known_face_names[best_match_index]
                            
                            current_time = datetime.now()
                            last_logged = self.last_log_time.get(name)
                            
                            if last_logged is None or (current_time - last_logged) > self.log_cooldown:
                                self.log_entry(name, "Recognized")
                                self.last_log_time[name] = current_time
                                self.status_label.config(text=f"Welcome, {name}!", fg="#2ecc71")
                            else:
                                # Face recognized, but on cooldown
                                self.status_label.config(text=f"{name} (Logged within the last hour)", fg="#f1c40f")
                        else:
                            self.status_label.config(text="Unknown face detected", fg="#e74c3c")
                            self.log_entry("Unknown", "Detected")
                    
                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                img_tk = ImageTk.PhotoImage(image=img)
                self.video_frame.config(image=img_tk)
                self.video_frame.image = img_tk
            self.after(10, self.update_logging_frame)

    def log_entry(self, name, status):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, name, status])

    def on_close(self):
        if self.cap:
            self.cap.release()
        self.is_logging = False
        self.callback_on_close()
        self.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceLoggerApp(root)
    root.mainloop()