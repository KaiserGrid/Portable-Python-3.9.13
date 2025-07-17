import cv2
import os
import numpy as np
import face_recognition
import csv
from datetime import datetime
import time

class FaceLoggerCLI:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.log_file = "logs.csv"
        self.init_log_file()
        self.load_known_faces()

    def init_log_file(self):
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Name", "Status"])

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
        print(f"[INFO] Loaded {len(self.known_face_names)} known faces.")

    def _process_frame(self, frame):
        """Ensures the frame is in the correct format (RGB, 8-bit) for face_recognition."""
        if frame is None:
            return None

        # Ensure frame is 8-bit
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)

        # Convert to RGB if it's a 3-channel image (assuming BGR from OpenCV)
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif len(frame.shape) == 2: # Grayscale, convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif len(frame.shape) == 3 and frame.shape[2] == 4: # RGBA, convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        else:
            print(f"[DEBUG] Unexpected frame shape or channels: {frame.shape}")
            return None

        # Ensure the array is contiguous in memory
        if not rgb_frame.flags['C_CONTIGUOUS']:
            rgb_frame = np.ascontiguousarray(rgb_frame)

        return rgb_frame

    def register_face(self):
        person_name = input("Enter the name of the person to register: ").strip()
        if not person_name:
            print("[ERROR] Name cannot be empty.")
            return

        person_dir = os.path.join("faces", person_name)
        if not os.path.exists(person_dir):
            os.makedirs(person_dir)
        else:
            print(f"[INFO] Directory for {person_name} already exists. Adding more images.")

        cap = cv2.VideoCapture(1)
        cap.set(cv2.CAP_PROP_FPS, 24)
        if not cap.isOpened():
            print("[ERROR] Could not open webcam.")
            return

        print("[INFO] Press 'c' to capture an image, 'q' to quit registration.")
        num_images_captured = 0

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[ERROR] Failed to grab frame or frame is empty.")
                break

            rgb_frame = self._process_frame(frame)
            if rgb_frame is None:
                continue

            try:
                face_locations = face_recognition.face_locations(rgb_frame)
            except RuntimeError as e:
                print(f"[ERROR] RuntimeError during face_locations: {e}")
                print(f"[DEBUG] Frame shape: {frame.shape}, Frame dtype: {frame.dtype}")
                continue

            for (top, right, bottom, left) in face_locations:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            cv2.imshow("Registration - Press 'c' to capture, 'q' to quit", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('c'):
                if face_locations:
                    try:
                        face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
                        num_images_captured += 1
                        file_path = os.path.join(person_dir, f"face_{num_images_captured}.npy")
                        np.save(file_path, face_encoding)
                        print(f"[INFO] Captured image {num_images_captured} for {person_name}.")
                    except RuntimeError as e:
                        print(f"[ERROR] RuntimeError during face_encodings: {e}")
                        print(f"[DEBUG] Frame shape: {frame.shape}, Frame dtype: {frame.dtype}")
                else:
                    print("[WARNING] No face detected. Please ensure your face is visible.")
            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.load_known_faces() # Reload known faces after registration
        print(f"[INFO] Registration for {person_name} complete. Captured {num_images_captured} images.")

    def start_logging(self):
        cap = cv2.VideoCapture(1)
        cap.set(cv2.CAP_PROP_FPS, 24)
        if not cap.isOpened():
            print("[ERROR] Could not open webcam.")
            return

        print("[INFO] Starting logging. Press 'q' to quit.")
        last_logged_name = None
        log_cooldown = 5 # seconds before logging the same person again
        last_log_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[ERROR] Failed to grab frame or frame is empty.")
                break

            rgb_frame = self._process_frame(frame)
            if rgb_frame is None:
                continue

            try:
                face_locations = face_recognition.face_locations(rgb_frame)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            except RuntimeError as e:
                print(f"[ERROR] RuntimeError during face processing: {e}")
                print(f"[DEBUG] Frame shape: {frame.shape}, Frame dtype: {frame.dtype}")
                continue

            current_time = time.time()

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                name = "Unknown"
                if self.known_face_encodings:
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]

                if name != "Unknown":
                    if name != last_logged_name or (current_time - last_log_time) > log_cooldown:
                        self.log_entry(name, "Recognized")
                        print(f'[LOG] {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - Recognized: {name}')
                        last_logged_name = name
                        last_log_time = current_time
                else:
                    print("[WARNING] Unknown face detected. Please register this face.")
                    self.log_entry("Unknown", "Detected")

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0) if name != "Unknown" else (0, 0, 255), 2)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

            cv2.imshow("Logging - Press 'q' to quit", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Logging stopped.")

    def log_entry(self, name, status):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, name, status])

    def run(self):
        while True:
            print("\n--- Face Recognition Entry Logger CLI ---")
            print("1. Register New Face")
            print("2. Start Logging")
            print("3. Exit")
            choice = input("Enter your choice: ").strip()

            if choice == '1':
                self.register_face()
            elif choice == '2':
                self.start_logging()
            elif choice == '3':
                print("Exiting. Goodbye!")
                break
            else:
                print("[ERROR] Invalid choice. Please try again.")

if __name__ == "__main__":
    app = FaceLoggerCLI()
    app.run()