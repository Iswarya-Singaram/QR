import time
import os
import json
import cv2
import torch
import base64
import webbrowser
import ffmpeg
from moviepy.editor import VideoFileClip
from pyzbar.pyzbar import decode
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Define directories
WATCH_DIRECTORY = r"C:\Users\vikra\Videos\Captures"
frames_folder = r"C:\Users\vikra\Videos\Captures\frames_temp"
model_weights = r"C:\Users\vikra\OneDrive\Desktop\Addc\best.pt"
output_folder = r"C:\Users\vikra\OneDrive\Desktop\Addc\mission\output"

# Global flags and output storage
processed_files = set()
found_qr = False
output = {"scanned_files": [], "errors": []}

def is_valid_video(file_path):
    """Checks if a video file is valid. Bypasses FFmpeg probe failures."""
    try:
        if not os.path.exists(file_path):
            print(f"❌ Error: File does not exist - {file_path}")
            return False

        probe = ffmpeg.probe(file_path)
        return "format" in probe
    except Exception as e:
        print(f"⚠️ Warning: FFmpeg probe failed ({e}). Proceeding anyway...")
        return True  # Proceed even if probe fails

def trim_video(file_path):
    """Trims the last 40 seconds of the video and saves it."""
    try:
        if not is_valid_video(file_path):
            print(f"❌ Error: Video file is corrupted or incomplete - {file_path}")
            return None

        video = VideoFileClip(file_path)
        duration = video.duration
        trimmed_video = video.subclip(max(0, duration - 40), duration)
        output_path = os.path.join(WATCH_DIRECTORY, "trimmed_" + os.path.basename(file_path))

        if os.path.exists(output_path):
            print("✅ Trimmed video already exists. Skipping...")
            return output_path

        trimmed_video.write_videofile(output_path, codec="libx264", fps=30)
        video.reader.close()
        if video.audio:
            video.audio.reader.close_proc()

        print(f"✅ Trimmed video saved: {output_path}")
        return output_path

    except Exception as e:
        print(f"❌ Error processing video: {e}")
        return None

def extract_frames(video_path, frame_skip=5):
    """Extracts frames from the trimmed video in reverse order until a QR code is found."""
    global found_qr
    os.makedirs(frames_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    saved_frames = 0

    for frame_idx in range(frame_count - 1, -1, -frame_skip):
        if found_qr:
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        frame_filename = os.path.join(frames_folder, f"frame_{frame_idx}.jpg")
        cv2.imwrite(frame_filename, frame)
        saved_frames += 1

    cap.release()
    print(f"📸 Extracted {saved_frames} frames in reverse order.")

def scan_qr_code_from_bbox(cropped_qr, filename_prefix, index):
    """Scans the QR code from a cropped image."""
    global found_qr
    decoded_objects = decode(cropped_qr)

    if not decoded_objects:
        return None

    for obj in decoded_objects:
        qr_data = obj.data.decode('utf-8')
        print(f"🔍 Detected QR Code: {qr_data}")

        found_qr = True  # Stop processing once QR is found

        if qr_data.startswith("http://") or qr_data.startswith("https://"):
            print(f"🌍 Opening URL: {qr_data}")
            webbrowser.open(qr_data)

        try:
            if qr_data.startswith("data:image"):
                header, encoded = qr_data.split(",", 1)
                image_data = base64.b64decode(encoded)
                image_filename = f"{filename_prefix}_qr_{index}.png"
                with open(image_filename, "wb") as img_file:
                    img_file.write(image_data)
                return {"file": image_filename, "qr_content": "Base64 image data"}
            else:
                return {"file": f"{filename_prefix}_qr_{index}.txt", "qr_content": qr_data}
        except Exception as e:
            output["errors"].append(str(e))
            return None

def process_frame_with_yolov5(frame, model, frame_name):
    """Processes a frame for QR code detection using YOLOv5."""
    global found_qr
    if found_qr:
        return []

    results = model(frame)
    detections = results.pandas().xyxy[0]

    if detections.empty:
        return []

    for index, row in detections.iterrows():
        if found_qr:
            break

        x_min, y_min, x_max, y_max = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
        cropped_qr = frame[y_min:y_max, x_min:x_max]

        filename_prefix = os.path.splitext(frame_name)[0]
        scanned_info = scan_qr_code_from_bbox(cropped_qr, filename_prefix, index)
        if scanned_info:
            output["scanned_files"].append({"frame": frame_name, **scanned_info})
            break  # Stop processing once QR is found

    return output["scanned_files"]

def process_frames_in_folder():
    """Processes frames in the folder using YOLOv5 in reverse order until a QR code is found."""
    global found_qr
    model = torch.hub.load(r"C:\Users\vikra\OneDrive\Desktop\Addc\yolov5", 'custom', path=model_weights, source='local', force_reload=True)
    os.makedirs(output_folder, exist_ok=True)

    # Sort frame names in descending order
    sorted_frames = sorted(os.listdir(frames_folder), reverse=True)

    for frame_name in sorted_frames:
        if found_qr:
            break

        frame_path = os.path.join(frames_folder, frame_name)
        if not frame_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        frame = cv2.imread(frame_path)
        if frame is None:
            output["errors"].append(f"Failed to read {frame_name}")
            continue

        scanned_data = process_frame_with_yolov5(frame, model, frame_name)
        if scanned_data:
            break  # Stop once QR is found

    with open(os.path.join(output_folder, "output.json"), "w") as json_file:
        json.dump(output, json_file, indent=4)

class VideoHandler(FileSystemEventHandler):
    def on_created(self, event):
        """Detects new videos, trims them, and starts QR detection."""
        if event.is_directory:
            return

        file_path = event.src_path
        file_name = os.path.basename(file_path)

        if file_name.startswith("trimmed_") or file_path in processed_files:
            return

        if file_path.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
            print(f"🎥 New video detected: {file_path}")
            wait_for_file_write_complete(file_path)

            trimmed_video_path = trim_video(file_path)
            if trimmed_video_path:
                extract_frames(trimmed_video_path)
                process_frames_in_folder()

            processed_files.add(file_path)

def wait_for_file_write_complete(file_path, timeout=180):
    """Waits until the file stops growing, ensuring it's fully written."""
    previous_size = -1
    start_time = time.time()

    while time.time() - start_time < timeout:
        time.sleep(10)
        current_size = os.path.getsize(file_path)

        if current_size == previous_size:
            print("✅ File is fully written.")
            return

        previous_size = current_size

    print("⚠️ Warning: File may still be writing, but processing anyway.")

if __name__ == "__main__":
    event_handler = VideoHandler()
    observer = Observer()
    observer.schedule(event_handler, WATCH_DIRECTORY, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
