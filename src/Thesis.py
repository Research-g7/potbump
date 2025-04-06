import cv2
import serial
import pynmea2
import math
import cvzone
import folium
from ultralytics import YOLO
import os
from supabase import create_client, Client
import hashlib
from datetime import datetime
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
from lxml import etree
import socket
from flask import Flask, Response
import qrcode
import threading
from playsound import playsound
import time
import requests  # Add this import
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path

# Set up base paths
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = DATA_DIR / "models"
SOUNDS_DIR = DATA_DIR / "sounds"

# Update model paths
pothole_model = YOLO(str(MODELS_DIR / "POTHOLE5/best.pt"))
speedbump_model = YOLO(str(MODELS_DIR / "SPEEDBUMPS5/best.pt"))

# Update sound paths
POTHOLE_SOUND = str(SOUNDS_DIR / "pothole_alert.mp3")
SPEEDBUMP_SOUND = str(SOUNDS_DIR / "speedbump_alert.mp3")

# Create required directories
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SOUNDS_DIR, exist_ok=True)

# Optional: Set CUDA device if available
import torch
if torch.cuda.is_available():
    pothole_model.to('cuda')
    speedbump_model.to('cuda')

# Define class labels
pothole_class_labels = ['Pothole']
speedbump_class_labels = ['Speed Bump']

# Supabase URL and Anon API Key
SUPABASE_URL = "https://nntpyaqqmlhjryyfybjo.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5udHB5YXFxbWxoanJ5eWZ5YmpvIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDEyNTY1NDAsImV4cCI6MjA1NjgzMjU0MH0.iZUl-zCmnCTxvgK8jwMa-1R_PPVUW2-G89lxf2zL-1U"

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize geolocator (for location)
geolocator = Nominatim(user_agent="PotholeDetectionApp")

# Global variables
cap = None
img = None
saved_image_hashes = set()
locations = []
last_alert_time = 0
ALERT_COOLDOWN = 2  # seconds

# Gmail credentials
gmail_user = "researchersgoup7@gmail.com"
gmail_password = "qref payr amqp hsav"  # Replace with your 16-character App Password
to_emails = ["firebase0192@gmail.com"] # List of recipient emails

def read_gps():
    port = "COM4"  # Use the port detected on your Windows machine
    baudrate = 9600  # Typical baudrate for u-blox modules
    try:
        with serial.Serial(port, baudrate, timeout=1) as gps_serial:
            print(f"Listening to GPS data on {port}...")
            while True:
                line = gps_serial.readline().decode('ascii', errors='ignore').strip()
                if line.startswith('$GNGGA') or line.startswith('$GPGGA'):
                    try:
                        msg = pynmea2.parse(line)
                        latitude = msg.latitude
                        longitude = msg.longitude
                        print(f"Latitude: {latitude}, Longitude: {longitude}")
                        return latitude, longitude
                    except pynmea2.ParseError as e:
                        print(f"Failed to parse: {e}")
    except serial.SerialException as e:
        print(f"Error: {e}. Check if the GPS is connected and the port is correct.")
    return "Unknown", "Unknown"

def upload_picture(image_bytes, detection_name, bucket_name="avatar"):
    try:
        supabase_path = f"Baclaran/{detection_name}.jpg"
        response = supabase.storage.from_(bucket_name).upload(supabase_path, image_bytes)
        if response.get("error"):
            raise Exception(f"Upload failed: {response['error']}")
        public_url = supabase.storage.from_(bucket_name).get_public_url(supabase_path)
        return public_url
    except Exception as e:
        print(f"Error uploading picture: {e}")
        return None

def upload_data_to_supabase(label, confidence, timestamp, latitude, longitude, distance, speed_text):
    data = {
        "Label": label,
        "Confidence": confidence,
        "Timestamp": timestamp,
        "Latitude": latitude,
        "Longitude": longitude,
        "Distance": round(distance, 1),
        "DetectionSpeed": speed_text
    }
    try:
        response = supabase.table("Research").insert(data).execute()
        if response.get("error"):
            raise Exception(f"Error inserting data: {response['error']}")
        print("Data inserted successfully:", response.data)
    except Exception as e:
        print(f"Error inserting data into Supabase: {str(e)}")

def generate_image_hash(image):
    return hashlib.md5(image).hexdigest()

def get_location():
    try:
        location = geolocator.geocode("Your Location", timeout=10)
        if location:
            return location.latitude, location.longitude
    except (GeocoderTimedOut, GeocoderUnavailable):
        print("Geocoding service is unavailable or timed out.")
    return "Unknown", "Unknown"

def create_kml(locations, output_file="Baclaran.kml"):
    kml = etree.Element("kml", xmlns="http://www.opengis.net/kml/2.2")
    document = etree.SubElement(kml, "Document")

    # Add style for Potholes (red pushpin)
    pothole_style = etree.SubElement(document, "Style", id="potholeStyle")
    pothole_icon_style = etree.SubElement(pothole_style, "IconStyle")
    pothole_icon = etree.SubElement(pothole_icon_style, "Icon")
    pothole_href = etree.SubElement(pothole_icon, "href")
    pothole_href.text = "http://maps.google.com/mapfiles/kml/pushpin/red-pushpin.png"

    # Add style for Speed Bumps (yellow pushpin)
    speedbump_style = etree.SubElement(document, "Style", id="speedbumpStyle")
    speedbump_icon_style = etree.SubElement(speedbump_style, "IconStyle")
    speedbump_icon = etree.SubElement(speedbump_icon_style, "Icon")
    speedbump_href = etree.SubElement(speedbump_icon, "href")
    speedbump_href.text = "http://maps.google.com/mapfiles/kml/pushpin/ylw-pushpin.png"

    for location in locations:
        placemark = etree.SubElement(document, "Placemark")
        name_tag = etree.SubElement(placemark, "name")
        name_tag.text = location["name"]
        
        # Add style reference based on detection type
        style_url = etree.SubElement(placemark, "styleUrl")
        if "Pothole" in location["name"]:
            style_url.text = "#potholeStyle"
        else:
            style_url.text = "#speedbumpStyle"
        
        # Create the image URL based on detection type with double slash
        image_url = f"https://nntpyaqqmlhjryyfybjo.supabase.co/storage/v1/object/public/avatar/September-2025/Baclaran//{location['name']}.jpg"
        
        # Description with image and details
        description = etree.SubElement(placemark, "description")
        description_text = f'''
        <![CDATA[
        <div style="width: 300px;">
            <h3>{location["name"]}</h3>
            <img src="{image_url}" width="300"/>
            <p>Latitude: {location["lat"]}</p>
            <p>Longitude: {location["lng"]}</p>
            <p>Type: {"Pothole" if "Pothole" in location["name"] else "Speed Bump"}</p>
        </div>
        ]]>
        '''
        description.text = description_text
        
        point = etree.SubElement(placemark, "Point")
        coordinates = etree.SubElement(point, "coordinates")
        coordinates.text = f"{location['lng']},{location['lat']},0"

    kml_string = etree.tostring(kml, pretty_print=True, xml_declaration=True, encoding="UTF-8")
    try:
        supabase_path = "Baclaran/locations.kml"
        response = supabase.storage.from_("avatar2").upload(supabase_path, kml_string)
        if response.get("error"):
            raise Exception(f"Upload failed: {response['error']}")
        print("KML file uploaded successfully")
    except Exception as e:
        print(f"Error uploading KML file: {e}")

def save_location_to_kml(label, latitude, longitude):
    global locations
    # Create the full detection name with coordinates
    detection_name = f"{label}_{latitude}_{longitude}"
    locations.append({"name": detection_name, "lat": latitude, "lng": longitude})
    create_kml(locations, output_file="Baclaran.kml")

def send_telegram_message(bot_token, group_id, message):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": group_id,
        "text": message,
        "parse_mode": "Markdown"
    }
    response = requests.post(url, json=payload)
    return response.json()

def send_email_alert(detection_name, clean_label, conf, timestamp, latitude, longitude, distance, speed_text):
    try:
        msg = MIMEMultipart()
        msg["From"] = gmail_user
        msg["To"] = ", ".join(to_emails)
        msg["Subject"] = "🚨 *New Detection Alert* 🚨"
        
        body = f"""
Details:
📸 *Image*: {detection_name}
🏷️ *Label*: {clean_label}
✅ *Confidence*: {conf}
🕒 *Timestamp*: {timestamp}
📍 *Location*:
   • Latitude: {latitude}
   • Longitude: {longitude}
📏 *Distance*: {distance} meters
⚡ *Detection Speed*: {speed_text}

This is an automated alert from the detection system.
        """
        
        msg.attach(MIMEText(body, "plain"))
        
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(gmail_user, gmail_password)
            server.send_message(msg)
            print("Email alert sent successfully!")
            
    except Exception as e:
        print(f"Error sending email: {str(e)}")

def process_detections(detection_results, class_labels, label_color, file_prefix, _):
    global saved_image_hashes, img
    BOT_TOKEN = "8099325668:AAFIL_C4PJ9xbHuMRKqom6Bi3KJftdQsWZQ"
    GROUP_ID = "-1002625293512"

    detection_speed = {
        'preprocess': detection_results[0].speed['preprocess']
    }
    
    speed_text = f"{detection_speed['preprocess']:.1f}"

    for r in detection_results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            if conf > 0.5:
                distance = estimate_distance(h)
                
                if class_labels[cls] == 'Pothole':
                    play_alert_sound(POTHOLE_SOUND)
                elif class_labels[cls] == 'Speed Bump':
                    play_alert_sound(SPEEDBUMP_SOUND)
                
                cvzone.cornerRect(img, (x1, y1, w, h), colorR=label_color, t=2)
                
                latitude, longitude = read_gps()
                
                clean_label = class_labels[cls]
                detection_name = f"{clean_label}_{latitude}_{longitude}"
                
                display_text = f'{clean_label} ({latitude}, {longitude}) {distance}m'
                cvzone.putTextRect(img, f'{display_text} {conf}', (x1, y1 - 10), scale=0.8, thickness=1, colorR=label_color)
                
                # Encode current frame for direct upload
                _, img_encoded = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 60])
                img_bytes = img_encoded.tobytes()
                
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Upload directly to Supabase without saving locally
                upload_data_to_supabase(clean_label, conf, timestamp, latitude, longitude, distance, speed_text)
                save_location_to_kml(clean_label, latitude, longitude)
                
                telegram_message = f"""
🚨 *New Detection Alert* 🚨

📸 *Image*: {detection_name}
🏷️ *Label*: {clean_label}
✅ *Confidence*: {conf}
🕒 *Timestamp*: {timestamp}
📍 *Location*:
   • Latitude: {latitude}
   • Longitude: {longitude}
📏 *Distance*: {distance} meters
⚡ *Detection Speed*: {speed_text}
"""
                send_telegram_message(BOT_TOKEN, GROUP_ID, telegram_message)
                send_email_alert(detection_name, clean_label, conf, timestamp, 
                               latitude, longitude, distance, speed_text)
    
    return None

def get_local_ip():
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    return local_ip

app = Flask(__name__)

@app.route('/')
def index():
    return "<h1>Camera Stream</h1><p>Use /video_feed to view the stream.</p>"

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    global cap, img
    frame_count = 0
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        # Process every third frame to reduce load with 4K
        frame_count += 1
        if frame_count % 3 != 0:  # Skip two frames
            continue
            
        # Resize frame for processing while maintaining quality
        img = resize_frame(frame, target_width=1920)  # Increased from 320 to 1920
        
        # Perform detection on the frame
        pothole_results = pothole_model(img)
        process_detections(pothole_results, pothole_class_labels, (255, 0, 0), "Pothole", None)
        speedbump_results = speedbump_model(img)
        process_detections(speedbump_results, speedbump_class_labels, (0, 255, 0), "SpeedBump", None)

        # Encode with better quality but optimized compression
        ret, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            continue
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def start_detection():
    global cap, img
    frame_count = 0
    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to read from camera")
            break
            
        # Process every 2nd or 3rd frame
        frame_count += 1
        if frame_count % 2 != 0:  # Skip every other frame
            continue
        
        # Resize frame for processing (use a smaller size)
        img = resize_frame(frame, target_width=480)  # Reduced from 640
        
        # Perform detection on the frame
        pothole_results = pothole_model(img)
        process_detections(pothole_results, pothole_class_labels, (255, 0, 0), "Pothole", None)
        speedbump_results = speedbump_model(img)
        process_detections(speedbump_results, speedbump_class_labels, (0, 255, 0), "SpeedBump", None)

        cv2.imshow('Detection', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def start_flask_server():
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

def estimate_distance(box_height):
    # Known parameters (you may need to adjust these based on your camera and typical object sizes)
    KNOWN_HEIGHT = 0.3  # Average height of pothole/speedbump in meters
    FOCAL_LENGTH = 800  # Focal length in pixels (needs calibration)
    
    # Calculate distance using triangle similarity
    distance = (KNOWN_HEIGHT * FOCAL_LENGTH) / box_height
    return round(distance, 1)  # Round to 1 decimal place

def play_alert_sound(sound_file):
    global last_alert_time
    current_time = time.time()
    if current_time - last_alert_time >= ALERT_COOLDOWN:
        try:
            threading.Thread(target=playsound, args=(sound_file,), daemon=True).start()
            last_alert_time = current_time
        except Exception as e:
            print(f"Error playing sound: {e}")

def resize_frame(frame, target_width=320):  # Reduced from 640 to 320
    """Resize frame while maintaining aspect ratio"""
    height, width = frame.shape[:2]
    aspect_ratio = width / height
    target_height = int(target_width / aspect_ratio)
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)

if __name__ == '__main__':
    try:
        # Initialize camera first
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("Could not open camera")
            
        # Optimized 4K camera settings for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)      # 4K width
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)     # 4K height
        cap.set(cv2.CAP_PROP_FPS, 30)               # Standard FPS for stability
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)         # Minimum buffer size
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # Use MJPG codec for better performance
        
        # Enable camera hardware optimization if available
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)          # Enable autofocus
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)   # Auto exposure

        # Get local IP and generate QR code
        ip = get_local_ip()
        port = 5000
        stream_url = f'http://{ip}:{port}/video_feed'

        # Generate and save the QR code
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=5,
            border=2,
        )
        qr.add_data(stream_url)
        qr.make(fit=True)

        qr_img = qr.make_image(fill='black', back_color='white')
        qr_img.save('camera_stream_qr.png')

        print(f"Stream URL: {stream_url}")
        print("QR code saved as 'camera_stream_qr.png'. Scan it to view the camera stream.")
        print("Press 'q' to quit the program")

        # Start Flask server in a separate thread
        flask_thread = threading.Thread(target=start_flask_server)
        flask_thread.daemon = True
        flask_thread.start()

        # Start detection immediately in the main thread
        start_detection()

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()