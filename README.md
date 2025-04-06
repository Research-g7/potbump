# Road Hazard Detection System

Real-time detection system for road hazards like potholes and speed bumps using YOLO and computer vision.

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/yourusername/road-hazard-detection.git
cd road-hazard-detection
```

2. Create and activate virtual environment:
```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. Install requirements:
```bash
pip install -r requirements.txt
```

4. Download required model files:
- Download YOLO model files and place them in:
  - `data/models/POTHOLE5/best.pt`
  - `data/models/SPEEDBUMPS5/best.pt`

5. Add sound files:
- Add alert sound files to:
  - `data/sounds/pothole_alert.mp3`
  - `data/sounds/speedbump_alert.mp3`

## Configuration
1. Update Supabase credentials in `src/Thesis.py`
2. Configure GPS port settings if needed
3. Update email and Telegram settings

## Running the Application
```bash
cd src
python Thesis.py
```

The application will:
- Start the camera feed
- Generate a QR code for remote viewing
- Begin detecting road hazards
- Send alerts via email and Telegram

Press 'q' to quit the program.

## Requirements
- Python 3.8+
- CUDA-compatible GPU (recommended)
- Webcam or compatible camera
- GPS module (optional)
