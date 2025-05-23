import streamlit as st
import flickrapi
import requests
from io import BytesIO
from PIL import Image, ImageDraw
import time
import cv2  # OpenCV for face detection
import numpy as np
import random
import os
import base64
import re
from ultralytics import YOLO
from datetime import datetime, timedelta



# Load YOLOv8 pretrained model
model = YOLO("yolov8n.pt")  # Use the nano model for lightweight performance


# Load font file and encode it
def load_font(font_path):
    with open(font_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Path to your font file
font_base64 = load_font("fonts/Baldessari-Regular.ttf")

# Inject CSS to load the font and hide header and footer.
st.markdown(
    f"""
    <style>
    @font-face {{
        font-family: 'Baldessari-Regular';
        src: url(data:font/ttf;base64,{font_base64}) format('truetype');
    }}

    /* Apply font to common Streamlit elements */
    html, body, [class*="css"], h1, h2, h3, h4, h5, h6, p,
    .stText, .stMarkdown, .stButton > button, .stLabel, .stRadio, .stSelectbox, .stTextInput, .stSlider {{
        font-family: 'Baldessari-Regular' !important;
        text-align: center; /* Center-align text */
    }}

    /* Add space between the image and the button */
    .stButton {{
        margin-top: 20px; /* Add space below the button */
        margin-bottom: 20px; /* Add space below the button */
    }}

    /* Override the padding (gap) for the container */
    .st-emotion-cache-vlxhtx {{
        gap: 0px !important; /* Remove the gap */
    }}

    footer {{display: none;}}
    header {{display: none;}}
    </style>
    """,
    unsafe_allow_html=True
)

# --- Your custom function for image enhancement ---
def enhance_image(image, contrast_factor=1.3, black_point=30):
    from PIL import ImageEnhance
    image = image.convert("RGB")
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast_factor)

    # Adjust black point using NumPy
    image_np = np.array(image)
    image_np[image_np < black_point] = black_point
    image = Image.fromarray(image_np, mode=image.mode)

    return image

# --- Load API keys from secrets.toml ---
api_key = st.secrets["flickr"]["api_key"]
api_secret = st.secrets["flickr"]["api_secret"]

# --- Setup ---
flickr = flickrapi.FlickrAPI(api_key, api_secret, format='parsed-json')
photos_folder = "photos"
os.makedirs(photos_folder, exist_ok=True)

# --- Load OpenCV Haar Cascade for face detection ---
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

# --- Session State: track where we are in time ---
if "timenow" not in st.session_state:
    st.session_state.timenow = int(time.time())
    st.session_state.time_cursor = st.session_state.timenow - (20 * 365 * 24 * 60 * 60)

if "processed_ids" not in st.session_state:
    st.session_state.processed_ids = set()

# --- Session State: track last trigger time ---
if "last_trigger_time" not in st.session_state:
    st.session_state.last_trigger_time = datetime.now()

# Initialize session state for last image
if "last_image" not in st.session_state:
    st.session_state.last_image = None

# --- Function to Check Idle Mode ---
def should_run_idle():
    idle_interval = timedelta(seconds=40)  # 40-second interval
    return datetime.now() - st.session_state.last_trigger_time > idle_interval

# --- Randomize the time_cursor between 18 and 20 years ago ---
def randomize_time_cursor():
    eighteen_years_ago = st.session_state.timenow - (18 * 365 * 24 * 60 * 60)
    twenty_years_ago = st.session_state.timenow - (20 * 365 * 24 * 60 * 60)
    return random.randint(twenty_years_ago, eighteen_years_ago)

def extract_random_word(title):
    """
    Extract a random word from the title.
    If the title is empty or no words are found, return 'Untitled'.
    """
    words = title.split()  # Split the title into words
    if words:
        title= random.choice(words).upper()  # Randomly pick a word and capitalize it
        cleaned_title = remove_numbers_from_title(title)
        return cleaned_title
    return "WRONG"  # Fallback if no words are found

def remove_numbers_from_title(title):
    """
    Remove all numbers from the given title string.
    
    :param title: The title string to process.
    :return: The title with numbers removed.
    """
    return re.sub(r'[\d_]+', '', title).strip()


def display_image_with_custom_height(image, max_height="90vh"):
    """
    Display the image with a custom height using CSS.
    The `max_height` parameter ensures the image does not exceed the screen height.

    :param image: The image to display (PIL object or URL).
    :param caption: The caption to display below the image.
    :param max_height: The maximum height of the image (default: "90vh", 90% of viewport height).
    """
    # Save the image to a temporary file
    image.save("temp_image.png")  # Save the image temporarily

    # Inject custom CSS to control the image height
    st.markdown(
        f"""
        <style>
            .custom-image {{
                max-height: {max_height};
                width: auto;
                display: block;
                margin: 0 auto;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Display the image with the custom class
    st.markdown(
        f'<img src="temp_image.png" alt="Image" class="custom-image" />',
        unsafe_allow_html=True,
    )

    # Display the caption below the image
    # st.caption(caption)


# --- Main Logic ---
def process_image():
    st.session_state.last_trigger_time = datetime.now()  # Update the last trigger time
    found_image = False
    tries = 0

    while not found_image and tries < 10:
        tries += 1
        #slightly_earlier = st.session_state.time_cursor - 120

        # Randomize the time window
        st.session_state.time_cursor = randomize_time_cursor()
        slightly_earlier = st.session_state.time_cursor - 120  # 2-minute window

        try:
            photos = flickr.photos.search(
                max_taken_date=st.session_state.time_cursor,
                min_taken_date=slightly_earlier,
                per_page="10",
            )

            for photo in photos['photos']['photo']:
                photo_id = photo.get('id')
                if photo_id in st.session_state.processed_ids:
                    continue

                st.session_state.processed_ids.add(photo_id)
                url = f"https://live.staticflickr.com/{photo.get('server')}/{photo.get('id')}_{photo.get('secret')}.jpg"
                response = requests.get(url, stream=True)
                response.raise_for_status()

                # Fetch photo title using flickr.photos.getInfo
                photo_info = flickr.photos.getInfo(photo_id=photo_id)
                photo_title = photo_info['photo']['title']['_content'] or "wrong"

                original_image = Image.open(BytesIO(response.content))
                enhanced_image = enhance_image(original_image)
                bw_image = enhanced_image.convert("L")

                # Convert to NumPy array for YOLO
                image_np = np.array(enhanced_image)

                # --- YOLO Face Detection ---
                results = model(image_np)
                detections = results[0].boxes.data.cpu().numpy()  # YOLO detections

                # Filter detections by confidence threshold
                CONFIDENCE_THRESHOLD = 0.5
                faces = [
                    (int(x1), int(y1), int(x2 - x1), int(y2 - y1)) 
                    for x1, y1, x2, y2, conf, cls in detections 
                    if conf > CONFIDENCE_THRESHOLD and cls == 0  # Class 0 is 'person'
                ]

                # Refine face detection using Haar cascades
                FACE_PROPORTION = 0.4  # Top 40% of the bounding box for the face
                refined_faces = []
                for (x, y, w, h) in faces:
                    # Extract the upper portion of the bounding box (face region)
                    face_region_y = y
                    face_region_h = int(h * FACE_PROPORTION)
                    face_region = image_np[face_region_y:face_region_y + face_region_h, x:x + w]

                    # Convert the face region to grayscale for Haar cascades
                    gray_face_region = cv2.cvtColor(face_region, cv2.COLOR_RGB2GRAY)

                    # Detect faces with Haar cascades
                    haar_faces = face_cascade.detectMultiScale(
                        gray_face_region, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                    )

                    if len(haar_faces) > 0:
                        # Use the first detected Haar face to refine the position
                        (fx, fy, fw, fh) = haar_faces[0]
                        refined_faces.append((x + fx, face_region_y + fy, fw, fh))
                    else:
                        # Fall back to YOLO's face region estimate
                        #refined_faces.append((x, face_region_y, w, face_region_h))
                        break

                if len(refined_faces) > 0:
                    # Create a high-resolution version of the image for anti-aliasing
                    scale_factor = 4  # Adjust as needed for better quality
                    high_res_size = (bw_image.width * scale_factor, bw_image.height * scale_factor)
                    high_res_image = bw_image.resize(high_res_size).convert("RGB")
                    draw = ImageDraw.Draw(high_res_image)
                    colors = ["#d93832", "#993333", "#4d8f56", "#3b86ac", "#e4d050", "#e0923b"]  # Updated hex colors

                    for (x, y, w, h) in refined_faces:
                        # Calculate center and radius (scaled)
                        cx = (x + w // 2) * scale_factor
                        cy = (y + h // 2) * scale_factor
                        radius = int(max(w, h) * 0.45 * scale_factor)  # Adjust radius multiplier
                        color = random.choice(colors)

                        # Draw a high-resolution ellipse
                        draw.ellipse(
                            [(cx - radius, cy - radius), (cx + radius, cy + radius)],
                            fill=color,
                            outline=color,
                            width=2 * scale_factor  # Scale the width for higher resolution
                        )

                    # Downscale the high-resolution image back to the original size
                    draw_image = high_res_image.resize(bw_image.size, resample=Image.Resampling.LANCZOS)



                    # Extract a random word from the title
                    random_word = extract_random_word(photo_title)
                                        

                    # st.image(draw_image, caption=f"Made with Flickr image {photo_id}", use_container_width=True)
                    st.image(draw_image, use_container_width=True)

                    # Example usage
                    # display_image_with_custom_height(
                    #     image=draw_image,
                    #     max_height="90vh",  # Fit 90% of the viewport height
                    # )



                    st.write(f'"{random_word}"')
                    found_image = True
                    break  # Done with one image

        except Exception as e:
            st.warning(f"Error: {e}")

        # Advance the time cursor for next search
        st.session_state.time_cursor += 240

    if not found_image:
        st.info("No suitable image found this time. Try again.")

# --- App Header ---
st.write("baldessari neverending")
#st.write("BALDESSARI NEVERENDING")

# --- Trigger Logic ---
manual_trigger = st.button("🔄 make another")  # Manual trigger
if manual_trigger:
    st.session_state.last_trigger_time = datetime.now()  # Update the last trigger time
    process_image()

# --- Idle Mode ---
# if should_run_idle():
#     st.write("idle")  # Debugging (remove after testing)
#     process_image()
#     st.rerun()  # Refresh the app after processing the image in idle mode
