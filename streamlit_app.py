
import streamlit as st
import flickrapi
import requests
from io import BytesIO
from PIL import Image, ImageDraw
import time
import face_recognition
import random
import os

# --- Your custom function for image enhancement ---
def enhance_image(image, contrast_factor=1.1, black_point=50):
    from PIL import ImageEnhance
    image = image.convert("RGB")
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast_factor)
    image = image.point(lambda p: max(p - black_point, 0))
    return image

# --- Load API keys from secrets.toml ---
api_key = st.secrets["flickr"]["api_key"]
api_secret = st.secrets["flickr"]["api_secret"]

# --- Setup ---
flickr = flickrapi.FlickrAPI(api_key, api_secret, format='parsed-json')
photos_folder = "photos"
os.makedirs(photos_folder, exist_ok=True)

# --- Session State: track where we are in time ---
if "timenow" not in st.session_state:
    st.session_state.timenow = int(time.time())
    st.session_state.time_cursor = st.session_state.timenow - (20 * 365 * 24 * 60 * 60)

if "processed_ids" not in st.session_state:
    st.session_state.processed_ids = set()

st.title("🔴 Baldessari Neverending")

st.write("Click the button below to search and process a new image taken ~20 years ago.")

if st.button("🔄 Get Next Image"):
    found_image = False
    tries = 0

    while not found_image and tries < 10:
        tries += 1
        slightly_earlier = st.session_state.time_cursor - 120

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

                original_image = Image.open(BytesIO(response.content))
                enhanced_image = enhance_image(original_image)
                bw_image = enhanced_image.convert("L")

                # Use face_recognition to find faces
                image_np = face_recognition.load_image_file(BytesIO(response.content))
                face_locations = face_recognition.face_locations(image_np)

                if face_locations:
                    draw_image = bw_image.convert("RGB")
                    draw = ImageDraw.Draw(draw_image)
                    colors = ["red", "green", "blue", "yellow", "orange"]

                    for (top, right, bottom, left) in face_locations:
                        cx = (left + right) // 2
                        cy = (top + bottom) // 2
                        radius = int(max((right - left), (bottom - top)) * 0.55)
                        color = random.choice(colors)
                        draw.ellipse(
                            [(cx - radius, cy - radius), (cx + radius, cy + radius)],
                            fill=color, outline=color, width=2
                        )

                    st.image(draw_image, caption=f"Faces hidden in image {photo_id}", use_container_width=True)
                    found_image = True
                    break  # Done with one image

        except Exception as e:
            st.warning(f"Error: {e}")

        # Advance the time cursor for next search
        st.session_state.time_cursor += 240

    if not found_image:
        st.info("No suitable image found this time. Try again.")
