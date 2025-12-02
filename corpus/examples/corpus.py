import onnxruntime as ort
import numpy as np
from PIL import Image
import requests
from io import BytesIO

MODEL_NAME = "corpus-{}.onnx"
MODEL_PATH = "../" + MODEL_NAME.format("small-static")
IMG_SIZE = (224, 224) # default MobileNet shape.

# Load the ONNX model.
try:
    session = ort.InferenceSession(
      MODEL_PATH,
      providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"Model '{MODEL_PATH}' loaded successfully.")
except Exception as e:
    print(f"Error loading ONNX model: {e}")
    exit()

def preprocess_image(url, target_size, batch_size=1):
    """Resize and expand image. Convert image into vector."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img = img.resize(target_size)
        img_array = np.asarray(img, dtype=np.float32)

        # Be aware if it's in NCHW format.
        img_array = np.expand_dims(img_array, axis=0)  # (1, H, W, C)
        img_array = np.repeat(img_array, batch_size, axis=0)  # (batch_size, H, W, C)

        return img_array

    except requests.exceptions.RequestException as e:
        print(f"Error fetching image {url}: {e}")
        return None
    except Exception as e:
        print(f"Error processing image {url}: {e}")
        return None

urls = [
  "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg",
]

for url in urls:
    img_array = preprocess_image(url, IMG_SIZE)

    if img_array is None:
        continue

    input_feed = {input_name: img_array}
    predictions = session.run([output_name], input_feed)

    # The output is a list of NumPy arrays. We access the first (and only) output.
    score = predictions[0][0][0] * 100 # Extract the scalar value and convert to percentage.

    if score >= 80:
        print("{} is a nude with {:.2f}% confidence".format(url, score))
    else:
        print("{} is **not** a nude with {:.2f}% confidence".format(url, 100 - score))