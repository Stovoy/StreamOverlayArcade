import sys
import os

# We need to modify the sys.path when calling from pyo3 due to some Windows issues.
lib_paths = [p for p in sys.path if p.endswith('\\Lib')]
for lib_path in lib_paths:
    site_packages_path = lib_path + '\\site-packages'
    if not os.path.exists(site_packages_path):
        continue
    if site_packages_path not in sys.path:
        sys.path.append(site_packages_path)

# Similarly, pyo3 has trouble with the DLL search path, so we need to add all directories manually.
for path_dir in os.environ.get('PATH', '').split(os.pathsep):
    if not os.path.exists(path_dir):
        continue
    os.add_dll_directory(path_dir)

import numpy as np
import mediapipe as mp
from PIL import Image
import colorsys
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/deeplab_v3.tflite')

# Create options for ImageSegmenter
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.ImageSegmenterOptions(base_options=base_options, output_category_mask=True)

# Create the image segmenter
segmenter = vision.ImageSegmenter.create_from_options(options)

# Complete category names mapping for the DeepLab model
DEFAULT_CATEGORY_NAMES = {
    0: "Background",
    1: "Aeroplane",
    2: "Bicycle",
    3: "Bird",
    4: "Boat",
    5: "Bottle",
    6: "Bus",
    7: "Car",
    8: "Cat",
    9: "Chair",
    10: "Cow",
    11: "Dining Table",
    12: "Dog",
    13: "Horse",
    14: "Motorbike",
    15: "Person",
    16: "Potted Plant",
    17: "Sheep",
    18: "Sofa",
    19: "Train",
    20: "TV/Monitor"
}

# Store the last detected categories
_last_categories = []

# Create a persistent color map for all possible class IDs (0-20 for DeepLab)
# This ensures consistent colors across frames
PERSISTENT_COLOR_MAP = {}

def generate_colors(num_colors):
    """
    Generate evenly distributed colors.
    
    Args:
        num_colors: Number of colors to generate
        
    Returns:
        List of RGB color tuples
    """
    colors = [(0, 0, 0)]  # Start with black for background
    
    for i in range(1, num_colors):
        # Use HSV with evenly distributed hue values
        h = i / num_colors
        # Convert HSV to RGB
        r, g, b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
        # Scale to 0-255 range for RGB
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    
    return colors

def initialize_color_map():
    global PERSISTENT_COLOR_MAP
    colors = generate_colors(len(DEFAULT_CATEGORY_NAMES))
    
    for i in range(len(DEFAULT_CATEGORY_NAMES)):
        PERSISTENT_COLOR_MAP[i] = colors[i]

initialize_color_map()

def process_image(image_data, category_names=None):
    """
    Process an image and return the segmentation mask with different colors.
    
    Args:
        image_data: A numpy array of shape (height, width, 3) in RGB format
        category_names: Optional dictionary mapping category IDs to names
        
    Returns:
        A tuple containing:
        - A numpy array of shape (height, width, 3) with colored mask
        - A numpy array of shape (height, width) with raw category values
        - A dictionary mapping category indices to RGB colors
        - A list of category information dictionaries with 'id', 'name', and 'color'
    """
    global _last_categories, PERSISTENT_COLOR_MAP
    
    # Create a MediaPipe image from the numpy array
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_data)
    
    # Retrieve the masks for the segmented image
    segmentation_result = segmenter.segment(mp_image)
    category_mask = segmentation_result.category_mask
    
    # Get the raw category values
    raw_mask = category_mask.numpy_view()
    
    # Get unique categories
    unique_categories = np.unique(raw_mask)
    
    # Use the persistent color map for consistent colors
    color_map = {category_id: PERSISTENT_COLOR_MAP.get(category_id, (255, 0, 255)) for category_id in unique_categories}
    
    # Use provided category names or default
    if category_names is None:
        category_names = DEFAULT_CATEGORY_NAMES
    
    # Create category information
    category_info = []
    for i, category_id in enumerate(unique_categories):
        # Get name from provided dictionary or use default naming
        name = category_names.get(category_id, f"Category {category_id}")
        
        category_info.append({
            'id': int(category_id),
            'name': name,
            'color': color_map[category_id]
        })
    
    # Store the categories for later retrieval
    _last_categories = category_info
    
    # Create colored mask
    height, width = raw_mask.shape
    colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Assign colors to each category
    for category, color in color_map.items():
        colored_mask[raw_mask == category] = color
    
    return colored_mask, raw_mask, color_map, category_info

    