from PIL import Image
import numpy as np
import mediapipe as mp
import cv2
import time

mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=0)

BG_COLOR = (192, 192, 192)  # Gray
MASK_COLOR = (255, 255, 255)  # White

def process_image(image_data):
    """
    Process an image and return the segmentation mask.
    
    Args:
        image_data: A numpy array of shape (256, 256, 3) in RGB format
        
    Returns:
        A numpy array of shape (256, 256) with binary mask values (0 or 255)
    """
    # Ensure image is in RGB format
    rgb_image = image_data
    
    # Process the image with MediaPipe
    results = selfie_segmentation.process(rgb_image)
    
    # Extract the mask (single channel)
    mask = results.segmentation_mask
    
    # Convert to binary mask (0 or 255)
    binary_mask = np.where(mask > 0.1, 255, 0).astype(np.uint8)
    
    return binary_mask 