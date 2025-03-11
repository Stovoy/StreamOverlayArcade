import numpy as np
from selfie_segmentation import process_image

def process_image_from_rust(image_data, width, height):
    """
    Process an image coming from Rust and return the segmentation mask.
    
    This function is designed to be called from Rust via PyO3.
    
    Args:
        image_data: A flat byte array representing RGB image data
        width: Image width
        height: Image height
        
    Returns:
        A flat byte array representing the binary mask
    """
    # Reshape the flat array into a 3D image
    image = np.frombuffer(image_data, dtype=np.uint8).reshape((height, width, 3))
    
    # Process the image
    mask = process_image(image)
    
    # Return the mask as a flat byte array
    return mask.tobytes() 