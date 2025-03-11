from PIL import Image
import numpy as np
import mediapipe as mp
import cv2
import time

mp_selfie_segmentation = mp.solutions.selfie_segmentation

IMAGE_FILENAMES = ['image.png']

DESIRED_HEIGHT = 256
DESIRED_WIDTH = 256
BG_COLOR = (192, 192, 192)
MASK_COLOR = (255, 255, 255)

def process_static_images():
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=0) as selfie_segmentation:
        for iteration in range(10):
            start_time = time.time()
            
            for idx, file in enumerate(IMAGE_FILENAMES):
                image = cv2.imread(file)
                if image is None:
                    print(f"Could not read image {file}")
                    continue
                    
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = selfie_segmentation.process(rgb_image)

                fg_image = np.zeros(image.shape, dtype=np.uint8)
                fg_image[:] = MASK_COLOR
                bg_image = np.zeros(image.shape, dtype=np.uint8)
                bg_image[:] = BG_COLOR
                
                condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
                mask_output = np.where(condition, fg_image, bg_image)
                
                if iteration == 9:  # Only save on the last iteration
                    cv2.imwrite(f'{file}_mask.png', mask_output)
            
            end_time = time.time()
            execution_time_ms = (end_time - start_time) * 1000
            print(f"Iteration {iteration+1}: {execution_time_ms:.2f} ms")

process_static_images()
