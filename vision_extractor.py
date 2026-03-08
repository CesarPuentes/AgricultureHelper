import numpy as np
import cv2
from plantcv import plantcv as pcv

# Optional: Turn off plotting to save memory on the Raspberry Pi
pcv.params.debug = "None" 

def calculate_living_canopy(image_path):
    """
    Reads an image of a plant tray and calculates 'Living Canopy Coverage'
    using the Excess Green (ExG) Index (2G - R - B). 
    This automatically ignores dead/brown leaves and soil.
    """
    try:
        original_color_image, _, _ = pcv.readimage(filename=image_path)
        
        # 1. Split into B, G, R channels (OpenCV loads as BGR)
        blue_channel, green_channel, red_channel = cv2.split(original_color_image)
        
        # 2. ExG Math requires slightly larger data types to prevent overflow/underflow
        blue_channel, green_channel, red_channel = blue_channel.astype(float), green_channel.astype(float), red_channel.astype(float)
        
        # 3. Calculate ExG: 2*Green - Red - Blue
        excess_green_index = (2 * green_channel) - red_channel - blue_channel
        
        # 4. Normalize and convert back to 8-bit image for thresholding
        excess_green_index = np.clip(excess_green_index, 0, 255).astype(np.uint8)
        
        # 5. Threshold: Any value above 20 is solid living green tissue
        living_tissue_binary_mask = pcv.threshold.binary(gray_img=excess_green_index, threshold=20, object_type="light")
        noise_reduced_mask = pcv.fill(bin_img=living_tissue_binary_mask, size=50)
        
        # 6. Math: Calculate coverage
        total_pixels = noise_reduced_mask.size
        living_plant_pixels = np.count_nonzero(noise_reduced_mask)
        living_canopy_percentage = (living_plant_pixels / total_pixels) * 100
        
        return round(living_canopy_percentage, 2)

    except Exception as e:
        print(f"Error processing camera image: {e}")
        return None

# --- Simulation (Hourly Cron Job) ---
if __name__ == "__main__":
    test_images = ["Gemini_clorosis.png", "Gemini_sana1.png", "Gemini_clorosis2.png", 
                   "Gemini_sana2.png", "Gemini_clorosis3.png", "Gemini_sana3.png"]

    for image_file in test_images:
        living_canopy_coverage = calculate_living_canopy(image_file)
        if living_canopy_coverage is not None:
            print(f"[{image_file}] Living Canopy Coverage (ExG): {living_canopy_coverage}%")