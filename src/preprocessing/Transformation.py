import cv2
import numpy as np
import os
import argparse
from plantcv import plantcv as pcv
import matplotlib.pyplot as plt

#baya chatgpt attim duzeltecegim koc
# Function to apply transformations
def apply_transformations(image_path, destination_folder):
    image = cv2.imread(image_path)
    
    # Transformations
    transformations = {
        "GaussianBlur": cv2.GaussianBlur(image, (5, 5), 0),
        "Mask": apply_mask(image),
        "ROI": apply_roi(image),
        "AnalyzeObject": analyze_object(image),
        "Pseudolandmarks": apply_pseudolandmarks(image),
        "ColorHistogram": plot_color_histogram(image)
    }
    
    # Save and display the transformations
    for name, transformed_img in transformations.items():
        output_path = os.path.join(destination_folder, f"{os.path.basename(image_path).split('.')[0]}_{name}.jpg")
        if name == "ColorHistogram":
            plt.hist(cv2.cvtColor(image, cv2.COLOR_BGR2RGB).ravel(), bins=256, histtype='step', color='black')
            plt.title('Color Histogram')
            plt.xlabel('Pixel Intensity')
            plt.ylabel('Frequency')
            plt.savefig(output_path)
            plt.close()
        else:
            cv2.imwrite(output_path, transformed_img)
        
        # Display image for reference
        cv2.imshow(f"{name} - {os.path.basename(image_path)}", transformed_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Example transformations
def apply_mask(image):
    # You can apply a simple threshold mask here (change based on your needs)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    return cv2.bitwise_and(image, image, mask=mask)

def apply_roi(image):
    # Define a region of interest (e.g., square section of the image)
    height, width = image.shape[:2]
    roi = image[int(height/4):int(3*height/4), int(width/4):int(3*width/4)]
    return roi

def analyze_object(image):
    # This is a placeholder for a more complex analysis (e.g., contour detection)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = image.copy()
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)  # Draw contours on the image
    return result

def apply_pseudolandmarks(image):
    # Example pseudolandmarks: Find and mark specific keypoints (e.g., leaf tip, leaf edges)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY)
    keypoints = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    
    # For demonstration, marking the contours as "landmarks"
    for contour in keypoints:
        for point in contour:
            cv2.circle(image, tuple(point[0]), 5, (0, 0, 255), -1)  # Mark the points with a red circle
    return image

def plot_color_histogram(image):
    hist = cv2.calcHist([image], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    return hist

def main():
    parser = argparse.ArgumentParser(description='Apply image transformations for leaf disease recognition.')
    parser.add_argument('src', type=str, help='Source image or directory of images.')
    parser.add_argument('dst', type=str, help='Destination directory to save transformed images.')
    args = parser.parse_args()
    
    # Check if src is a directory
    if os.path.isdir(args.src):
        for filename in os.listdir(args.src):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(args.src, filename)
                apply_transformations(image_path, args.dst)
    else:
        apply_transformations(args.src, args.dst)

if __name__ == '__main__':
    main()
