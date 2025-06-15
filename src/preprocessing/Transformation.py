import cv2
import os
import argparse
import matplotlib.pyplot as plt


def apply_transformations(image_path, destination_folder):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return

    os.makedirs(destination_folder, exist_ok=True)

    transformed_images = {}
    transformed_images["GaussianBlur"] = cv2.GaussianBlur(image, (5, 5), 0)
    transformed_images["Mask"] = apply_mask(image)
    transformed_images["ROI"] = apply_roi(image)
    transformed_images["AnalyzeObject"] = analyze_object(image)
    transformed_images["Pseudolandmarks"] = apply_pseudolandmarks(image.copy())

    for name, transformed_img in transformed_images.items():
        a = f"{os.path.basename(image_path).split('.')[0]}_{name}.jpg"
        output_path = os.path.join(destination_folder, a)
        cv2.imwrite(output_path, transformed_img)

        cv2.imshow(f"{name} - {os.path.basename(image_path)}", transformed_img)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()

    c_hist = f"{os.path.basename(image_path).split('.')[0]}_ColorHistogram.jpg"
    output_path = os.path.join(destination_folder, c_hist)
    plot_color_histogram(image, output_path)


def apply_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    return cv2.bitwise_and(image, image, mask=mask)


def apply_roi(image):
    height, width = image.shape[:2]
    roi = image[int(height/4):int(3*height/4), int(width/4):int(3*width/4)]
    return roi


def analyze_object(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    result = image.copy()
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
    return result


def apply_pseudolandmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        for point in contour:
            cv2.circle(image, tuple(point[0]), 5, (0, 0, 255), -1)
    return image


def plot_color_histogram(image, output_path):
    """Plot and save color histogram as an image file"""
    plt.figure(figsize=(10, 6))
    colors = ('b', 'g', 'r')

    for i, color in enumerate(colors):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=color)

    plt.title('Color Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.xlim([0, 256])
    plt.savefig(output_path)
    plt.close()


def get_transformed_images(image):
    if image is None:
        raise ValueError("Input image is None")

    transformed_images = {
        "Original": image.copy(),
        "GaussianBlur": cv2.GaussianBlur(image, (5, 5), 0),
        "Mask": apply_mask(image),
        "ROI": apply_roi(image),
        "AnalyzeObject": analyze_object(image),
        "Pseudolandmarks": apply_pseudolandmarks(image.copy())
    }

    return transformed_images


def main():
    desc = 'Apply image transformations for leaf disease recognition.'
    parser = argparse.ArgumentParser(description=desc)
    help1 = 'Source image or directory of images.'
    parser.add_argument('src', type=str, help=help1)
    help2 = 'Destination directory to save transformed images.'
    parser.add_argument('dst', type=str, help=help2)
    args = parser.parse_args()

    if os.path.isdir(args.src):
        for filename in os.listdir(args.src):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(args.src, filename)
                apply_transformations(image_path, args.dst)
    else:
        apply_transformations(args.src, args.dst)


if __name__ == '__main__':
    main()
