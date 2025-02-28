import os
import sys
import matplotlib.pyplot as plt
from collections import Counter

def analyze_directory(directory_path):
    if not os.path.exists(directory_path):
        print(f"Directory {directory_path} does not exist.")
        sys.exit(1)

    subdirectories = [d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))]
    class_counts = Counter()
    image_counts_per_class = []
    for subdirectory in subdirectories:
        plant_type = subdirectory.split("/")[-1]
        image_count = len(os.listdir(os.path.join(directory_path, subdirectory)))
        class_counts[plant_type] += image_count
        image_counts_per_class.append(image_count)

    return class_counts, image_counts_per_class

def generate_charts(class_counts, directory_name):
    labels = class_counts.keys()
    sizes = class_counts.values()

    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title(f"{directory_name} Class Distribution")
    plt.axis('equal')
    plt.show()

    plt.figure(figsize=(14, 6))
    plt.bar(class_counts.keys(), class_counts.values())
    plt.xlabel('Plant Type')
    plt.ylabel('Image Count')
    plt.title(f"{directory_name} Class Distribution Bar Chart")
    plt.show()


def main(directory_path):
    directory_name = os.path.basename(directory_path)
    class_counts, _ = analyze_directory(directory_path)
    generate_charts(class_counts, directory_name)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide a directory as an argument.")
        sys.exit(1)

    directory_path = sys.argv[1]
    main(directory_path)
