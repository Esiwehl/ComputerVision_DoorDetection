"""
Object Detection Script

Dit script detecteert objecten zoals deuren en ramen in afbeeldingen met behulp van contourdetectie.
Het slaat geannoteerde afbeeldingen en uitgesneden objecten op in een georganiseerde mapstructuur.
Een database van gedetecteerde objecten wordt geëxporteerd naar een Excel-bestand.
"""

import cv2
import os
import time
import numpy as np
import pandas as pd
from torch.utils.tensorboard.summary import image_boxes

COLOR_DICT = {
        "balkon": (0, 128, 128),
        "brievenbus": (128, 128, 0),
        "deur": (0, 0, 255),
        "deurbel": (255, 255, 0),
        "deurklink": (0, 255, 255),
        "deurmat": (255, 0, 255),
        "glaspaneel": (0, 255, 0),
        "lamp": (128, 0, 128),
        "raam": (255, 0, 0),
        "trap": (128, 128, 128),
    }

def append_to_excel(file_path, new_data):
    """
    Voeg nieuwe gegevens toe aan een bestaande Excel-bestand of maak een nieuw bestand.

    Parameters:
        file_path (str): Pad naar het Excel-bestand.
        new_data (pandas.DataFrame): Nieuwe gegevens om toe te voegen.

    Returns:
        None
    """
    if os.path.exists(file_path):
        # Laad bestaande gegevens en voeg nieuwe gegevens toe
        existing_data = pd.read_excel(file_path)
        combined_data = pd.concat([existing_data, new_data], ignore_index=True)
    else:
        # Gebruik alleen nieuwe gegevens als het bestand niet bestaat
        combined_data = new_data

    # Sla het bijgewerkte bestand op
    combined_data.to_excel(file_path, index=False)


def save_cropped_objects(image, x, y, w, h, label, output_dir):
    """
    Slaat een uitgesneden object op en retourneert de metadata.

    Parameters:
        image (numpy.ndarray): Originele afbeelding.
        x, y, w, h (int): Coördinaten en grootte van het object.
        label (str): Label van het object ('deur' of 'raam').
        output_dir (str): Directory waar de uitgesneden objecten worden opgeslagen.

    Returns:
        dict: Metadata van het uitgesneden object.
    """
    cutout_dir = os.path.join(output_dir, "cutouts")
    os.makedirs(cutout_dir, exist_ok=True)

    # Crop het object
    cropped_img = image[y:y + h, x:x + w]
    cropped_img_name = f"{label}_{x}_{y}.jpg"
    cropped_img_path = os.path.join(cutout_dir, cropped_img_name)
    cv2.imwrite(cropped_img_path, cropped_img)

    # Bereken gemiddelde kleur (RGB)
    avg_color_per_row = np.average(cropped_img, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0).astype(int)  # BGR
    avg_color_rgb = avg_color[::-1]  # Omzetten naar RGB

    return {
        "label": label,
        "positie_x1": x,
        "positie_y1": y,
        "positie_x2": x + w,
        "positie_y2": y + h,
        "breedte": w,
        "hoogte": h,
        "aspect_ratio": w / h,
        "average_color_rgb": avg_color_rgb.tolist(),
        "object_image_path": cropped_img_path,
    }


def filter_contours(edges, image, image_name, output_dir):
    """
    Detecteert contouren en slaat de geannoteerde afbeelding en metadata op.

    Parameters:
        edges (numpy.ndarray): Binaire afbeelding van randen.
        image (numpy.ndarray): Originele afbeelding.
        image_name (str): Naam van de afbeelding (zonder extensie).
        output_dir (str): Directory waar de resultaten worden opgeslagen.

    Returns:
        list: Metadata van alle gedetecteerde objecten.
    """
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_image = image.copy()
    results = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Filter objecten op grootte
        if w > 50 and h > 100:
            aspect_ratio = float(w) / h

            # Label bepalen
            if aspect_ratio < 0.5:
                label = "deur"
            elif 0.5 <= aspect_ratio < 1.5:
                label = "raam"
            else:
                continue

            kleur = COLOR_DICT.get(label, (255, 255, 255))

            # Teken bounding box en label op de afbeelding
            cv2.rectangle(contour_image, (x, y), (x + w, y + h), kleur, 2)
            cv2.putText(
                contour_image,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                kleur,
                2,
            )

            # Sla metadata en uitgesneden object op
            result = save_cropped_objects(image, x, y, w, h, label, output_dir)
            results.append(result)

    # Sla de geannoteerde afbeelding op
    annotated_image_path = os.path.join(output_dir, f"{image_name}_annotated.jpg")
    cv2.imwrite(annotated_image_path, contour_image)

    return results


def detect_doors_with_sobel(image):
    """
    Voert Sobel-detectie uit en genereert een binaire afbeelding met randen.

    Parameters:
        image (numpy.ndarray): Originele afbeelding.

    Returns:
        numpy.ndarray: Binaire afbeelding met randen.
    """
    start_time = time.time()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite('Gray.png', gray)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    # cv2.imwrite('Gaussian_.png', blurred)
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(sobel_x, sobel_y)
    sobel_image = np.uint8(np.clip(magnitude, 0, 255))
    # cv2.imwrite('Sobel_.png', sobel_image)

    edges = cv2.Canny(sobel_image, 100, 250)
    # cv2.imwrite('Canny_.png', edges)

    print(f"Detectie voltooid in {time.time() - start_time:.2f} seconden")
    return edges


def process_image(image_path, results):
    """
    Verwerkt een enkele afbeelding en slaat de resultaten op.

    Parameters:
        image_path (str): Pad naar de afbeelding.
        results (list): Lijst waarin metadata van gedetecteerde objecten wordt opgeslagen.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Kon afbeelding niet laden: {image_path}")
        return

    image_name = os.path.basename(image_path).split('.')[0]
    output_dir = os.path.join("detected_objects", image_name)
    os.makedirs(output_dir, exist_ok=True)

    edges = detect_doors_with_sobel(image)
    image_results = filter_contours(edges, image, image_name, output_dir)

    for res in image_results:
        res["afbeelding"] = image_name
    results.extend(image_results)


def main():
    """
    Hoofdfunctie om afbeeldingen te verwerken en een Excel-database te genereren.
    """

    image_path = "../data/SomeImage.jpg"

    results = []
    process_image(image_path, results)

    results_df = pd.DataFrame(results)

    append_to_excel("conventioneel_detected_objects.xlsx", results_df)
    print("Export compleet. Resultaten opgeslagen in conventioneel_detected_objects.xlsx")


if __name__ == "__main__":
    main()
