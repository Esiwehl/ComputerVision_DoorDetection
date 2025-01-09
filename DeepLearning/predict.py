import os
import cv2
import pandas as pd
from dotenv import load_dotenv
from roboflow import Roboflow

def initialize_roboflow(api_key, project_name, version):
    """
    Initialiseer de Roboflow API en laad het model.

    Parameters:
        api_key (str): De API-sleutel voor toegang tot Roboflow.
        project_name (str): Naam van het Roboflow-project.
        version (int): Versienummer van het model.

    Returns:
        model: Het geladen Roboflow-model.
    """
    rf = Roboflow(api_key=api_key)
    project = rf.workspace().project(project_name)
    return project.version(version).model


def resize_image(image_path, max_width=1024, max_height=1024):
    """
    Verklein een afbeelding tot een maximale breedte en hoogte.

    Parameters:
        image_path (str): Pad naar de afbeelding.
        max_width (int): Maximale breedte van de afbeelding.
        max_height (int): Maximale hoogte van de afbeelding.

    Returns:
        str: Pad naar de verkleinde afbeelding.
    """
    img = cv2.imread(image_path)
    height, width = img.shape[:2]

    if width > max_width or height > max_height:
        scale = min(max_width / width, max_height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        resized_path = f"resized_{os.path.basename(image_path)}"
        cv2.imwrite(resized_path, resized_img)
        return resized_path
    return image_path

def extract_main_color(image):
    """
    Analyseer de hoofdkleur van een afbeelding door deze te verkleinen tot één pixel.

    Parameters:
        image (numpy.ndarray): De afbeelding om te analyseren.

    Returns:
        list: Een RGB-waarde van de hoofdkleur.
    """
    resized = cv2.resize(image, (1, 1), interpolation=cv2.INTER_AREA)
    return resized[0, 0].tolist()


def create_directories(base_output_dir, annotated_image_name):
    """
    Maak de benodigde directories voor outputbestanden.

    Parameters:
        base_output_dir (str): Hoofdmap voor outputbestanden.
        annotated_image_name (str): Naam van de geannoteerde afbeelding.

    Returns:
        tuple: Paden voor de hoofdoutputmap en de objectuitknipmap.
    """
    output_dir = os.path.join(base_output_dir, annotated_image_name)
    object_cutout_dir = os.path.join(output_dir, f"cutouts")
    os.makedirs(object_cutout_dir, exist_ok=True)
    return output_dir, object_cutout_dir


def process_predictions(predictions, copy_image, object_cutout_dir):
    """
    Verwerk de voorspellingen en genereer metadata en objectuitsneden.

    Parameters:
        predictions (list): Een lijst met voorspellingen van het model.
        copy_image (numpy.ndarray): Een kopie van de originele afbeelding.
        object_cutout_dir (str): Map waarin de objectuitsneden worden opgeslagen.

    Returns:
        list: Metadata voor alle gedetecteerde objecten.
    """
    rows = []
    for detected_obj in predictions:
        x, y, w, h = detected_obj['x'], detected_obj['y'], detected_obj['width'], detected_obj['height']
        class_name = detected_obj['class']
        confidence = detected_obj['confidence']

        # Bereken bounding box-coördinaten
        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)

        # Crop het object
        cropped_obj = copy_image[y1:y2, x1:x2]
        if cropped_obj.size == 0:
            continue

        # Analyseer hoofdkleur
        main_color = extract_main_color(cropped_obj)

        # Bereken breedte/hoogte-verhouding
        aspect_ratio = round(w / h, 2)

        # Bestandslocatie voor het uitgesneden object
        cropped_image_name = f"{class_name}_{x1}_{y1}.jpg"
        cropped_image_path = os.path.join(object_cutout_dir, cropped_image_name)
        cv2.imwrite(cropped_image_path, cropped_obj)

        # Voeg objectgegevens toe aan metadata lijst
        rows.append({
            "annotated_image": "",  # Leeg voor objecten
            "object_image": cropped_image_path,
            "object_type": class_name,
            "confidence": confidence,
            "position_x1": x1,
            "position_y1": y1,
            "position_x2": x2,
            "position_y2": y2,
            "main_color": main_color[::-1],
            "aspect_ratio": aspect_ratio,
        })
    return rows


def annotate_image(original_image, predictions, color_dict, default_color=(255, 255, 255)):
    """
    Voeg annotaties toe aan de originele afbeelding.

    Parameters:
        original_image (numpy.ndarray): De originele afbeelding.
        predictions (list): Een lijst met voorspellingen van het model.
        color_dict (dict): Een dictionary met kleuren per objecttype.
        default_color (tuple): De standaardkleur als het objecttype niet in color_dict staat.

    Returns:
        numpy.ndarray: De afbeelding met annotaties.
    """
    for detected_obj in predictions:
        x, y, w, h = detected_obj['x'], detected_obj['y'], detected_obj['width'], detected_obj['height']
        class_name = detected_obj['class']
        confidence = detected_obj['confidence']

        # Bereken bounding box
        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)

        # Haal de kleur op uit de dictionary
        color = color_dict.get(class_name, default_color)

        # Teken bounding box en label
        cv2.rectangle(original_image, (x1, y1), (x2, y2), color, 2)
        label_text = f"{class_name} ({confidence:.2f})"
        cv2.putText(original_image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return original_image


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


# *** Hoofdscript ***
def main():
    """
    Hoofdfunctie voor het uitvoeren van objectdetectie, annotaties en opslag van metadata.
    """
    # Configuratie
    load_dotenv()

    API_KEY = os.getenv("API_KEY")
    if not API_KEY:
        raise ValueError("API_KEY niet gevonden in .env bestand.")
    PROJECT_NAME = "voordeuren-amsterdam"
    VERSION = 4
    IMAGE_PATH= "../data/SomeImage.jpg"
    BASE_OUTPUT_DIR = "detected_objects"

    # Definieer kleuren per objecttype
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

    # Initialiseer model
    model = initialize_roboflow(API_KEY, PROJECT_NAME, VERSION)

    # resize afbeelding
    resized_image_path = resize_image(IMAGE_PATH)

    # Laad resized afbeelding
    original_image = cv2.imread(resized_image_path)
    copy_image = original_image.copy()
    annotated_image_name = os.path.basename(resized_image_path).split(".")[0]

    # Maak outputmappen
    output_dir, object_cutout_dir = create_directories(BASE_OUTPUT_DIR, annotated_image_name)

    # Voer voorspellingen uit
    response = model.predict(resized_image_path, confidence=30, overlap=20).json()
    predictions = response['predictions']

    # Verwerk voorspellingen en genereer metadata
    rows = process_predictions(predictions, copy_image, object_cutout_dir)

    # Voeg geannoteerde afbeelding toe aan metadata
    annotated_image_path = os.path.join(output_dir, "annotated_image.jpg")
    rows.insert(0, {
        "annotated_image": annotated_image_path,
        "object_image": "",
        "object_type": "",
        "confidence": "",
        "position_x1": "",
        "position_y1": "",
        "position_x2": "",
        "position_y2": "",
        "main_color": "",
        "aspect_ratio": "",
    })

    # Annoteren en opslaan van de afbeelding
    annotated_image = annotate_image(original_image, predictions, COLOR_DICT)
    cv2.imwrite(annotated_image_path, annotated_image)

    # Voeg nieuwe metadata toe aan Excel
    output_excel_path = os.path.join(BASE_OUTPUT_DIR, "detections.xlsx")
    new_data = pd.DataFrame(rows)
    append_to_excel(output_excel_path, new_data)

    # Eindmelding
    print(f"Geannoteerde afbeelding opgeslagen in: {annotated_image_path}")
    print(f"Objectuitsneden opgeslagen in map: {object_cutout_dir}")
    print(f"Metadata opgeslagen in Excel-bestand: {output_excel_path}")


# Start het script
if __name__ == "__main__":
    main()
