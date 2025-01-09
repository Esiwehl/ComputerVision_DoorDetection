import cv2
import numpy as np
import os

def load_image(image_path):
    """
    Laad een afbeelding vanaf het opgegeven bestandspad.
    """
    img = cv2.imread(image_path)
    return img

def load_templates(template_folder):
    """
    Laad alle templates (sub-afbeeldingen) vanuit de opgegeven map.
    """
    templates = {}
    for filename in os.listdir(template_folder):
        if filename.endswith('.png'):
            name = filename.split('.')[0]
            template_path = os.path.join(template_folder, filename)
            template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

            # Maak varianten van de template (origineel, gespiegeld en geroteerd)
            template_variants = [
                template,
                cv2.flip(template, 1),
                cv2.flip(template, 0),
                cv2.rotate(template, cv2.ROTATE_90_CLOCKWISE),
                cv2.rotate(template, cv2.ROTATE_180),
                cv2.rotate(template, cv2.ROTATE_90_COUNTERCLOCKWISE)
            ]
            templates[name] = template_variants
    return templates

def non_max_suppression(detections, overlap_thresh=0.3):
    """
    Pas Niet-Maximum Suppression toe om overlappende detecties te filteren.

    Parameters:
    - detections: lijst van detecties, elk met een 'bbox' sleutel die een tuple bevat (x1, y1, x2, y2)
    - overlap_thresh: drempel voor overlap, typisch tussen 0.3 en 0.7

    Retourneert:
    - filtered_detections: lijst van gefilterde detecties zonder overlappende bounding boxes
    """
    if len(detections) == 0:
        return []

    # Haal de bounding boxes en coördinaten op
    boxes = np.array([d["bbox"] for d in detections])
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]
    scores = np.array([1.0] * len(detections))  # Bij template matching hebben we geen score, dus vullen we het in met 1

    # Bereken de oppervlakte van elke bounding box
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)
    order = scores.argsort()[::-1]  # Sorteer op basis van scores (optioneel in dit geval)

    filtered_detections = []
    while len(order) > 0:
        i = order[0]
        filtered_detections.append(detections[i])

        xx1 = np.maximum(start_x[i], start_x[order[1:]])
        yy1 = np.maximum(start_y[i], start_y[order[1:]])
        xx2 = np.minimum(end_x[i], end_x[order[1:]])
        yy2 = np.minimum(end_y[i], end_y[order[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # Bereken de overlap ratio
        overlap = (w * h) / areas[order[1:]]

        # Verwijder overlappende bounding boxes boven de drempel
        order = np.delete(order, np.concatenate(([0], np.where(overlap > overlap_thresh)[0] + 1)))

    return filtered_detections

def detect_objects_with_edges(image, templates, thresholds, debug=False):
    """
    Detecteer objecten in de afbeelding door eerst Canny edge-detection toe te passen
    en vervolgens template matching alleen in de gebieden met randen.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detections = []
    debug_image = image.copy()

    # Stap 1: Pas Canny edge-detection toe
    edges = cv2.Canny(gray_image, 100, 200)

    # Optioneel: Toon de randen voor debugging
    if debug:
        cv2.imshow("Canny Edges", edges)
        cv2.waitKey(500)

    # Stap 2: Zoek contouren in de randafbeelding
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Stap 3: Definieer ROI's rond de contouren en pas alleen daar template matching toe
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        roi = gray_image[y:y + h, x:x + w]  # Alleen in dit gebied zoeken

        # Pas template matching toe binnen de ROI
        for name, template_variants in templates.items():
            for variant in template_variants:
                for scale in np.linspace(0.8, 1.2, 3):  # Beperkt bereik van schalen
                    resized_template = cv2.resize(variant, None, fx=scale, fy=scale)

                    # Zorg ervoor dat de geschaalde template in de ROI past
                    if resized_template.shape[0] > roi.shape[0] or resized_template.shape[1] > roi.shape[1]:
                        continue

                    # Pas template matching toe
                    result = cv2.matchTemplate(roi, resized_template, cv2.TM_CCOEFF_NORMED)
                    threshold = thresholds.get(name, 0.9)  # Hogere threshold voor striktere matching
                    loc = np.where(result >= threshold)  # Voor TM_CCOEFF_NORMED gebruik >= threshold

                    for pt in zip(*loc[::-1]):
                        # Bereken de positie in de originele afbeelding
                        x1, y1 = pt[0] + x, pt[1] + y
                        x2, y2 = x1 + resized_template.shape[1], y1 + resized_template.shape[0]
                        bbox = (x1, y1, x2, y2)
                        detections.append({'name': name, 'bbox': bbox})

                        if debug:
                            print(f"Detected {name} at {bbox} with scale {scale}")
                            cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(debug_image, f"{name} ({scale:.2f})", (x1, y1 - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # Toon debug afbeelding met alle gedetecteerde templates if true
    if debug:
        cv2.imshow("Debug - All Detected Templates with Canny Edges", debug_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite("all_detected_templates_with_canny.jpeg", debug_image)

    # Pas Niet-Maximum Suppression toe om overlappende detecties te filteren
    filtered_detections = non_max_suppression(detections, overlap_thresh=0.6)

    # Teken de gefilterde detecties op de originele afbeelding
    for detection in filtered_detections:
        x1, y1, x2, y2 = detection["bbox"]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(image, detection["name"], (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return filtered_detections

def main():
    """
    Hoofdfunctie om de objectdetectie en classificatie uit te voeren.
    """
    image_path = "../data/SomeImage.jpg"
    template_folder = "templates"

    image = load_image(image_path)
    templates = load_templates(template_folder)

    thresholds = {
        'brievenbus': 0.50,
        'deur': 0.1,
        'deurbel': 0.85,
        'deurklink': 0.60,
        'slot': 0.60,
        'glaspaneel': 0.15
    }

    detections = detect_objects_with_edges(image, templates, thresholds, debug=True)
    cv2.imshow("Gedetecteerde Objecten", image)
    cv2.imwrite('output.jpeg', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
