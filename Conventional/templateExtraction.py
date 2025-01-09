"""
Template Extractie Script met Muisinteractie
============================================

Dit script maakt het mogelijk om templates direct uit een afbeelding te extraheren
door gebruik te maken van muisinteractie. Gebruikers kunnen een bounding box tekenen
op de afbeelding en deze een label geven. Het geselecteerde gebied wordt opgeslagen
als een nieuwe template in de juiste map.

Invoer: Hoofdafbeelding (in .png formaat)
Uitvoer: Opgeslagen template-afbeelding in de opgegeven directory.

Gebruikte bibliotheken:
- cv2: voor beeldverwerking
- numpy: voor matrixbewerkingen
"""

import cv2
import os

# Globale variabelen voor het opslaan van coördinaten en de status
ref_point = []
cropping = False


def click_and_crop(event, x, y, flags, param):
    """
    Callback-functie om de muisinteractie af te handelen en een bounding box te tekenen.

    Parameters:
    - event: type muisgebeurtenis
    - x, y: coördinaten van de muis
    - flags: extra parameters (wordt niet gebruikt)
    - param: extra parameters (wordt niet gebruikt)
    """
    global ref_point, cropping

    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point = [(x, y)]
        cropping = True

    elif event == cv2.EVENT_LBUTTONUP:
        ref_point.append((x, y))
        cropping = False

        # Teken de rechthoek op de afbeelding
        cv2.rectangle(image, ref_point[0], ref_point[1], (0, 255, 0), 2)
        cv2.imshow("Afbeelding", image)


def save_cropped_template(image, ref_point, label, output_dir):
    """
    Sla het geselecteerde gebied op als een nieuwe template met de gegeven labelnaam.

    Parameters:
    - image: originele afbeelding
    - ref_point: coördinaten van de bounding box [(x1, y1), (x2, y2)]
    - label: labelnaam voor het geselecteerde gebied
    - output_dir: map waar de template opgeslagen moet worden
    """
    if len(ref_point) == 2:
        interest_area = image[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:ref_point[1][0]]
        output_path = os.path.join(output_dir, f"{label}.png")
        cv2.imwrite(output_path, interest_area)
        print(f"Template opgeslagen als: {output_path}")
    else:
        print("Geen geldig selectiegebied.")


def main():
    """
    Hoofdfunctie voor het interactief selecteren en opslaan van templates.
    """
    global image

    image_path = "../data/SomeImage.png"
    output_dir = "templates"
    os.makedirs(output_dir, exist_ok=True)

    # Laad de afbeelding
    image = cv2.imread(image_path)
    clone = image.copy()
    cv2.namedWindow("Afbeelding")
    cv2.setMouseCallback("Afbeelding", click_and_crop)

    while True:
        cv2.imshow("Afbeelding", image)
        key = cv2.waitKey(1) & 0xFF

        # Druk op 'r' om te resetten
        if key == ord("r"):
            image = clone.copy()

        # Druk op 'c' om de selectie te labelen en op te slaan
        elif key == ord("c"):
            if len(ref_point) == 2:
                label = input("Voer een label in voor het geselecteerde object: ")
                save_cropped_template(clone, ref_point, label, output_dir)

        # Druk op 'q' om het programma te verlaten
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
