def get_center_of_bbox(bbox):
    # Der Mittelpunkt (Center) des Bounding-Box wird berechnet, indem die Durchschnittswerte der X- und Y-Koordinaten
    # von den beiden gegenüberliegenden Ecken des Rechtecks verwendet werden.
    # Dies ist der Mittelwert der X-Werte (x1, x2) und Y-Werte (y1, y2).
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) /2)

def get_bbox_width(bbox):
    # Die Breite des Bounding-Box wird berechnet, indem die Differenz der X-Koordinaten (x2 - x1) ermittelt wird.
    return bbox[2] - bbox[0]

def measure_distance(p1, p2):
    # Der euklidische Abstand zwischen zwei Punkten wird berechnet.
    # Formel: sqrt((x2 - x1)^2 + (y2 - y1)^2) = Satz des Pythagoras
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5