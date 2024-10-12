import sys
sys.path.append('../')
from utils import get_center_of_bbox, measure_distance


class PlayerBallAssigner():
    def __init__(self):
        # Maximale Distanz, innerhalb derer der Ball einem Spieler zugeordnet wird
        self.max_player_ball_distance = 70

    def assign_ball_to_player(self, players, ball_bbox):
        # Die Position des Balls wird anhand des Mittelpunkts der Ball-Bounding-Box ermittelt
        ball_position = get_center_of_bbox(ball_bbox)

        # Initiale Werte für die minimalste Distanz und den zugeordneten Spieler
        minimum_distance = 99999
        assigned_player = -1

        # Für jeden Spieler wird überprüft, ob der Ball in dessen Nähe ist
        for player_id, player in players.items():
            player_bbox = player['bbox']

            # Distanz vom linken und rechten Fuß des Spielers zum Ball wird berechnet (BBOX unten links und rechts)
            distance_left = measure_distance((player_bbox[0], player_bbox[-1]), ball_position)
            distance_right = measure_distance((player_bbox[2], player_bbox[-1]), ball_position)
            # Die geringere der beiden Distanzen wird gewählt
            distance = min(distance_left, distance_right)

            # Wenn die Distanz kleiner als die maximale Zuweisungsdistanz ist und die geringste bisherige Distanz unterschreitet,
            # wird der Spieler als Ballbesitzer gesetzt
            if distance < self.max_player_ball_distance:
                if distance < minimum_distance:
                    minimum_distance = distance
                    assigned_player = player_id

        # Der Spieler, der dem Ball am nächsten ist (und innerhalb der maximalen Distanz liegt), wird zurückgegeben
        return assigned_player
