from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self):
        # Zwei leere Dictionaries:
        # 'team_colors' speichert die Farben der Teams, 'player_team_dict' speichert, welchem Team ein Spieler zugeordnet wurde.
        self.team_colors = {}
        self.player_team_dict = {}

    def get_clustering_model(self, image):
        # Bild wird in ein 2D-Array umgeformt, bei dem jede Zeile die RGB-Werte eines Pixels enthält
        image_2d = image.reshape(-1, 3)

        # K-Means Clustering wird mit zwei Clustern ausgeführt, um zwei Hauptfarben (z.B. Teamfarben) zu erkennen
        kmeans = KMeans(n_clusters=2, init='k-means++', n_init=1)
        kmeans.fit(image_2d)

        return kmeans

    def get_player_color(self, frame, bbox):
        # Schneidet den Bereich des Bildes aus, der durch die 'Bounding Box' des Spielers definiert wird
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        # Nimmt nur die obere Hälfte des Bildes, da diese in der Regel den Körper des Spielers zeigt
        top_half_image = image[0:int(image.shape[0] / 2 ), :]

        # Führt K-Means Clustering auf die obere Bildhälfte durch, um die dominierenden Farben zu finden
        kmeans = self.get_clustering_model(top_half_image)

        # Die Labels jedes Pixels werden geholt, um zu sehen, zu welchem Cluster jeder Pixel gehört
        labels = kmeans.labels_

        # Das clustered Image wird zurück in die normale Form gebracht
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])

        # Die Pixel in den Ecken des Bildes werden verwendet, um den Hintergrundcluster zu identifizieren,
        # da dieser normalerweise die dominierende Farbe außerhalb des Spielers ist.
        corner_clusters = [clustered_image[0, 0], clustered_image[0, 1], clustered_image[-1, 0], clustered_image[-1, 1]]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        # Der Spielercluster ist der andere Cluster, da es nur zwei Cluster gibt (Hintergrund und Spieler)
        player_cluster = 1 - non_player_cluster

        # Der Farbwert des Spielerclusters wird als Spielerfarbe zurückgegeben
        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color

    def assign_team_color(self, frame, player_detections):
        # Liste, um die Farben der Spieler zu speichern
        player_colors = []
        # Für jeden Spieler wird die 'Bounding Box' (bbox) verwendet, um die Spielerfarbe zu bestimmen
        for _, player_detection in player_detections.items():
            bbox = player_detection['bbox']
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)

        # Führt K-Means Clustering auf den gesammelten Spielerfarben durch, um die Spieler in zwei Teams zu gruppieren
        kmeans = KMeans(n_clusters=2, init='k-means++', n_init=1)
        kmeans.fit(player_colors)

        # Speichert K-Means Modell
        self.kmeans = kmeans

        # Die Teamfarben werden den Clustern zugeordnet
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]


    def get_player_team(self, frame, player_bbox, player_id):
        # Wenn der Spieler bereits einem Team zugeordnet wurde, wird dies zurückgegeben
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        # Ansonsten wird die Farbe des Spielers bestimmt
        player_color = self.get_player_color(frame, player_bbox)

        # K-Means wird verwendet, um vorherzusagen, zu welchem Cluster (Team) der Spieler gehört
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
        team_id += 1

        # Der Spieler wird dem entsprechenden Team zugeordnet
        self.player_team_dict[player_id] = team_id
        return team_id