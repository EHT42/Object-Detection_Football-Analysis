import cv2
from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import sys
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width

class Tracker:
    def __init__(self, model_path):
        # YOLO-Modell wird mit dem übergebenen Modellpfad initialisiert
        self.model = YOLO(model_path)
        # ByteTrack wird zur Objektverfolgung verwendet
        self.tracker = sv.ByteTrack()

    def interpolate_ball_positions(self, ball_positions):
        # Bounding Boxes des Balls werden extrahiert
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        # Die Daten werden in ein DataFrame konvertiert, um fehlende Werte zu interpolieren
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1','y1','x2','y2'])

        # Fehlende Werte werden interpoliert und anschließend werden restliche Lücken gefüllt
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        # Die interpolierten Werte werden wieder konvertiert
        ball_positions = [{1: {'bbox':x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def detect_frames(self, frames):
        # Stapelgröße wird für die Verarbeitung der Frames festgelegt
        batch_size = 20
        detections = []
        # Frames werden in Stapeln verarbeitet, um Objekte zu erkennen
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        # Wenn ein Stub-Pfad angegeben und die Datei vorhanden ist, werden die Tracks geladen
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)

        # Objekterkennung wird auf den Frames durchgeführt
        detections = self.detect_frames(frames)

        # Ein leeres Dictionary zur Verfolgung von Spielern, Schiedsrichtern und dem Ball wird erstellt
        tracks={
            'players': [],
            'referees': [],
            'ball': []
        }

        for frame_num, detection in enumerate(detections):
            # Klassennamen aus den Detektionen werden abgerufen
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Torhüter werden als "Spieler" klassifiziert
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == 'goalkeeper':
                    detection_supervision.class_id[object_ind] = cls_names_inv['player']

            # Tracks werden basierend auf den erkannten Objekten aktualisiert
            detections_with_tracks = self.tracker.update_with_detections(detection_supervision)

            # Leere Einträge für Spieler, Schiedsrichter und Ball für das aktuelle Frame werden erstellt
            tracks['players'].append({})
            tracks['referees'].append({})
            tracks['ball'].append({})

            # Detections werden verarbeitet, und die Bounding Box sowie die Track-ID werden gespeichert
            for frame_detection in detections_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                # Spieler werden zu den Tracks hinzugefügt
                if cls_id == cls_names_inv['player']:
                    tracks['players'][frame_num][track_id] = {'bbox': bbox}
                # Schiedsrichter werden zu den Tracks hinzugefügt
                if cls_id == cls_names_inv['referee']:
                    tracks['referees'][frame_num][track_id] = {'bbox': bbox}

            # Ball-Detektion wird separat verarbeitet und hinzugefügt
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks['ball'][frame_num][1] = {'bbox': bbox}

        # Wenn ein Stub-Pfad vorhanden ist, werden die Tracks als Datei gespeichert
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        # Eine Ellipse wird um das erkannte Objekt (z.B. Spieler oder Schiedsrichter) gezeichnet
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(frame,
                    center=(x_center, y2),
                    axes=(int(width), int(0.35*width)),
                    angle=0.0,
                    startAngle=-45,
                    endAngle=235,
                    color=color,
                    thickness=2,
                    lineType=cv2.LINE_4
                    )

        # Ein Rechteck zur Beschriftung mit der Track-ID wird gezeichnet
        reactangle_width = 40
        reactangle_height = 20
        x1_rect = x_center - reactangle_width / 2
        x2_rect = x_center + reactangle_width / 2
        y1_rect = (y2 - reactangle_height // 2) + 15
        y2_rect = (y2 + reactangle_height // 2) + 15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect), int(y1_rect)),
                          (int(x2_rect), int(y2_rect)),
                          color,
                          cv2.FILLED)

            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10

            cv2.putText(frame,
                        f'{track_id}',
                        (int(x1_text), int(y1_rect+15)),
                        cv2.FONT_ITALIC,
                        0.6,
                        (0,0,0),
                        2
                        )

        return frame

    def draw_triangle(self, frame, bbox, color):
        # Ein Dreieck wird über der angegebenen Position gezeichnet
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        triangle_points = np.array([[x,y],[x-10, y-20], [x+10, y-20]])

        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0,0,0), 2)

        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        # Die Ballbesitzstatistik der Teams wird dargestellt
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 1000), (255, 255, 255, cv2.FILLED))
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        team_ball_control_while_frame = team_ball_control[:frame_num + 1]

        # Ballbesitz für jedes Team wird berechnet
        team_1_num_frames = team_ball_control_while_frame[team_ball_control_while_frame == 1].shape[0]
        team_2_num_frames = team_ball_control_while_frame[team_ball_control_while_frame == 2].shape[0]
        team_1 = team_1_num_frames/(team_1_num_frames+team_2_num_frames)
        team_2 = team_2_num_frames/(team_1_num_frames+team_2_num_frames)

        # Die Ballbesitzstatistik wird angezeigt
        cv2.putText(frame, f'Team 1 Ball Control: {team_1 * 100:.2f}%',(1400, 900), cv2.FONT_ITALIC, 1, (0, 0, 0), 3)
        cv2.putText(frame, f'Team 1 Ball Control: {team_2 * 100:.2f}%', (1400, 950), cv2.FONT_ITALIC, 1,(0, 0, 0), 3)

        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control):
        # Alle Annotationen (Spieler, Schiedsrichter, Ballbesitz etc.) werden zu den Frames hinzugefügt
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks['players'][frame_num]
            referee_dict = tracks['referees'][frame_num]
            ball_dict = tracks['ball'][frame_num]

            # Spieler-Ellipsen werden gezeichnet
            for track_id, player in player_dict.items():
                color = player.get('team_color', (0, 0, 255))
                frame = self.draw_ellipse(frame, player['bbox'], color, track_id)

                # Falls der Spieler den Ball hat, wird ein Dreieck gezeichnet
                if player.get('has_ball', False):
                    frame = self.draw_triangle(frame, player['bbox'], (0,0,255))

            # Schiedsrichter-Ellipsen werden gezeichnet
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee['bbox'], (0, 255, 255))

            # Ball-Dreieck wird gezeichnet
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball['bbox'], (0,255,0))

            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_video_frames.append(frame)

        return output_video_frames