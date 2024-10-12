import cv2
from utils import save_video, read_video
from trackers import Tracker
from team_assigner import TeamAssigner
import numpy as np
from player_ball_assigner import PlayerBallAssigner


def main():

    # Das Video wird eingelesen
    video_frames = read_video('input_vids/08fd33_4.mp4')

    # Der Tracker wird initialisiert
    tracker = Tracker('models/best.pt')

    # Objekttracks werden ermittelt, ggf. aus einem Stub geladen
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='stubs/track_stubs.pkl')

    # Ballpositionen werden interpoliert
    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])

    # Die Teamfarben werden den Spielern zugewiesen
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    # Für jeden Frame werden die Spieler den Teams zugeordnet
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # Leere Liste für Ballkontrolle
    team_ball_control = []

    # Der Ball wird den Spielern zugewiesen
    player_assigner = PlayerBallAssigner()
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        # Falls der Ball einem Spieler zugewiesen wird, wird dem player ['has_ball'] hinzugefügt
        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            # Das Team des Spielers wird beim Ballbesitz berücksichtigt
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            # Wenn kein Spieler zugeordnet ist, bleibt die Ballkontrolle unverändert
            team_ball_control.append(team_ball_control[-1])
    team_ball_control = np.array(team_ball_control)

    # Die Annotationen (Spieler, Schiedsrichter, Ballbesitz etc.) werden auf die Frames gezeichnet
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    # Das Video wird gespeichert
    save_video(output_video_frames, 'output_vids/output_video.avi')



if __name__ == '__main__':
    main()

