# Object-Detection_Football-Analysis

Das Ziel der Anwendung ist es, Fussballszenen zu analysieren und sowohl die Spieler, als auch den Ball zu tracken. 
Auch soll der Ballbesitz errechnet und den Teams zugeordnet werden. Das ganze wird realisiert mithilfe von YOLO.
YOLO (You Only Look Once) ist ein Echtzeit-Objekterkennungsmodell, das Bilder in einem einzigen Durchlauf durch 
das neuronale Netzwerk analysiert, um Objekte zu erkennen und ihre Positionen (Bounding Boxes) in einem Bild zu 
bestimmen. Es gehört zur Familie der CNN-basierten (Convolutional Neural Networks) Deep-Learning-Modelle und wird 
hauptsächlich für die Aufgaben der Objektlokalisierung und -klassifizierung eingesetzt.

Step 1 (Preparing):
Unter input_vids sollte eine Szene als Video-File importiert werden. Im Anwendungsbeispiel sieht das so aus:
![Rohes Videomaterial](https://i.giphy.com/media/v1.Y2lkPTc5MGI3NjExMHc0d2FvdTBiajBiN25ybm5yajZ1b2cwd2doNnp5dHBxdWVjMmY0MiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/8LeIydH1KVC118Wg4E/giphy.gif)


Step 2(Preparing):
Jetzt kann das pretrained Yolo-Modell über yolo.py benutzt werden für ein erstes Ergebnis:

![Pretrained Modell](https://i.giphy.com/media/v1.Y2lkPTc5MGI3NjExb2x1OG0xbG9mN2w4NjNyMTYzNmtuNm80bmVpZzg1YmR4djN0cm5jaiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/elUlCvaQakx9Sdv0V2/giphy-downsized.gif)

Da in einem Fussballspiel viel los ist (viele bewegende Spieler, Ball, Schiedsrichter, etc.) sieht das ganze 
natürlich etwas überladen aus, dennoch funktioniert das Tracking schon ganz ordentlich.

Um das Tracking spezifisch anzupassen, sodass nur das Fussballfeld (und nicht das ganze Bild) getrackt wird, 
muss das Yolo-Modell angepasst/trainiert werden. Auf Roboflow gibt es bereits Trainingssets, die man dafür
verwenden kann. Im folgenden wird dieses Set angewandt:
https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc/dataset/1
Im Ordner /training wird mit der preparing.ipynb das Set heruntergeladen und sämtliche Abhängigkeiten installiert.

Im mitgelieferten Dataset ist eine YAML Datei, die in der train_yolo.py angegeben werden muss. Danach wird das Modell
trainiert und das aktualisierte Modell wird in /training/runs/detect/weights gespeichert. Die Modelle werden in den 
/models Ordner verschoben, danach kann die yolo_trained.py angewendet werden, um folgendes Ergebnis zu erhalten:

![Dataset-trained Modell](https://i.giphy.com/media/v1.Y2lkPTc5MGI3NjExNXNlZG1rb2NkMndpNzlvamRocXltN2R5N2t1aG9kcWl3YW0wNDRocyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/MzFozYme2qFKMqNMjX/giphy-downsized.gif)

Step 3:
Die Bounding-Boxes inkl. der Beschriftung nehmen viel Platz weg, deshalb werden diese in den nächsten Steps angepasst
und gestyled. Im folgenden wird die main.py ausgeführt und folgende Aktionen ausgeführt:

![Ellipse_red](https://i.giphy.com/media/v1.Y2lkPTc5MGI3NjExZm5uZnFwZjZyb3A0ZzR5a21xM2Vscm42OGw4dGdzbnptZzd1cXRyZyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/tkHy1s5rVpso8fIoEc/giphy-downsized.gif)

Die Bounding-Boxen werden entfernt und stattdessen wird eine Ellipse unter den Spielern (und dem Schiedsrichter) gezeichnet.
Ebenfalls erscheint eine Track-ID des jeweiligen Objekts.
Im nächsten Step werden die Ellipsen von den Farben her angepasst, sodass die der Teamfarbe entsprechend sind:

![Ellipse_team_colors](https://i.giphy.com/media/v1.Y2lkPTc5MGI3NjExZjhiZnZueTMzczF0bGg4dHQydXpra3U1ZDI5NzJxeW9qbTFuNnBlOCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/GW0W4Z0yPs4z4yg9tY/giphy.gif)

Im Anschluss darauf soll der Ball getracked werden:

![ball_tracked_soft](https://i.giphy.com/media/v1.Y2lkPTc5MGI3NjExazhnZHdoMXdzenhuYWN6MGVjdWNqcXI4N3JjeHBtMG9xZnFxMXEzciZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/0ZpiL39jC6H6UFhJPd/giphy-downsized.gif)

Problem: Man sieht, dass der Ball nicht immer erkannt wird. Generell hat Yolo seine Schwachstelle im detecten von kleinen 
Objekten. Das grüne Dreieckt ist nicht immer auf dem Ball. Um das Problem zu umgehen, wird Interpolation genutzt, um
fehlende Werte der Bounding-Boxes aufzufüllen:

![ball_tracked_hard](https://i.giphy.com/media/v1.Y2lkPTc5MGI3NjExdzVxaG1rczR6MjlqNGk3aHVkZGhmMm5sYWQ4Y2Jid21hZ3IxY2dweiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/GW0W4Z0yPs4z4yg9tY/giphy-downsized.gif)

Auch sollen die Spieler, die am nächsten zum Ball stehen detected werden, um später den Ballbesitzwert ermitteln zu
können:

![player_detect](https://i.giphy.com/media/v1.Y2lkPTc5MGI3NjExZzJkd2Z3aXM4ZTVsaHQ4ZXFmZGR4d3E3bGdzN2o2cHlrYzhxcDZzdyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/lgfJi8gT1Ip65E7HNj/giphy-downsized.gif)

Die Werte können nun den Teams zugeordnet werden und der prozentuale Ballbesitz kann angezeigt werden:

![possesion](https://i.giphy.com/media/v1.Y2lkPTc5MGI3NjExMGpnYWFmZ3Zobmg3dHo3ZjZ0dzVyM2ZlOGJjMW5iMTl2YXlrMzJ1MiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/LXLdkjjig6n7An4XKA/giphy.gif)




