# Object-Detection_Football-Analysis

Das Ziel der Anwendung ist es, Fussballszenen zu analysieren und sowohl die Spieler, als auch den Ball zu tracken. 
Auch soll der Ballbesitz errechnet und den Teams zugeordnet werden. Das ganze wird realisiert mithilfe von YOLO.
YOLO (You Only Look Once) ist ein Echtzeit-Objekterkennungsmodell, das Bilder in einem einzigen Durchlauf durch 
das neuronale Netzwerk analysiert, um Objekte zu erkennen und ihre Positionen (Bounding Boxes) in einem Bild zu 
bestimmen. Es gehört zur Familie der CNN-basierten (Convolutional Neural Networks) Deep-Learning-Modelle und wird 
hauptsächlich für die Aufgaben der Objektlokalisierung und -klassifizierung eingesetzt.

Step 1:
Unter input_vids sollte eine Szene als Video-File importiert werden. Im Anwendungsbeispiel sieht das so aus:
![Rohes Videomaterial](https://i.giphy.com/media/v1.Y2lkPTc5MGI3NjExMHc0d2FvdTBiajBiN25ybm5yajZ1b2cwd2doNnp5dHBxdWVjMmY0MiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/8LeIydH1KVC118Wg4E/giphy.gif)


Step 2:
Jetzt kann das pretrained Yolo-Modell über yolo.py benutzt werden für ein erstes Ergebnis:
![2](https://i.giphy.com/media/v1.Y2lkPTc5MGI3NjExNW4ycWV3azMxa3lwYjRwbXdxanFsZGtsajA5eTRybml1dnExczBxNSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/elUlCvaQakx9Sdv0V2/giphy.gif)

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
![3](https://i.giphy.com/media/v1.Y2lkPTc5MGI3NjExNXNlZG1rb2NkMndpNzlvamRocXltN2R5N2t1aG9kcWl3YW0wNDRocyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/MzFozYme2qFKMqNMjX/giphy.gif)



