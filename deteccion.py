import cv2
from ultralytics import YOLO

# Inicializa el modelo YOLOv8
model = YOLO('yolov8n.pt')  # Asegúrate de que el modelo 'yolov8n.pt' esté en el directorio actual o proporciona la ruta completa.

# Inicializar la cámara
cap = cv2.VideoCapture(0)  # El argumento '0' usualmente se refiere a la cámara integrada.

while True:
    # Leer imagen del video
    ret, frame = cap.read()
    if not ret:
        break

    # Realizar detección
    results = model(frame)

    # Dibujar los cuadros delimitadores para cada detección
    annotated_frame = results[0].plot(show=False, save=False)  # El método plot devuelve una imagen anotada

    # Mostrar la imagen
    cv2.imshow('YOLOv8 Inference', annotated_frame)

    # Romper el ciclo si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()
