import cv2
import mediapipe

import Operations
from Recorder import Recorder
from Camera import Camera


if __name__ == "__main__":
    #Cree un dataset de los videos donde aún no se han extraído puntos de referencia
    videos = Operations.cargar_dataset()

    #Crea un marco de datos de signos de referencia (nombre, modelo, distancia)
    reference_signs = Operations.cargar_referencia_señales(videos)

    #Objeto que almacena resultados de mediapipe y calcula similitudes de signos
    sign_recorder = Recorder(reference_signs)

    #Objeto que dibuja puntos clave y muestra resultados
    webcam_manager = Camera()

    #Enciende la camara
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    #Configurar el entorno de Mediapipe
    with mediapipe.solutions.holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as holistic:
        while cap.isOpened():

            #Leer fuente
            ret, frame = cap.read()

            #Hacer detecciones
            image, results = Operations.deteccion_mediapipe(frame, holistic)

            #Procesar resultados
            sign_detected, is_recording = sign_recorder.procesar_resultado(results)

            #Actualice el frame
            webcam_manager.actualizar(frame, results, sign_detected, is_recording)

            pressedKey = cv2.waitKey(1) & 0xFF
            if pressedKey == ord("r"):  #Grabacion presionando r
                sign_recorder.registro()
            elif pressedKey == ord("q"):  #Detener presionando q
                break

        cap.release()
        cv2.destroyAllWindows()
