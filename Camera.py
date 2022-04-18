import cv2
import numpy as np
import mediapipe as mp


WHITE_COLOR = (245, 242, 226)
RED_COLOR = (25, 35, 240)

HEIGHT = 600

posicion = (300, 40)
colorInput = (17,95,0)
colorOutput = (148,53, 34)

class Camera(object):

    def __init__(self):
        self.sign_detected = ""

    def actualizar(
        self, frame, results, sign_detected, is_recording
    ):
        self.sign_detected = sign_detected

        #Dibujar puntos de referencia
        self.draw_landmarks(frame, results)

        WIDTH = int(HEIGHT * len(frame[0]) / len(frame))
        #Redimensionar
        frame = cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)

        #Voltear la imagen vertical en efecto espejo
        frame = cv2.flip(frame, 1)

        #Escribe resultado
        frame = self.draw_text(frame)

        myText = ""
        #Elige color de circulo
        color = WHITE_COLOR
        if is_recording:
            color = RED_COLOR
            myText = "Grabando"

        #Actualizar el frame
        #cv2.circle(frame, (30, 30), 20, color, -1)
        cv2.putText(frame, myText, posicion, cv2.FONT_HERSHEY_DUPLEX, 1.5, colorInput, 2)

        cv2.imshow("Fuente OpenCV", frame)

    def draw_text(
        self,
        frame,
        font = cv2.FONT_HERSHEY_COMPLEX,
        font_size = 1,
        font_thickness = 2,
        offset = (HEIGHT * 0.02),
        bg_color = (245, 242, 176, 0.85),
    ):
        window_w = (HEIGHT * len(frame[0]) / len(frame))

        (text_w, text_h), _ = cv2.getTextSize(
            self.sign_detected, font, font_size, font_thickness
        )

        text_x, text_y = int((window_w - text_w) / 2), HEIGHT - text_h - offset
       
        cv2.putText(frame, self.sign_detected, posicion, cv2.FONT_HERSHEY_DUPLEX, 1.5, colorOutput, 2)

        return frame

    @staticmethod
    def draw_landmarks(image, results):
        mp_holistic = mp.solutions.holistic  #Modelo holistico
        mp_drawing = mp.solutions.drawing_utils  #Dibujar utilidades

        colorLineas = (255,99,9)
        thicknessLineas = 3
        radiusLineas = 5

        colorPuntos = (0,0,204)
        thicknessPuntos = 5
        radiusPuntos = 1

        #Dibujar las conexiones de mano izquierda
        mp_drawing.draw_landmarks(
            image,
            landmark_list=results.left_hand_landmarks,
            connections=mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(
                color=colorPuntos, thickness=thicknessPuntos, circle_radius=radiusPuntos
            ),
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=colorLineas, thickness=thicknessLineas, circle_radius=radiusLineas

            ),
        )
        #Dibujar las conexiones de mano derecha
        mp_drawing.draw_landmarks(
            image,
            landmark_list=results.right_hand_landmarks,
            connections=mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(
                color=colorPuntos, thickness=thicknessPuntos, circle_radius=radiusPuntos
            ),
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=colorLineas, thickness=thicknessLineas, circle_radius=radiusLineas
                
            ),
        )
