from typing import List

import numpy as np
import mediapipe as mp


class HandShape(object):

    def __init__(self, landmarks):

        #Define las conexiones
        self.connections = mp.solutions.holistic.HAND_CONNECTIONS

        #Crear vector de características (lista de los ángulos entre todas las conexiones)
        landmarks = np.array(landmarks).reshape((21, 3))
        self.feature_vector = self._obtener_caract_vector(landmarks)

    def _obtener_conexiones_puntos_referencia(self, landmarks):
    
        return list(
            map(
                lambda t: landmarks[t[1]] - landmarks[t[0]],
                self.connections,
            )
        )   
    
    def _obtener_caract_vector(self, landmarks):

        connections = self._obtener_conexiones_puntos_referencia(landmarks)

        lista_angulos = []
        for connection_from in connections:
            for connection_to in connections:
                angulo = self._obtener_angulos_entre_vectores(connection_from, connection_to)
                #Si el ángulo no es NaN la almacenamos de lo contrario almacenamos 0
                if angulo == angulo:
                    lista_angulos.append(angulo)
                else:
                    lista_angulos.append(0)
        return lista_angulos

    @staticmethod
    def _obtener_angulos_entre_vectores(u, v):

        if np.array_equal(u, v):
            return 0
        dot_product = np.dot(u, v)
        norm = np.linalg.norm(u) * np.linalg.norm(v)
        return np.arccos(dot_product / norm)
