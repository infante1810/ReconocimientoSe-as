from typing import List
import numpy as np
from HandShape import HandShape


class GestureModel(object):
    def __init__(self, left_hand_list, right_hand_list):

        self.has_left_hand = np.sum(left_hand_list) != 0
        self.has_right_hand = np.sum(right_hand_list) != 0

        self.lh_embedding = self._obtener_lista_puntos_referencia(left_hand_list)
        self.rh_embedding = self._obtener_lista_puntos_referencia(right_hand_list)

    @staticmethod
    def _obtener_lista_puntos_referencia(hand_list):

        embedding = []
        for frame_idx in range(len(hand_list)):
            if np.sum(hand_list[frame_idx]) == 0:
                continue

            hand_gesture = HandShape(hand_list[frame_idx])
            embedding.append(hand_gesture.feature_vector)
        return embedding
