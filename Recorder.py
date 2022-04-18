import pandas as pd
import numpy as np
from collections import Counter
from GestureModel import GestureModel
import Operations
from fastdtw import fastdtw


class Recorder(object):
    def __init__(self, reference_signs, seq_len=50):
        #Variables para grabar
        self.is_recording = False
        self.seq_len = seq_len

        #Lista de resultados almacenados en cada frame
        self.recorded_results = []

        #DataFrame que almacena las distancias entre el signo grabado
        #todos los signos de referencia del conjunto de datos
        self.reference_signs = reference_signs

    def registro(self):

        self.reference_signs["distance"].values[:] = 0
        self.is_recording = True

    def procesar_resultado(self, results):

        if self.is_recording:
            if len(self.recorded_results) < self.seq_len:
                self.recorded_results.append(results)
            else:
                self.calcular_distancias()
                print(self.reference_signs)

        if np.sum(self.reference_signs["distance"].values) == 0:
            return "", self.is_recording
        return self._obtener_signo_predicho(), self.is_recording

    def calcular_distancias(self):
 
        left_hand_list, right_hand_list = [], []
        for results in self.recorded_results:
            _, left_hand, right_hand = Operations.extraer_puntos_referencia(results)
            left_hand_list.append(left_hand)
            right_hand_list.append(right_hand)

        #Crea un objeto GestureModel con los puntos de referencia recopilados
        recorded_sign = GestureModel(left_hand_list, right_hand_list)

        #Calcular la similitud de signos con DTW
        self.reference_signs = self.dtw_distancias(recorded_sign, self.reference_signs)

        self.recorded_results = []
        self.is_recording = False

    def _obtener_signo_predicho(self, batch_size=5, threshold=0.4):

        #Obtener la lista de los signos de referencia más similares
        sign_names = self.reference_signs.iloc[:batch_size]["name"].values

        #Cuenta las ocurrencias de cada signo y clasifíquelas en orden descendente
        sign_counter = Counter(sign_names).most_common() 

        predicted_sign, count = sign_counter[0]
        if count / batch_size < threshold:
            return "Señal desconocida: {} {} {}".format(count, batch_size, count/batch_size)
        return predicted_sign


    def dtw_distancias(self, recorded_sign, reference_signs):

        #Incrustaciones del signo grabado
        rec_left_hand = recorded_sign.lh_embedding
        rec_right_hand = recorded_sign.rh_embedding

        for idx, row in reference_signs.iterrows():
            #Inicializar las variables de fila
            ref_sign_name, ref_sign_model, _ = row

            #Si el signo de referencia tiene el mismo número de manos, calcule fastdtw
            if (recorded_sign.has_left_hand == ref_sign_model.has_left_hand) and (
                recorded_sign.has_right_hand == ref_sign_model.has_right_hand
            ):
                ref_left_hand = ref_sign_model.lh_embedding
                ref_right_hand = ref_sign_model.rh_embedding

                if recorded_sign.has_left_hand:
                    row["distance"] += list(fastdtw(rec_left_hand, ref_left_hand))[0]
                if recorded_sign.has_right_hand:
                    row["distance"] += list(fastdtw(rec_right_hand, ref_right_hand))[0]

            #Si no, la distancia es igual a infinito
            else:
                row["distance"] = np.inf
        return reference_signs.sort_values(by=["distance"])