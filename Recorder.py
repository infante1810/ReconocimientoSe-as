import pandas as pd
import numpy as np
from collections import Counter

from GestureModel import GestureModel
import Operations

from fastdtw import fastdtw


class Recorder(object):
    def __init__(self, reference_signs: pd.DataFrame, seq_len=50):
        # Variables for recording
        self.is_recording = False
        self.seq_len = seq_len

        # List of results stored each frame
        self.recorded_results = []

        # DataFrame storing the distances between the recorded sign & all the reference signs from the dataset
        self.reference_signs = reference_signs

    def record(self):
        """
        Initialize sign_distances & start recording
        """
        self.reference_signs["distance"].values[:] = 0
        self.is_recording = True

    def process_results(self, results):
        """
        If the Recorder is in the recording state:
            it stores the landmarks during seq_len frames and then computes the sign distances
        :param results: mediapipe output
        :return: Return the word predicted (blank text if there is no distances)
                & the recording state
        """
        if self.is_recording:
            if len(self.recorded_results) < self.seq_len:
                self.recorded_results.append(results)
            else:
                self.compute_distances()
                print(self.reference_signs)

        if np.sum(self.reference_signs["distance"].values) == 0:
            return "", self.is_recording
        return self._get_sign_predicted(), self.is_recording

    def compute_distances(self):
        """
        Updates the distance column of the reference_signs
        and resets recording variables
        """
        left_hand_list, right_hand_list = [], []
        for results in self.recorded_results:
            _, left_hand, right_hand = Operations.extract_landmarks(results)
            left_hand_list.append(left_hand)
            right_hand_list.append(right_hand)

        # Create a GestureModel object with the landmarks gathered during recording
        recorded_sign = GestureModel(left_hand_list, right_hand_list)

        # Compute sign similarity with DTW (ascending order)
        self.reference_signs = self.dtw_distances(recorded_sign, self.reference_signs)

        # Reset variables
        self.recorded_results = []
        self.is_recording = False

    def _get_sign_predicted(self, batch_size=5, threshold=0.4):
        """
        Method that outputs the sign that appears the most in the list of closest
        reference signs, only if its proportion within the batch is greater than the threshold

        :param batch_size: Size of the batch of reference signs that will be compared to the recorded sign
        :param threshold: If the proportion of the most represented sign in the batch is greater than threshold,
                        we output the sign_name
                          If not,
                        we output "Sign not found"
        :return: The name of the predicted sign
        """
        # Get the list (of size batch_size) of the most similar reference signs
        sign_names = self.reference_signs.iloc[:batch_size]["name"].values

        # Get the list of values of the most similar signs
        # sign_values = self.reference_signs.iloc[:batch_size]["distance"].values

        # signs = self.reference_signs.iloc[:batch_size]
        # print("Sings==============")
        # print(signs)
        # sum =  np.sum(signs[:]["distance"].values)
        # for idx, row in signs.iterrows():
        #     row["distance"] = (sum - row["distance"])
        # sum =  np.sum(signs[:]["distance"].values)
        # # Weighted sum of sign_values
        # weighted_signs = {} # array of tuples
        # for idx, row in signs.iterrows():
        #     row["distance"] = row["distance"]/sum
        #     if row["name"] in weighted_signs:
        #         weighted_signs.update({row["name"] : row["distance"]+weighted_signs.get(row["name"])})
        #     else:
        #         weighted_signs.update({row["name"]: row["distance"]})
        # print(weighted_signs)
            

        # Get the most represented sign
        # weighted_signs.sort()
        # return weighted_signs[0]

        # Count the occurrences of each sign and sort them by descending order
        sign_counter = Counter(sign_names).most_common() # [(sign_name, count), ...]

        predicted_sign, count = sign_counter[0]
        if count / batch_size < threshold:
            return "Señal desconocida: {} {} {}".format(count, batch_size, count/batch_size)
        return predicted_sign





    ############# DTW.PY

    def dtw_distances(self, recorded_sign: GestureModel, reference_signs: pd.DataFrame):
        """
        Use DTW to compute similarity between the recorded sign & the reference signs

        :param recorded_sign: a GestureModel object containing the data gathered during record
        :param reference_signs: pd.DataFrame
                                columns : name, dtype: str
                                        sign_model, dtype: GestureModel
                                        distance, dtype: float64
        :return: Return a sign dictionary sorted by the distances from the recorded sign
        """
        # Embeddings of the recorded sign
        rec_left_hand = recorded_sign.lh_embedding
        rec_right_hand = recorded_sign.rh_embedding

        for idx, row in reference_signs.iterrows():
            # Initialize the row variables
            ref_sign_name, ref_sign_model, _ = row

            # If the reference sign has the same number of hands compute fastdtw
            if (recorded_sign.has_left_hand == ref_sign_model.has_left_hand) and (
                recorded_sign.has_right_hand == ref_sign_model.has_right_hand
            ):
                ref_left_hand = ref_sign_model.lh_embedding
                ref_right_hand = ref_sign_model.rh_embedding

                if recorded_sign.has_left_hand:
                    row["distance"] += list(fastdtw(rec_left_hand, ref_left_hand))[0]
                if recorded_sign.has_right_hand:
                    row["distance"] += list(fastdtw(rec_right_hand, ref_right_hand))[0]

            # If not, distance equals infinity
            else:
                row["distance"] = np.inf
        return reference_signs.sort_values(by=["distance"])