import os
import pandas as pd
from tqdm import tqdm
from GestureModel import GestureModel

import warnings
warnings.filterwarnings('ignore')
import cv2
import os
import numpy as np
import pickle as pkl
import mediapipe as mp

filenameDataset = 'gestures_dataset.pickle'


def cargar_dataset():
    videos = [
        file_name.replace(".mp4", "")
        for root, dirs, files in os.walk(os.path.join("data", "videos"))
        for file_name in files
        if file_name.endswith(".mp4")
    ]
    dataset = [
        file_name.replace(".pickle", "").replace("pose_", "")
        for root, dirs, files in os.walk(os.path.join("data", "dataset"))
        for file_name in files
        if file_name.endswith(".pickle") and file_name.startswith("pose_")
    ]

    #Crea el dataset a partir de los videos de referencia
    videos_not_in_dataset = list(set(videos).difference(set(dataset)))
    n = len(videos_not_in_dataset)
    if n > 0:
        #print(f"\nExtraer puntos de referencia de nuevos videos: {n} videos detectados\n")
        #rint(f"Videos detectados:")
        #for video_name in videos_not_in_dataset:
            #print(f"{video_name}")

        for idx in tqdm(range(n)):
            save_landmarks_from_video(videos_not_in_dataset[idx])

    return videos


def cargar_referencia_señales(videos):
    reference_signs = pd.DataFrame(columns=["name", "sign_model", "distance"])
    for video_name in videos:
        sign_name = video_name.split("-")[0]
        path = os.path.join("data", "dataset", sign_name, video_name)

        left_hand_list = load_array(os.path.join(path, f"lh_{video_name}.pickle"))
        right_hand_list = load_array(os.path.join(path, f"rh_{video_name}.pickle"))


        reference_signs = reference_signs.append(
            {
                "name": sign_name,
                "sign_model": GestureModel(left_hand_list, right_hand_list),
                "distance": 0,
            },
            ignore_index=True,
        )

    # print(sign_name)
    # reference_signs = pd.concat(tmp, ignore_index=True)
    # print(reference_signs)
    # type(reference_signs)
    #return reference_signs

    reference_signs.to_pickle(filenameDataset)


def extraer_puntos_referencia(results):

    pose = landmark_to_array(results.pose_landmarks).reshape(99).tolist()

    left_hand = np.zeros(63).tolist()
    if results.left_hand_landmarks:
        left_hand = landmark_to_array(results.left_hand_landmarks).reshape(63).tolist()

    right_hand = np.zeros(63).tolist()
    if results.right_hand_landmarks:
        right_hand = (
            landmark_to_array(results.right_hand_landmarks).reshape(63).tolist()
        )
    return pose, left_hand, right_hand

def landmark_to_array(mp_landmark_list):
    keypoints = []
    for landmark in mp_landmark_list.landmark:
        keypoints.append([landmark.x, landmark.y, landmark.z])
    return np.nan_to_num(keypoints)

def save_landmarks_from_video(video_name):
    landmark_list = {"pose": [], "left_hand": [], "right_hand": []}
    sign_name = video_name.split("-")[0]

    #Establecer la transmisión de video
    cap = cv2.VideoCapture(
        os.path.join("data", "videos", sign_name, video_name + ".mp4")
    )
    with mp.solutions.holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                #Hacer detecciones
                image, results = deteccion_mediapipe(frame, holistic)

                #Almacenar resultados
                pose, left_hand, right_hand = extraer_puntos_referencia(results)
                landmark_list["pose"].append(pose)
                landmark_list["left_hand"].append(left_hand)
                landmark_list["right_hand"].append(right_hand)
            else:
                break
        cap.release()

    #Crear la carpeta de señas si no existe
    path = os.path.join("data", "dataset", sign_name)
    if not os.path.exists(path):
        os.mkdir(path)

    #Crea la carpeta de los datos de video si no existe
    data_path = os.path.join(path, video_name)
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    #Guardando la Landmark_list en la carpeta correcta
    save_array(
        landmark_list["pose"], os.path.join(data_path, f"pose_{video_name}.pickle")
    )
    save_array(
        landmark_list["left_hand"], os.path.join(data_path, f"lh_{video_name}.pickle")
    )
    save_array(
        landmark_list["right_hand"], os.path.join(data_path, f"rh_{video_name}.pickle")
    )

def load_array(path):
    file = open(path, "rb")
    arr = pkl.load(file)
    file.close()
    return np.array(arr)

def save_array(arr, path):
    file = open(path, "wb")
    pkl.dump(arr, file)
    file.close()


def deteccion_mediapipe(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_landmarks(image, results):
    mp_holistic = mp.solutions.holistic  # Holistic model
    mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

    #Dibujar conexiones de mano izquierda
    image = mp_drawing.draw_landmarks(
        image,
        landmark_list=results.left_hand_landmarks,
        connections=mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(
            color=(232, 254, 255), thickness=1, circle_radius=4
        ),
        connection_drawing_spec=mp_drawing.DrawingSpec(
            color=(255, 249, 161), thickness=2, circle_radius=2
        ),
    )

    #Dibujar conexiones de mano derecha
    image = mp_drawing.draw_landmarks(
        image,
        landmark_list=results.right_hand_landmarks,
        connections=mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(
            color=(232, 254, 255), thickness=1, circle_radius=4
        ),
        connection_drawing_spec=mp_drawing.DrawingSpec(
            color=(255, 249, 161), thickness=2, circle_radius=2
        ),
    )
    return image
