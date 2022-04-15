# Hand Gesture Recognition Mexico

## Dependencies
```
fastdtw
mediapipe
numpy
opencv-contrib-python
opencv-python
pandas
pytube
tdqm
```
To install dependencies:
```bash
pip3 install fastdtw mediapipe numpy opencv-contrib-python opencv-python pandas pytube tdqm
```

## Workflow
### Create dataset
The dataset is created running an script that uses the [`yt_links.csv`](yt_links.csv) file
```
python3 yt_download.py
```

### Create signals from dataset
Once the videos are in the `data/videos` folder, running the [`main.py`](main.py) will extract the features from the videos and save them into `data/dataset` using the pickle format.
```
python3 main.py
```
One more videos are added to `data/videos` the program calculates the difference between the condensed videos and the raw videos, thus condensing **only** the new videos.

### Camera loop
After the camera will start and once `r` is pressed it will start reading for a signal. When `q` is pressed it will quit.

Detailed results are shown in terminal.

## Reference
- [Mediapipe : Pose classification](https://google.github.io/mediapipe/solutions/pose_classification.html)

## To Do
* Refactor code
* Change `append` method in [dataset_utils.py](utils/dataset_utils.py) (DEPRECATED)
* Change `_get_sign_predicted` in [sign_recorder.py](sign_recorder.py) for a better matching metric.