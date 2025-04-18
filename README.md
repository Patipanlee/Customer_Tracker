# Customer_Tracker

This project aims to experiment with the use of a pretrained model for human detection and tracking, along with gender and age prediction, using YOLO with BoT-SORT and Insightface for the experiment.

## To start

1. Create Virtual Environment
```sh
python -m venv CT-env
```

2. Activate Virtual Environment
```sh
env\Scripts\activate
```

3. Install Package
```sh
pip install -U -r requirements.txt
```
## Prepare Project

1. configure a custom tracker
- use *"\CT-env\Lib\site-packages\ultralytics\cfg\trackers\botsort.yaml"*
- copy and rename
- "\CT-env\Lib\site-packages\ultralytics\cfg\trackers\custom.yaml"

2. download ReID model
- [ReID Model Zoo](https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO#same-domain-reid)
select download on **msmt17**
- Place at the same level as the file to be used.

## Build Your Model

In Tracker_Model() , the base models are yolo11 and buffalo_s .

### Method

1. **customer_tracking()**
- track to get id
2. **get_face_features()**
- predict age and gender
3. **plot_line()**
- draw a path