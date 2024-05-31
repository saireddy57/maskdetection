import torch
import numpy as np
import cv2
import io
from PIL import Image
# from PIL import Image
from io import BytesIO
import numpy as np
import copy

def prediction(file_obj,model):
    # model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_file,force_reload=True)
    image_bytes = file_obj.read()
    image_str = Image.open(BytesIO(image_bytes))
    image_array = np.array(image_str)
    model_out = model(image_array)
    df = model_out.pandas().xyxy[0]
    img = copy.deepcopy(image_array)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for index, row in df.iterrows():
        xmin = row['xmin']
        ymin = row['ymin']
        xmax = row['xmax']
        ymax = row['ymax']
        # confidence = row['confidence']
        # class_value = row['class']
        name = row['name']
        cv2.rectangle(img, (round(xmin), round(ymin)), (round(xmax), round(ymax)), (255, 0, 0), 3)
        cv2.putText(img, name, (round(xmin) + 2, round(ymin) - 2), font, 1, (255, 0, 0), 1, cv2.LINE_AA)
    # cv2.imwrite('/home/saireddy/Videos/maskdetection/static/predict_face.png', img)
    return img