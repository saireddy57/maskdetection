from flask import Flask, request, render_template
from flask_cors import CORS
from predict import prediction
import torch
from PIL import Image
from io import BytesIO
import base64

app = Flask(__name__)
CORS(app)
model = torch.hub.load('ultralytics/yolov5', 'custom', path="Quant_MASKDET.onnx",force_reload=True)

# class MaskApp:
#     def __init__(self, model,file_obj):
#         self.file_obj = file_obj
#         self.model = model
#         self.classifier = prediction(self.model, self.file_obj)
#         print("SELF=========================",self.file_obj)
#         print("SELF CLASSIFIER_----------------------",self.classifier)

@app.route("/")
def main():
    return render_template("index.html")

# Prediction route
@app.route('/prediction', methods=['POST','GET'])
def predict_image_file():
    # try:
    if request.method == 'POST':
        temp_file = request.files['file']
        arr = prediction(temp_file,model)
        img = Image.fromarray(arr.astype('uint8'))
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        img_str = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
        return render_template("result.html",img_data=img_str)
    return render_template("result.html")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=False)
