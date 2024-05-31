import cv2
import torch
import time

def profile_inference(model,video):
    cap = cv2.VideoCapture(video_path)
    # Get video properties
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(5))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_path = 'output_video_MASK.avi'
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        count+=1
        print(count)
        if not ret:
            break

        result = model(frame)
        res = result.pandas().xyxy[0]
        for idx,row in res.iterrows():
            x,y,w,h = round(row['xmin']),round(row['ymin']),round(row['xmax']),round(row['ymax'])
            cv2.rectangle(frame, (x, y), (int(w), int(h)), (0, 255, 0), 2)
            cv2.putText(frame, row['name'] , (x,y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 1)
        out.write(frame)


if __name__ == '__main__':
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best_MASKDET.pt',force_reload=True)
    video_path = 'LA_Mayor_Eric_Garcetti_Mandates_Face_Masks_Outside_Homes.mp4'

    start_time = time.time()
    profile_inference(model,video_path)
    end_time = time.time()
    # Calculate the execution time
    execution_time_seconds = end_time - start_time

    print("Execution time with Quantised model:", execution_time_seconds, "seconds")
    time.sleep(5)

    n_model = torch.hub.load('ultralytics/yolov5', 'custom', path='Quant_MASKDET.onnx',force_reload=True)

    start_time = time.time()
    profile_inference(n_model,video_path)
    end_time = time.time()
    # Calculate the execution time
    execution_time_seconds = end_time - start_time

    print("Execution time with Original model:", execution_time_seconds, "seconds")




