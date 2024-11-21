# !/bin/bash
echo "Downloading YOLOv11 - M model"
echo "-----------------------------"
echo "Size      : 640x640"
echo "mAP       : 51.5 (COCO)"
echo "Params    : 20.1 M"
echo "FLOPs    : 68.0 B"
echo "-----------------------------"
echo "Downloading YOLOv11 - M model"
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt -O ./pretrained/yolov11m.pt