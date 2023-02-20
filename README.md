# Fall_Detection
## Summary 
Implementation of "YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors"

Pose estimation implimentation is based on [YOLO-Pose](https://arxiv.org/abs/2204.06806)
## Prepare dataset
[Keypoints Labels of MS COCO 2017](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/coco2017labels-keypoints.zip)
## Training
python train.py --data data/coco_kpts.yaml --cfg cfg/yolov7-w6-pose.yaml --weights weights/yolov7-w6-person.pt --batch-size 128 --img 960 --hyp data/hyp.pose.yaml --sync-bn
## Testing
[yolov7-w6-pose.pt](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt)

python3 fall_detection.py (replace your video path)
## References
[https://github.com/WongKinYiu/yolov7/tree/pose](https://github.com/WongKinYiu/yolov7/tree/pose)
