

```bash
pip install -r requirements.txt  # install
```
# Export torch to onnx and openvino
```bash
python3 convert.py --weights test_model/activity_small.pt --imgsz 416 --batch-size 1 --include openvino
```

# inference
```bash
python detect.py --weights test_model/activity_small.pt --img 416 --conf 0.25 --source data/videos/single_person.mp4 --view-img
#torch

```
```bash
python detect.py --weights test_model/activity_small --img 416 --conf 0.25 --source data/videos/single_person.mp4 --view-img
#openvino

```
```bash
python detect.py --weights test_model/activity_small.onnx --img 416 --conf 0.25 --source data/videos/single_person.mp4 --view-img
#onnx

```