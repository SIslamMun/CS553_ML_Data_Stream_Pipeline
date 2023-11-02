

```bash
pip install -r requirements.txt  # install
```
# Export torch to onnx and openvino
```bash
python3 convert.py --weights {model path} --imgsz 416 --batch-size 1 --include openvino
```

# inference
```bash
python detect.py --weights {model path} --img 416 --conf 0.25 --source data/videos/single_person.mp4 --view-img
#torch

```
```bash
python detect.py --weights {model path} --img 416 --conf 0.25 --source data/videos/single_person.mp4 --view-img
#openvino

```
```bash
python detect.py --weights {model path} --img 416 --conf 0.25 --source data/videos/single_person.mp4 --view-img
#onnx

```