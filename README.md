# yolov5 for competition
 Consider both legibility and effectiveness. So called simple.
## Usage
```
# train
python train.py -p config/driver/driver-m.yaml --batch-size 16 --img 640 --epochs 300

# predict
CUDA_VISIBLE_DEVICES=1 python predict --input-images /home/work/xx --output-images /home/work/yy --checkpoint ./checkpoints/driver/xx.pt --conf 0.4
```

