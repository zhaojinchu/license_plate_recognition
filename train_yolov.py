import os

yolo_dir = "yolov5" 
dataset_yaml = "license_plate.yaml"
weights = "outputs/train/license_detection/weights/last.pt"  # Using previously trained weights
output_dir = "outputs/train/license_detection"
os.system(f"""
    python {yolo_dir}/train.py \
        --img 416 \
        --batch 16 \
        --epochs 50 \
        --data {dataset_yaml} \
        --weights {weights} \
        --project outputs/train \
        --name license_detection
""")
