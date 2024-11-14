import torch
from pathlib import Path
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ultralytics import YOLO
    
def export_hc_pt(yolo_pt:str):
    root = Path(yolo_pt).parent
    name = Path(yolo_pt).name
    # Load a model
    model = YOLO(yolo_pt)  # load an official model
    md = model.model.state_dict()
    yolo_pt = root.joinpath('_' + name)
    torch.save({'model':md}, yolo_pt)

if __name__ == '__main__':
    export_hc_pt(r'D:\work\deepcode\pretrain-models\yolo11l.pt')