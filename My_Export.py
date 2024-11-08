from ultralytics import YOLO
import torch
import onnx
from onnxsim import simplify
import openvino as ov
import os
from pathlib import Path
# from openvino.tools.mo import convert_model

def export_hc_pt(yolo_pt:str):
    root = Path(yolo_pt).parent
    name = Path(yolo_pt).name
    # Load a model
    model = YOLO(yolo_pt)  # load an official model
    md = model.model.state_dict()
    yolo_pt = root.joinpath('_' + name)
    torch.save({'model':md}, yolo_pt)

if __name__ == '__main__':
    export_hc_pt(r'E:\work\code\ultralytics\zhitong_yolov8l\v2.1\train\weights\best.pt')




    # # # Load a model
    # # # model = YOLO("yolov8-cls.yaml")  # build a new model from scratch
    # # # model = YOLO(r'D:\work\deepcode\project\zhitong-cls_resnet34\v8.0\epoch_250.pt', task="classify")  # load a custom trained
    # # # model = YOLO(r'E:\work\code\ultralytics\project\zhitong\best-1010.pt')  # load a custom trained
    # # # model = YOLO('yolov8l.yaml')
    # model = YOLO(r'E:\work\code\ultralytics\zhitong_yolov8l\v2.1\train\weights\best.pt', task='detect')  # load a custom trained
    # # #Export the model
    # # model.export(format='onnx', half=False, dynamic=True, opset=10)
    # # model.export(format='onnx', dynamic=True, opset=10)
    # # model.export(format='onnx', batch=8, dynamic=False)
    # model.export(format='onnx', opset=10)
    
    
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = torch.load(r'E:\work\code\ultralytics\zhitong_yolov8l\v2.1\train\weights\best.pt', map_location=device)
    # model["model"].eval()
    # inputs = torch.randn(1,3,1280,1280, device=device)
    # torch.onnx.export(model["model"],inputs, 'dynamic_model.onnx', training=False, input_names=["images"], dynamic_axes={'images': {0: 'batch_size'}})
    
    
    # model.eval()
    # dummy_input = torch.randn(1, 3, 1280, 1280, requires_grad=True)  # 这里使用了静态的batch_size
    # torch.onnx.export(model, dummy_input, "dynamic_model.onnx", opset_version=10, input_names=["images"], dynamic_axes={'images': {0: 'batch_size'}})
    
    
    
    # ####onnx2openvino
    # ov_model = ov.convert_model(r'E:\work\code\ultralytics\zhitong-cls_yolov8s/v1.1/weights/yolov8_cls_best.onnx')
    # # Optionally adjust model by embedding pre-post processing here...
    # ov.save_model(ov_model, output_model=r'E:\work\code\ultralytics\zhitong-cls_yolov8s/v1.1/weights/yolov8_cls_best.xml', compress_to_fp16=True)
    
    
    
    
    
    
    
    # #pt转onnx
    # model = YOLO(r'E:\work\code\ultralytics\bufeng_yolov8s640\v1.0\weights\last.pt', task='detect')
    # model.export(format='onnx', opset=10)
    
    
    #onnx转OV
    temp_dir = Path(r'E:\work\code\ultralytics\bufeng_yolov8s640\v1.0\weights\ov')
    temp_onnx_path = temp_dir.joinpath(os.path.basename("last.onnx"))
    # temp_ov_model = ov.convert_model(temp_onnx_path)
    temp_ov_path = temp_onnx_path.with_suffix('.xml')
    # ov.save_model(temp_ov_model, output_model=temp_ov_path, compress_to_fp16=False)
    
    
    
    #OV加密
    from comutils.simplecrypto import MultiFileEncryption, FileEncryption
    dataset_dir = r"E:\work\Data\bufeng"
    xml_path_file = temp_ov_path
    out_path_file = str(xml_path_file).replace('.xml', '.hcov')
    bin_path_file = str(xml_path_file).replace('.xml', '.bin')
    encry = None
    if Path(dataset_dir + '/labelinfo/Categories.json').exists():
        encry = MultiFileEncryption({'model_xml':str(xml_path_file), 'model_bin':str(bin_path_file), 'categories':dataset_dir + '/labelinfo/Categories.json'})
        encry.add_buffer({"UseModelName": "detection_yolov8"})
    else:
        encry = MultiFileEncryption({'model_xml':str(xml_path_file), 'model_bin':str(bin_path_file)})
    chiper_buffer = encry()
    if chiper_buffer is not None:
        with open(out_path_file, 'wb') as f:
            f.write(chiper_buffer)