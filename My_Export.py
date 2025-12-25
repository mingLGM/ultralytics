from ultralytics import YOLO
import torch
import onnx
from onnxsim import simplify
import openvino as ov
import os
from pathlib import Path
from comutils.simplecrypto import MultiFileEncryption, FileEncryption
from comutils.My_simplecrypto import encrypt_files, decrypt_files
# from openvino.tools.mo import convert_model
from ultralytics import RTDETR


def export_hc_pt(yolo_pt:str):
    root = Path(yolo_pt).parent
    name = Path(yolo_pt).name
    # Load a model
    model = YOLO(yolo_pt)  # load an official model
    md = model.model.state_dict()
    yolo_pt = root.joinpath('_' + name)
    torch.save({'model':md}, yolo_pt)



if __name__ == '__main__':
    # export_hc_pt(r'E:\work\code\ultralytics\zhitong_yolov8l\v2.1\train\weights\best.pt')




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
    
    
    temp_pt_path  = r'rtdetr-x.pt'
    # temp_dir = Path(r'D:\HCAI\Result\Project\Prj000_检测形状\models\exp4')
    # temp_pt_path = temp_dir.joinpath(os.path.basename("best.pt"))
    model = RTDETR(temp_pt_path)
    # model.export(format='onnx', opset=16, simplify=True, dynamic=False, half=False)
    
    # state_dict = model.state_dict()
    # model_dict = {'model': state_dict}
    # torch.save(model_dict, 'our.pt')
    
    
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = torch.load(r'E:\work\code\ultralytics\zhitong_yolov8l\v2.1\train\weights\best.pt', map_location=device)
    # model["model"].eval()
    # inputs = torch.randn(1,3,1280,1280, device=device)
    # torch.onnx.export(model["model"],inputs, 'dynamic_model.onnx', training=False, input_names=["images"], dynamic_axes={'images': {0: 'batch_size'}})
    
    
    # model.eval()
    # dummy_input = torch.randn(1, 3, 1280, 1280, requires_grad=True)  # 这里使用了静态的batch_size
    # torch.onnx.export(model, dummy_input, "dynamic_model.onnx", opset_version=10, input_names=["images"], dynamic_axes={'images': {0: 'batch_size'}})
    
    
    
    # # #任务: QR码
    # # #pt转onnx
    # temp_dir = Path(r'E:\work\code\ultralytics\project\QRDM_yolov8Seg_M1_n640_cls3\Ftrain_1.0\weights')
    # temp_pt_path = temp_dir.joinpath(os.path.basename("best.pt"))
    # model = YOLO(temp_pt_path, task='segment')  #'segment'  'pose'
    # model.export(format='onnx', opset=10, simplify=True)  # , simplify=True
    
    # # ##onnx转openvino
    # temp_onnx_path = temp_dir.joinpath(os.path.basename("best.onnx"))
    # xml_path_file = temp_onnx_path.with_suffix('.xml')
    # bin_path_file = temp_onnx_path.with_suffix('.bin')
    # temp_ov_model = ov.convert_model(temp_onnx_path)
    # ov.save_model(temp_ov_model, output_model=xml_path_file, compress_to_fp16=True)
    
    # ## openvino加密 
    # hcov_path_file = xml_path_file.with_suffix('.hcir')
    # encrypt_files(xml_path_file, bin_path_file, hcov_path_file)
    # # decrypt_files(hcov_path_file, xml_path_file, bin_path_file)
    
    
    
    
    
    # # 铭的任务: 布缝、线束等
    # ##pt转onnx
    # temp_dir = Path(r'E:\work\code\ultralytics\project\xianshu_jiaoke_yolov11s320_cls-1\train_1.0\weights')
    # temp_pt_path = temp_dir.joinpath(os.path.basename("best.pt"))
    # model = YOLO(temp_pt_path, task='segment')  ##'detect' 'segment' 
    # model.export(format='onnx', opset=10)
    
    
    # # #onnx转OV
    # temp_onnx_path = temp_dir.joinpath(os.path.basename("best.onnx"))
    # xml_path_file = temp_onnx_path.with_suffix('.xml')
    # bin_path_file = str(xml_path_file).replace('.xml', '.bin')
    # temp_ov_model = ov.convert_model(temp_onnx_path)
    # ov.save_model(temp_ov_model, output_model=xml_path_file, compress_to_fp16=True)
    
    
    # ##OV加密
    # # #### HC加密
    # # # from comutils.simplecrypto import MultiFileEncryption, FileEncryption
    # # # dataset_dir = r"E:\work\Data\bufeng_2"
    # # # out_path_file = str(xml_path_file).replace('.xml', '.engineOV')
    # # # encry = None
    # # # if Path(dataset_dir + '/labelinfo/Categories.json').exists():
    # # #     encry = MultiFileEncryption({'model_xml':str(xml_path_file), 'model_bin':str(bin_path_file), 'categories':dataset_dir + '/labelinfo/Categories.json'})
    # # #     encry.add_buffer({"UseModelName": "detection_yolo11"})  #"detection_yolov5", "detection_yolov8", "detection_yolo11"
    # # # else:
    # # #     encry = MultiFileEncryption({'model_xml':str(xml_path_file), 'model_bin':str(bin_path_file)})
    # # # chiper_buffer = encry()
    # # # if chiper_buffer is not None:
    # # #     with open(out_path_file, 'wb') as f:
    # # #         f.write(chiper_buffer)
    # #### ming加密
    # hcov_path_file = xml_path_file.with_suffix('.EngineOV')
    # encrypt_files(xml_path_file, bin_path_file, hcov_path_file)
    # # decrypt_files(hcov_path_file, xml_path_file, bin_path_file)
    
    
    
    
    # temp_dir = Path(r'E:\work\code\ultralytics\models')
    # temp_pt_path = temp_dir.joinpath(os.path.basename("models/yolov8s640-seg_official.pt"))
    # model = YOLO(temp_pt_path, task = 'segment')
    # model.export(format='onnx', simplify=True)