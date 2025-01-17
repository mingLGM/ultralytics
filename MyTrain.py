from ultralytics import YOLO

if __name__ == "__main__":
    # # Load a model
    # model = YOLO("yolov8l.yaml")  # build a new model from scratch
    # #model = YOLO(r'D:\lzm\work\ultralytics\zhitong_yolov8l\v2.1\train\weights\best.pt', task="detect")  # load a pretrained model (recommended for training)
    # model = YOLO(r'E:\项目\纸筒项目\models\best1107.pt', task="detect")
    # # model.train(data="coco128.yaml", epochs=3, batch=8, imgsz=640, workers=0, device="1")  # 训练模型
    # model.train(data='zhitong.yaml', lr0=0.001,  epochs=200, patience=500, batch=2, rgb2gray=True, imgsz=1280, save=True, save_period=40, device="0", workers=0,
    #             project='zhitong_yolov8l', name='v3.0', optimizer='SGD', cos_lr=True, amp=True)  # 训练模型
    # model.val() # 在验证集模型上评估模型性能
    
    
    # # 纸筒分类
    # model = YOLO("yolov8-cls.yaml")  # build a new model from scratch
    # model = YOLO(r'D:\lzm\work\ultralytics\zhitong-cls_yolov8s\v1.0_中断3\weights\best.pt', task="classify")  # load a pretrained model (recommended for training)
    # # model.train(data='zhitong_cls', epochs=3, batch=8, imgsz=320, workers=0, device="1")  # 训练模型
    # model.train(data='zhitong_cls', lr0=0.01,  epochs=100, patience=50, batch=64, imgsz=640, save=True, save_period=10, device="0", workers=0,
    #             project='zhitong-cls_yolov8s', name='v1.1', optimizer='SGD', cos_lr=True, amp=True)  # 训练模型
    # # model.val() # 在验证集模型上评估模型性能
    
    
    # bufeng
    model_yaml = r"yolo11s.yaml"
    data_yaml  = r"bufeng.yaml"
    pre_model  = r"E:\work\code\ultralytics\project\bufeng_yolov11s640\v1.4\weights\best.pt"  #r"yolo11s.pt"
    model = YOLO(model_yaml)
    model = YOLO(pre_model, task="detect")
    model.train(data=data_yaml, lr0=0.001,  epochs=1000, patience=0, batch=16, imgsz=640, save=True, save_period=50, device="0", workers=4,
                project='./project/bufeng_yolov11s640', name='v1.5', optimizer='SGD', cos_lr=True, amp=True) 
    model.val(data=data_yaml, imgsz=640) 
    
    
    
    # model = YOLO("yolo11m.pt")
    # # model = YOLO("yolov8m.yaml")  # build a new model from scratch
    # #model = YOLO(r'D:\lzm\work\ultralytics\zhitong_yolov8l\v2.1\train\weights\best.pt', task="detect")  # load a pretrained model (recommended for training)
    # model.train(data="coco128.yaml", epochs=500, batch=4, imgsz=640, workers=0, device="0")  # 训练模型
    # # model.train(data='zhitong.yaml', lr0=0.001,  epochs=200, patience=500, batch=2, rgb2gray=True, imgsz=1280, save=True, save_period=40, device="0", workers=0,
    # #             project='zhitong_yolov8l', name='v3.0', optimizer='SGD', cos_lr=True, amp=True)  # 训练模型
    # model.val() # 在验证集模型上评估模型性能
