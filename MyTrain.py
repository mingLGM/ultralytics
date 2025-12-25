from ultralytics import YOLO
from ultralytics import RTDETR

if __name__ == "__main__":
    
    # 任务: 纸筒
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
    
    
    # # 任务: 布缝
    # model_yaml = r"yolo11s.yaml"
    # data_yaml  = r"bufeng.yaml"
    # pre_model  = r"E:\work\code\ultralytics\project\bufeng_yolov11s640_cls-2\v1.0\weights\best.pt"  #r"yolo11s.pt"
    # model = YOLO(model_yaml)
    # model = YOLO(pre_model, task="detect")
    # model.train(data=data_yaml, lr0=0.001,  epochs=1, patience=0, batch=16, imgsz=640, save=True, save_period=50, device="0", workers=4,
    #             project='./project/bufeng_yolov11s640_cls-2', name='v2.0', optimizer='SGD', cos_lr=True, amp=True) 
    # # model.val(data=data_yaml, imgsz=640) 
    
    
    
    # # #任务: 线束端子——“目标检测”
    # model_yaml = r"yolo11s.yaml"
    # data_yaml  = r"xianshu.yaml"
    # pre_model  = r"yolo11s.pt"  #r"yolo11s.pt"
    # model = YOLO(model_yaml)
    # model = YOLO(pre_model, task="detect")
    # model.train(data=data_yaml, lr0=0.001,  epochs=500, patience=0, batch=8, imgsz=640, save=True, save_period=50, device="0", workers=0,
    #             project='./project/xianshu_yolov11s640_cls-7', name='test', optimizer='AdamW', warmup_epochs = 5, cos_lr=True, amp=True, close_mosaic=10, multi_scale = True) 
    # # model.val(data=data_yaml, imgsz=640) 
    
    
    # #任务: 线束胶壳——“实例分割”
    # model_yaml = r"yolo11s-seg.yaml"
    # data_yaml  = r"xianshu_jiaoke_Inseg.yaml"
    # pre_model  = r'yolo11s-seg.pt' #r"yolo11m-pose.pt"  #r"yolo11s.pt"
    # model = YOLO(model_yaml)
    # model = YOLO(pre_model, task="segment")
    # model.train(data=data_yaml, lr0=0.001, lrf=0.01, cos_lr=True,  epochs=500, patience=0, batch=8, imgsz=640, save=True, save_period=50, device="0", workers=4, mask_ratio = 1,
    #             project='./project/xianshu_jiaoke_yolov11s640_cls-1', name='train_1.0', optimizer='AdamW', warmup_epochs = 5, amp=True, close_mosaic=10, multi_scale = True)  
    #     # muti_scale：图片会有多种shape传入网络，训练效果更好，但是很耗显存
    # model.val(data=data_yaml, imgsz=640) 
    
    
    
    # # 任务: QR关键点
    # ####关键点算法
    # model_yaml = r"yolo11m-pose_QR.yaml"
    # data_yaml  = r"QR-pose.yaml"
    # pre_model  = r'E:\work\code\ultralytics\project\QRDM_yolo11l320_cls2\7.0\best.pt' #r"yolo11m-pose.pt"  #r"yolo11s.pt"
    # model = YOLO(model_yaml)
    # model = YOLO(pre_model, task="pose")
    # model.train(data=data_yaml, lr0=0.0005,  epochs=1000, patience=0, batch=16, imgsz=320, save=True, save_period=50, device="0", workers=8,
    #             project='./project/QRDM_yolo11l320_cls2', name='train_8.0', optimizer='AdamW', warmup_epochs = 5, cos_lr=True, amp=True, close_mosaic=10, multi_scale = True)  
    #     # muti_scale：图片会有多种shape传入网络，训练效果更好，但是很耗显存
    # # model.val(data=data_yaml, imgsz=640) 
    
    
    
    
    # ####分割算法
    # ##大模型
    # model_yaml = r"yolov8n-seg.yaml"
    # data_yaml  = r"QRDM-seg_raw&crop.yaml"
    # pre_model  = r'yolov8n-seg.pt' #r"yolo11m-pose.pt"  #r"yolo11s.pt"  r'/home/zhangshuwen/work/code/ultralytics/project/QRDM_yolov8Seg_s640_cls2/train_1.1/weights/best.pt'
    # model = YOLO(model_yaml)
    # model = YOLO(pre_model, task="segment")
    # model.train(data=data_yaml, lr0=0.0001, lrf=0.01, cos_lr=True,  epochs=200, patience=0, batch=16, imgsz=640, save=True, save_period=50, device="0", workers=8, mask_ratio = 1,
    #             project='./project/QRDM_yolov8Seg_n640_cls2', name='train_1.0', optimizer='AdamW', warmup_epochs = 5, amp=True, close_mosaic=10, multi_scale = True)  
    #     # muti_scale：图片会有多种shape传入网络，训练效果更好，但是很耗显存
    # model.val(data=data_yaml, imgsz=640) 
    
    # ##小模型
    # model_yaml = r"yolov8n-seg.yaml"
    # data_yaml  = r"QRDM-seg_crop.yaml"
    # pre_model  = r'yolov8n-seg.pt' #r"yolo11m-pose.pt"  #r"yolo11s.pt"
    # model = YOLO(model_yaml)
    # model = YOLO(pre_model, task="segment")
    # model.train(data=data_yaml, lr0=0.001, lrf=0.01, cos_lr=True,  epochs=500, patience=0, batch=16, imgsz=320, save=True, save_period=50, device="0,1", workers=8, mask_ratio = 1,
    #             project='./project/QRDM_yolov8Seg_n320_cls2', name='train_1.0', optimizer='AdamW', warmup_epochs = 5, amp=True, close_mosaic=10, multi_scale = True)  
    #     # muti_scale：图片会有多种shape传入网络，训练效果更好，但是很耗显存
    # model.val(data=data_yaml, imgsz=320) 
    
    
    
    ## 欧普-黑点项目
    ### RT-DETR
    model_yaml = r"rtdetr-l.yaml"
    data_yaml  = r"xz_RTDETR.yaml"  
    pre_model  = r'rtdetr-l.pt'
    model = RTDETR(model_yaml)
    model = RTDETR(pre_model)
    model.info()
    results = model.train(data=data_yaml, lr0=0.0001, lrf=0.01, weight_decay=0.0005, cos_lr=True, epochs=100, patience=0, batch=4, imgsz=320, device="0", save=True, workers=0,
                        project='./project/RTDETR_xz', name='test', optimizer='AdamW', warmup_epochs = 5, amp=True, close_mosaic=10, multi_scale = True, translate= 0.1)
    # model.val(data=data_yaml, imgsz=640) 


    
    # model = YOLO("yolo11m.pt")
    # # model = YOLO("yolov8m.yaml")  # build a new model from scratch
    # #model = YOLO(r'D:\lzm\work\ultralytics\zhitong_yolov8l\v2.1\train\weights\best.pt', task="detect")  # load a pretrained model (recommended for training)
    # model.train(data="coco128.yaml", epochs=500, batch=4, imgsz=640, workers=0, device="0")  # 训练模型
    # # model.train(data='zhitong.yaml', lr0=0.001,  epochs=200, patience=500, batch=2, rgb2gray=True, imgsz=1280, save=True, save_period=40, device="0", workers=0,
    # #             project='zhitong_yolov8l', name='v3.0', optimizer='SGD', cos_lr=True, amp=True)  # 训练模型
    # model.val() # 在验证集模型上评估模型性能
