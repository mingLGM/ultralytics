from ultralytics import YOLO
from ultralytics import RTDETR

# model = YOLO(r"C:\Users\Administrator\Desktop\czs\ultralytics-main\runs\detect\train7\weights\best.pt")
# model.predict(
#     r"D:\lzm\work\ultralytics\ultralytics\assets",
#     save=True,
#     imgsz=1280,
#     conf=0.5,
#     iou=0.2
# )

if __name__ == "__main__":

    # ####----检测----
    # # model = YOLO('yolov8l.yaml')
    # # model = YOLO(r'D:\lzm\work\ultralytics\zhitong_yolov8l\v2.1\train\weights\best.pt')  # 加载预训练的 YOLOv8n 模型
    # # model = YOLO(r'E:\work\code\ultralytics\bufeng_yolov8s640\v1.0\weights\last.pt', task='detect')  # 加载预训练的 YOLOv8n 模型
    # # model.predict(source=r'E:\铭\workspace\布缝\1031(1)\1031\无框', save=True, save_txt=True, save_conf=True, save_crop=False, conf=0.15, iou=0.3, device="0", imgsz=1280)  # 对图像进行预测
    # model = YOLO(r'E:\work\code\ultralytics\project\QRDM_yolov8Seg_s640_cls2\Ftrain_1.1\weights\best.pt', task='segment')  # 加载预训练的 YOLOv8n 模型
    # model.predict(source=r'E:\work\Data\QRDM\datasets_InSeg\test\images', save=True, save_txt=True, save_conf=True, save_crop=True, conf=0.15, iou=0.3, device="0", imgsz=640)  # 对图像进行预测
    # # model.export(format='onnx') # 将模型导出为 ONNX 格式
    
    pre_model  = r'E:\work\code\ultralytics\project\RTDETR_guangji_cls3\v1.0\weights\best.pt'
    model = RTDETR(pre_model)
    model.predict(
        source=r'E:\项目\光机\1_分割标注\crop\crop_ori\images',  # 预测的数据源，可以是图片、文件夹、视频路径等
        imgsz=640,                     # 图片的输入尺寸
        project='runs/detect/',        # 预测结果保存的项目目录
        name='RTDETR_guangji_cls3',    # 子目录名（最终保存路径为 runs/detect/exp）
        save=True,                     # 是否保存预测后的图片
        # visualize=True,              # 是否可视化中间特征图（默认注释掉，即不启用）
        # show_conf=False,             # 是否显示置信度（关闭显示）
        # show_labels=False,           # 是否显示标签（关闭显示）
    )

    # ####----分类----
    # # model = YOLO(r'E:\work\code\ultralytics\zhitong-cls_yolov8s\v1.1\weights\best.pt')  # 加载预训练的 YOLOv8n 模型
    # model = YOLO(r'D:\work\deepcode2\project\zhitong-cls_resnet34\v4\best.pt')
    # model.predict(source=r'E:\work\code\ultralytics\zhitong_yolov8l\v_xuexiao\test_0917-wujian\crops\划痕', save=True, save_txt=True,save_conf=True, conf=0.25, iou=0.2, device="0")  # 对图像进行预测



    # model = YOLO(r'E:\work\code\ultralytics\project\zhitong\best-1013.pt')
    # model.predict(source=r'E:\项目\纸筒项目\素材\Prj005_跑图\第一次\OK', save=True, conf=0.15, iou=0.3, imgsz=1280)





# import cv2
# import numpy as np

# if __name__ == "__main__":
#     img_path = "Output_onnx.jpg"
#     img = cv2.imread(img_path)
#     #获取图片的宽和高
#     width,height = img.shape[:2][::-1]
#     #将图片缩小便于显示观看
#     img_resize = cv2.resize(img,
#     (int(width*0.5),int(height*0.5)),interpolation=cv2.INTER_CUBIC)
#     cv2.imshow("img",img_resize)
#     print("img_reisze shape:{}".format(np.shape(img_resize)))

#     #将图片转为灰度图
#     img_gray = cv2.cvtColor(img_resize, cv2.COLOR_RGB2GRAY)
#     img_gray = cv2.cvtColor(img_resize, cv2.COLOR_GRAY2BGR)
#     #img_gray = np.repeat(img_gray[..., np.newaxis], 3, 2)
#     cv2.imshow("img_gray",img_gray)
#     print("img_gray shape:{}".format(np.shape(img_gray)))
#     cv2.waitKey()