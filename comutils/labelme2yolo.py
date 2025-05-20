import os
import json
import shutil
from pathlib import Path


# 函数 处理单个labelme标注json文件
def ProcessSingleJson_Detect(labelme_path, save_folder='../../labels', Labeling_software='Labelme'):
    
    file_name, file_extension = os.path.splitext(labelme_path)
    if(file_extension != '.json'):
        print('{} 不是json文件'.format(labelme_path))
        return 0
    
    with open(labelme_path, 'r', encoding='utf-8') as f:
        labelme = json.load(f)

    img_width = labelme['imageWidth']   # 图像宽度
    img_height = labelme['imageHeight'] # 图像高度

    # 生成 YOLO 格式的 txt 文件
    # suffix = labelme_path.split('.')[-2]
    yolo_txt_path = file_name + '.txt'

    with open(yolo_txt_path, 'w', encoding='utf-8') as f:
        for each_ann in labelme['shapes']: # 遍历每个标注
            if each_ann['shape_type'] == 'rectangle': # 每个框，在 txt 里写一行
                yolo_str = ''
                ## 框的信息
                # 框的类别 ID
                bbox_class_id = bbox_class[each_ann['label']]
                yolo_str += '{} '.format(bbox_class_id)
                # 左上角和右下角的 XY 像素坐标
                if Labeling_software == 'Labelme':
                    bbox_top_left_x = int(min(each_ann['points'][0][0], each_ann['points'][1][0]))
                    bbox_bottom_right_x = int(max(each_ann['points'][0][0], each_ann['points'][1][0]))
                    bbox_top_left_y = int(min(each_ann['points'][0][1], each_ann['points'][1][1]))
                    bbox_bottom_right_y = int(max(each_ann['points'][0][1], each_ann['points'][1][1]))
                elif Labeling_software == 'XAnyLabeling':
                    bbox_top_left_x = int(min(each_ann['points'][0][0], each_ann['points'][2][0]))
                    bbox_bottom_right_x = int(max(each_ann['points'][0][0], each_ann['points'][2][0]))
                    bbox_top_left_y = int(min(each_ann['points'][0][1], each_ann['points'][2][1]))
                    bbox_bottom_right_y = int(max(each_ann['points'][0][1], each_ann['points'][2][1]))
                # 框中心点的 XY 像素坐标
                bbox_center_x = int((bbox_top_left_x + bbox_bottom_right_x) / 2)
                bbox_center_y = int((bbox_top_left_y + bbox_bottom_right_y) / 2)
                # 框宽度
                bbox_width = bbox_bottom_right_x - bbox_top_left_x
                # 框高度
                bbox_height = bbox_bottom_right_y - bbox_top_left_y
                # 框中心点归一化坐标
                bbox_center_x_norm = bbox_center_x / img_width
                bbox_center_y_norm = bbox_center_y / img_height
                # 框归一化宽度
                bbox_width_norm = bbox_width / img_width
                # 框归一化高度
                bbox_height_norm = bbox_height / img_height

                yolo_str += '{:.5f} {:.5f} {:.5f} {:.5f} '.format(bbox_center_x_norm, bbox_center_y_norm, bbox_width_norm, bbox_height_norm)

                # 写入 txt 文件中
                f.write(yolo_str + '\n')
                
    shutil.move(yolo_txt_path, save_folder)
    # print('{} --> {} 转换完成'.format(labelme_path, yolo_txt_path)) 
    return 1


# 函数 处理单个labelme标注json文件
def ProcessSingleJson_Pose(labelme_path, save_folder='../../labels'):
    
    file_name, file_extension = os.path.splitext(labelme_path)
    if(file_extension != '.json'):
        print('{} 不是json文件'.format(labelme_path))
        return 0
    
    with open(labelme_path, 'r', encoding='utf-8') as f:
        labelme = json.load(f)

    img_width = labelme['imageWidth']   # 图像宽度
    img_height = labelme['imageHeight'] # 图像高度

    # 生成 YOLO 格式的 txt 文件
    # suffix = labelme_path.split('.')[-2]
    yolo_txt_path = file_name + '.txt'

    with open(yolo_txt_path, 'w', encoding='utf-8') as f:
        for each_ann in labelme['shapes']: # 遍历每个标注
            if each_ann['shape_type'] == 'rectangle': # 每个框，在 txt 里写一行
                yolo_str = ''
                ## 框的信息
                # 框的类别 ID
                bbox_class_id = bbox_class[each_ann['label']]
                yolo_str += '{} '.format(bbox_class_id)
                # 左上角和右下角的 XY 像素坐标
                bbox_top_left_x = int(min(each_ann['points'][0][0], each_ann['points'][1][0]))
                bbox_bottom_right_x = int(max(each_ann['points'][0][0], each_ann['points'][1][0]))
                bbox_top_left_y = int(min(each_ann['points'][0][1], each_ann['points'][1][1]))
                bbox_bottom_right_y = int(max(each_ann['points'][0][1], each_ann['points'][1][1]))
                # 框中心点的 XY 像素坐标
                bbox_center_x = int((bbox_top_left_x + bbox_bottom_right_x) / 2)
                bbox_center_y = int((bbox_top_left_y + bbox_bottom_right_y) / 2)
                # 框宽度
                bbox_width = bbox_bottom_right_x - bbox_top_left_x
                # 框高度
                bbox_height = bbox_bottom_right_y - bbox_top_left_y
                # 框中心点归一化坐标
                bbox_center_x_norm = bbox_center_x / img_width
                bbox_center_y_norm = bbox_center_y / img_height
                # 框归一化宽度
                bbox_width_norm = bbox_width / img_width
                # 框归一化高度
                bbox_height_norm = bbox_height / img_height

                yolo_str += '{:.5f} {:.5f} {:.5f} {:.5f} '.format(bbox_center_x_norm, bbox_center_y_norm, bbox_width_norm, bbox_height_norm)

                ## 找到该框中所有关键点，存在字典 bbox_keypoints_dict 中
                bbox_keypoints_dict = {}
                for each_ann in labelme['shapes']: # 遍历所有标注
                    if each_ann['shape_type'] == 'point': # 筛选出关键点标注
                        # 关键点XY坐标、类别
                        x = int(each_ann['points'][0][0])
                        y = int(each_ann['points'][0][1])
                        label = each_ann['label']
                        if (x>bbox_top_left_x) & (x<bbox_bottom_right_x) & (y<bbox_bottom_right_y) & (y>bbox_top_left_y): # 筛选出在该个体框中的关键点
                            bbox_keypoints_dict[label] = [x, y]

                ## 把关键点按顺序排好
                for each_class in keypoint_class: # 遍历每一类关键点
                    if each_class in bbox_keypoints_dict:
                        keypoint_x_norm = bbox_keypoints_dict[each_class][0] / img_width
                        keypoint_y_norm = bbox_keypoints_dict[each_class][1] / img_height
                        yolo_str += '{:.5f} {:.5f} {} '.format(keypoint_x_norm, keypoint_y_norm, 2) # 2-可见不遮挡 1-遮挡 0-没有点
                    else: # 不存在的点，一律为0
                        yolo_str += '0 0 0 '
                # 写入 txt 文件中
                f.write(yolo_str + '\n')
                
    shutil.move(yolo_txt_path, save_folder)
    # print('{} --> {} 转换完成'.format(labelme_path, yolo_txt_path)) 
    return 1



def convert_YOLOPose2Seg(yolo_pose_path, save_folder):
    # 定义外接矩形类别映射
    bbox_class = {
        'QRrect': 0,
        'QRrect_DM': 0,
        'QRrect_QR': 1
    }
    
    file_name, file_extension = os.path.splitext(yolo_pose_path)
    if(file_extension != '.json'):
        # print('{} 不是json文件'.format(labelme_path))
        return 0
    
    # 读取YOLO-Pose标签文件
    with open(yolo_pose_path, 'r', encoding='utf-8') as f:
        yolo_data = json.load(f)
    
    file_name = Path(yolo_data["imagePath"])
    json_seg_path = Path(save_folder) / f"{file_name.name}.json"  #f"{image_path.stem}{image_path.suffix}.json"
    
    # 初始化自定义标签结构
    custom_data = {
        "annotations": [],
        "images": [
            {
                "ArtificialDiv": False,
                "NoTarget": False,
                "file_name": file_name.name, # 包含后缀的文件名
                "height": yolo_data["imageHeight"],
                "imagestatus": 0,
                "width": yolo_data["imageWidth"]
            }
        ]
    }

    # 第一步：收集所有二维码实例
    qr_instances = {}
    current_qr_id = 0  # 用于自动分配二维码ID
    
    for shape in yolo_data["shapes"]:
        # 处理外接矩形（用于确定类别）
        if shape["shape_type"] == "rectangle" and shape["label"] in bbox_class:
            current_qr_id += 1
            qr_instances[current_qr_id] = {
                "category": bbox_class[shape["label"]],
                "points": {}
            }
        
        # 处理关键点（格式为QRpoint_1到QRpoint_4）
        elif shape["shape_type"] == "point" and shape["label"].startswith("QRpoint"):
            point_id = int(shape["label"].split('_')[1])  # 提取数字1-4
            if current_qr_id not in qr_instances:
                qr_instances[current_qr_id] = {
                    "category": 0,  # 默认类别
                    "points": {}
                }
            qr_instances[current_qr_id]["points"][point_id] = shape["points"][0]

    # 第二步：为每个二维码生成多边形标注
    for qr_id, instance in qr_instances.items():
        if len(instance["points"]) == 4:  # 确保有4个关键点
            # 按1-2-3-4顺序排列点（顺时针或逆时针均可）
            sorted_points = [
                instance["points"][1],
                instance["points"][2],
                instance["points"][3],
                instance["points"][4]
            ]
            
            # 展平坐标并转换为整数
            flat_points = [int(round(coord)) for point in sorted_points for coord in point]
            
            # 计算多边形面积
            area = 0
            for i in range(4):
                x1, y1 = sorted_points[i]
                x2, y2 = sorted_points[(i + 1) % 4]
                area += (x1 * y2 - x2 * y1)
            area = int(abs(area) // 2)
            
            # 添加到annotations
            custom_data["annotations"].append({
                "angle": 0.0,
                "area": area,
                "category_id": instance["category"],
                "id": len(custom_data["annotations"]),
                "points": flat_points,
                "type": 2
            })
    
    # 写入输出文件
    with open(json_seg_path, 'w', encoding='utf-8') as f:
        json.dump(custom_data, f, indent=4)

    return 1




if __name__ == '__main__':
    
    
    # ## Detect: 线束端子
    # bbox_class = {
    #     'JPYJ':0 ,  #胶皮压脚有无检测
    #     'JPCD':1 ,  #胶皮出头检测
    #     'JPCT':2 ,  #胶皮长度检测
    #     'XXYJ':3 ,  #线芯压脚检测
    #     'XXCT':4 ,  #线芯出头检测
    #     'FS'  :5 ,  #飞丝检测
    # }
    
    
    ## Pose: 二维码QR、DM
    # 框的类别
    bbox_class = {
        'QRrect':0 , 
        'QRrect_DM':0 , 
        'QRrect_QR':1  
    }
    # 关键点的类别
    keypoint_class = ['QRpoint_1', 'QRpoint_2', 'QRpoint_3', 'QRpoint_4']



    # 数据集文件夹名称
    Labeling_software = 'Labelme'  ## 'Labelme'  'XAnyLabeling'
    Dataset_root = Path(r'E:\work\Data\QR\temp\raw2')  #Path(r'D:\HCAI\Result\Project\Prj007_实例分割_二维码\inputImages')  #r'E:\work\Data\QR'  
    labelme_json_path = Dataset_root.joinpath('jsons')
    # labelme_json_path = Dataset_root
    yolo_txt_path = Dataset_root.joinpath('labels')
    yolo_seg_path = Dataset_root.joinpath('jsons_seg')
    if not os.path.exists(yolo_txt_path):
        os.mkdir(yolo_txt_path)
    if not os.path.exists(yolo_seg_path):
        os.mkdir(yolo_seg_path)
    os.chdir(labelme_json_path)
    num = 0
    for labelme_path in os.listdir():
        try:
            # num += ProcessSingleJson_Detect(labelme_path, save_folder=yolo_txt_path, Labeling_software = Labeling_software)
            # num += ProcessSingleJson_Pose(labelme_path, save_folder=yolo_txt_path, Labeling_software = Labeling_software)
            num += convert_YOLOPose2Seg(labelme_path, save_folder=yolo_seg_path)
        except:
            print('******有误******', labelme_path)
    print('YOLO格式的txt标注文件已保存至: ', yolo_txt_path)
    print('转换标签文件数量: ', num)



