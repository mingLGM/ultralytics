import json
import cv2
import os
import numpy as np
import random
import copy 
import math
from shapely.geometry import Polygon
from pathlib import Path



def crop_and_adjust_jsons(original_images_dir,original_jsons_dir,cropped_img_dir,cropped_jsons_dir, outward, iou_threshold=0.5):
    os.makedirs(cropped_img_dir,exist_ok=True)

    os.makedirs(cropped_jsons_dir,exist_ok=True)
    
    #遍历原始json标注文件，读取对应图像，筛选出矩形标注，逐个调用process_single_crop完成裁剪与标注调整，输出新图像和新json
    for json_file in os.listdir(original_jsons_dir):
        if not json_file.endswith('.json'):
            continue
        json_path = os.path.join(original_jsons_dir,json_file)
        with open(json_path,'r',encoding='utf-8')as f:
            data =json.load(f)#data保留json的信息

        img_name = data['imagePath']#图片名字
        img_path = os.path.join(original_images_dir,img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图片:{img_path}")
            continue
    #记录当前图片关联的标注信息列表
        id = 0
    #遍历标注
        for idx,shape in enumerate(data['shapes']):
            if shape['shape_type'] not in ['rectangle','polygon']:
                continue        
            process_single_crop(img,data,shape,idx,original_images_dir,cropped_img_dir,cropped_jsons_dir,iou_threshold, img_path, id, outward)
            id = id +1        

#单矩形处理函数（针对单个矩形标注，完成图像裁剪，矩形标注坐标转换（矩形+关键点），输出裁剪后图像与新json标注）
def process_single_crop(img,original_data,box_shape,box_idx,original_images_dir,cropped_img_dir,cropped_jsons_dir,iou_threshold, img_path, id, outward):
    #适配polygon：取所有顶点计算包围盒
    points = np.array(box_shape['points'],dtype=np.float32)#rectangle标注的两个对角点
    x1,y1 = np.min(points,axis=0)
    x2,y2 = np.max(points,axis=0)
    
    box_width=x2-x1
    box_height=y2-y1
    crop_size=max(box_width,box_height)*(1 + 1/outward)
    #裁剪区域中心点
    center_x=(x1+x2)/2
    center_y=(y1+y2)/2
    #计算裁剪区域
    crop_x1=max(0,int(center_x-crop_size/2))
    crop_y1=max(0,int(center_y-crop_size/2))
    crop_x2=min(img.shape[1],int(center_x+crop_size/2))
    crop_y2=min(img.shape[0],int(center_y+crop_size/2))
    
    cropped_img=img[crop_y1:crop_y2,crop_x1:crop_x2]
    transform_matrix = np.array([[1,0,-crop_x1],[0,1,-crop_y1],[0,0,1]])#转换矩阵
    
    new_shapes =[]
    ##存储有效标注框信息
    valid_boxes=[]
    #
    #第一步：处理所有矩形和多边形标注
    for shape in original_data['shapes']:
        #按shape_type分发处理（rectangle/polygon/point...）
        if shape['shape_type'] in ['rectangle','polygon']:
            new_shape=process_polygon_or_rectangle(shape,transform_matrix,(crop_x1,crop_y1,crop_x2,crop_y2),iou_threshold)
            if new_shape:
                new_shapes.append(new_shape)       
                ##保存有效标注框信息用于点标注处理
                if new_shape['shape_type'] == 'rectangle':
                    box_points=new_shape['points']
                    valid_boxes.append({
                        'type':'rectangle',
                        'points':box_points,
                        'min_x':min(box_points[0][0],box_points[1][0]),
                        'min_y':min(box_points[0][1],box_points[1][1]),
                        'max_x':max(box_points[0][0],box_points[1][0]),
                        'max_y':max(box_points[0][1],box_points[1][1]),
                    })
                elif new_shape['shape_type'] == 'polygon' and len(new_shape['points']) >= 4:
                    box_points = new_shape['points']
                    xs = [p[0] for p in box_points]
                    ys = [p[1] for p in box_points]
                    valid_boxes.append({
                        'type':'polygon',
                        'points':box_points,
                        'min_x':min(xs),
                        'min_y':min(ys),
                        'max_x':max(xs),
                        'max_y':max(ys),
                    })
    ##第二步：处理所有点标注，只保留有效标注框内的点
    for shape in original_data['shapes']:
        if shape['shape_type'] == 'point':
            new_point = process_point(shape,transform_matrix,cropped_img.shape[:2],valid_boxes)
            if new_point:
                new_shapes.append(new_point)
    ##
    #保存裁剪后的图像和json
    base_name = os.path.splitext(original_data['imagePath'])[0]
    suffix = f'_crop{id}_outward{outward}'
    cropped_img_name = f"{base_name}{suffix}.jpg"
    cropped_json_name = f"{base_name}{suffix}.json"
    
    cv2.imwrite(os.path.join(cropped_img_dir,cropped_img_name),cropped_img)
    
    cropped_data = {
                "version":original_data["version"],
                "flags":original_data["flags"],
                "shapes": new_shapes,
                "imageData":None,
                "imagePath" :cropped_img_name,
                "imageHeight" :cropped_img.shape[0],
                "imageWidth" :cropped_img.shape[1],
                }
    with open(os.path.join(cropped_jsons_dir,cropped_json_name),'w',encoding='utf-8')as f:
        json.dump(cropped_data,f,indent=2)
    
#矩形标注转换（判断原始矩形标注与裁剪区域的交集，转换矩阵坐标到裁剪后图像，输出新矩阵标注）
def process_polygon_or_rectangle(shape,matrix,crop_region,iou_threshold):
    #统一用多边形处理（rectangle可视为特殊的polygon）
    points = np.array(shape['points'])
    x1,y1 =np.min(points,axis=0)
    x2,y2 =np.max(points,axis=0)
    #构造原始多边形
    original_poly = Polygon([(x1,y1),(x2,y1),(x2,y2),(x1,y2)])
    crop_x1,crop_y1,crop_x2,crop_y2 = crop_region
    #构造裁剪区域多边形（矩形）
    crop_poly=Polygon([(crop_x1,crop_y1),(crop_x2,crop_y1),(crop_x2,crop_y2),(crop_x1,crop_y2)])
    
    #检查原始标注框是否完全在裁剪区域内
    if not crop_poly.contains(original_poly):
        return None

    #转换所有顶点坐标
    new_points=[]
    for x,y in points:
        pt = np.dot(matrix,[x,y,1])[:2]
        new_x =np.clip(pt[0],0,crop_x2-crop_x1)
        new_y =np.clip(pt[1],0,crop_y2-crop_y1)
        new_points.append([float(new_x),float(new_y)])
    #强制转换:rectangle输出2个点,polygon输出4个点
    if shape['shape_type'] == 'rectangle':
        if len(new_points)<2:
            return None
        min_pt=np.min(new_points,axis=0)
        max_pt=np.max(new_points,axis=0)
        new_points=[[min_pt[0],min_pt[1]],[max_pt[0],max_pt[1]]]
    return{
        "label":shape["label"],
        "points":new_points,
        "group_id":None,
        "description": " ",
        "shape_type":shape["shape_type"],
        "flags":shape["flags"],
        "mask":None,
    }

#点标注转换（转换带你标注坐标到裁剪后图像，过滤裁剪区域外的点）
def process_point(shape,matrix,img_size,valid_boxes):
    x,y=shape['points'][0]
    pt=np.dot(matrix,[x,y,1])[:2]
    new_x=float(pt[0])
    new_y=float(pt[1])
    
    ##检查点是否在图像范围内
    if not (0 <= new_x < img_size[1] and 0 <= new_y < img_size[0]):
        return None
    #检查点是否在任意一个有效标注框内
    for box in valid_boxes:
        #首先检查是否在边界框内（快速检查）
        if not (box['min_x'] <= new_x <= box['max_x'] and box['min_y'] <= new_y <= box['max_y']):
            continue
        #对于矩形框：直接检查坐标范围
        if box['type'] == 'rectangle':
            if box['min_x'] <= new_x <= box['max_x'] and box['min_y'] <= new_y <= box['max_y']:
                return {
                    "label":shape["label"],
                    "points":[[new_x,new_y]],
                    "group_id":None,
                    "description": " ",
                    "shape_type":"point",
                    "flags":shape["flags"],
                    "mask":None,
                }
        #对于多边框：使用射线法判断点是否在多边形内
        elif box['type'] == 'polygon':
            if is_point_in_polygon((new_x,new_y),box['points']):
                return {
                    "label":shape["label"],
                    "points":[[new_x,new_y]],
                    "group_id":None,
                    "description": " ",
                    "shape_type":"point",
                    "flags":shape["flags"],
                    "mask":None,
                }
                
    return None
    ##        
    
##判断点是否在多边形内（射线法）
def is_point_in_polygon(point,polygon):
    x,y = point
    n = len(polygon)
    inside = False
    
    p1x,p1y = polygon[0]
    for i in range(n+1):
        p2x,p2y = polygon[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x,p1y = p2x,p2y
        
    return inside
##

def rotate_and_adjust_jsons(cropped_img_dir,cropped_jsons_dir,output_img_dir,output_json_dir):
    os.makedirs(output_img_dir,exist_ok=True)

    os.makedirs(output_json_dir,exist_ok=True)

    for json_file in os.listdir(cropped_jsons_dir) :
        if not json_file.endswith('.json'):
            continue
        #拼接json文件完整路径
        cropped_path = os.path.join(cropped_jsons_dir,json_file,)
        #拼接当前json对应的图像路径，准备读取图像
        with open(cropped_path,'r',encoding='utf-8')as f:
            cropped_data =json.load(f)
        #从json中取图像文件名
        cropped_img_name = cropped_data['imagePath']
        cropped_img_path = os.path.join(cropped_img_dir,cropped_img_name)
        #用OpenCV读取图像
        cropped_img = cv2.imread(cropped_img_path)
        if cropped_img is None:
            print(f"警告，无法读取图像{cropped_img_path}")
            continue
        #对每个角度执行旋转
        for i in x:
            angle =random.randint(-i, i)
            #调用旋转+保存函数，处理单张图+标注
            rotate_and_save(
                img=cropped_img,
                data=cropped_data,
                output_img_dir=output_img_dir,
                output_json_dir=output_json_dir,
                angle=angle,
                rotation_idx = i
            )

def rotate_img (img,angle):
    #图像旋转
    h,w = img.shape[:2]
    center =(w / 2,h / 2)
    M=cv2.getRotationMatrix2D(center,angle,1.0)
    cos = np.abs(M[0,0])
    sin = np.abs(M[0,1])
    new_w =int((h*sin)+(w*cos))
    new_h =int((h*cos)+(w*sin))
    #调整旋转矩阵，让旋转后的图像居中显示
    M[0,2] += (new_w - w) / 2
    M[1,2] += (new_h - h) / 2
    rotated_img= cv2.warpAffine(img,M,(new_w,new_h),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(127,127,127)
    )

    return rotated_img,M,new_w,new_h

def transform_polygon(points,M,new_w,new_h):
    #复用矩阵变换多边形，矩阵坐标变换
    points_homogeneous = np.hstack([np.array(points,dtype=np.float32),np.ones((len(points),1),dtype=np.float32)])
    transformed = np.dot(M,points_homogeneous.T).T
    x_coords = transformed[:,0]
    y_coords = transformed[:,1]
    #裁剪到图像尺寸范围内
    x_coords=np.clip(x_coords,0,new_w - 1)
    y_coords=np.clip(y_coords,0,new_h - 1)
    return [[float(x),float(y)] for x,y in zip(x_coords,y_coords)]

def transform_point(point,M,new_w,new_h):
    #单个点坐标变换
    homogeneous=np.array([point[0],point[1],1])
    transformed =np.dot(M,homogeneous)
    x = np.clip(transformed[0],0,new_w - 1)
    y = np.clip(transformed[1],0,new_h - 1)
    return [float(x),float(y)]

def process_annotations(data,M,new_w,new_h):
    #标注坐标转换
    #存储变换后的标注数据
    transformed_data =[]
    for shape in data['shapes']:
        cropped_type = shape['shape_type']
        #获取原始标注坐标
        cropped_points = shape['points']

        #处理rectangle类型，转换为polygon
        if cropped_type == "rectangle":
            x1,y1=cropped_points[0]
            x2,y2=cropped_points[1]
            rect_points=[[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
            new_points=transform_polygon(rect_points,M,new_w,new_h)
            new_type="polygon"
        else:
            #非矩形（如多边形）：逐个点变换
            #对多边形的每个店调用transfoem_point做坐标变换
            new_points =transform_polygon(cropped_points,M,new_w,new_h)
            new_type=cropped_type
        #过滤超出旋转后图像范围的点
        valid_points =[]
        for point in new_points:
            x,y=point
            #检查坐标是否在旋转后图像尺寸内
            if 0<= x <= new_w and 0 <= y <= new_h:
                #保留有效点
                valid_points.append([float(x),float(y)])
        #构造变换后的shape数据，存入trandformed_data
        transformed_data.append({
            'label':shape["label"],
            'points':valid_points,
            'shape_type':new_type,
            'flags':shape.get("flags",{}),
            #冗余字段（或有特殊用途）
            **({'group_id':shape["group_id"]} if "group_id" in shape else {})
        })
    return transformed_data

def rotate_and_save(img,data,output_img_dir,output_json_dir,angle,rotation_idx):
    #图像旋转与标注保存
    #执行图像旋转（调用rotate_img,返回旋转后图像，变换矩阵，新宽高）
    rotated_img,M,new_w,new_h = rotate_img(img,angle)
    #处理标注（调用process_annotations,根据旋转矩阵变换标注坐标）
    transformed_data =process_annotations(data,M,new_w,new_h)
    #构造新文件名（给旋转后的图像/标注加后缀，区分原始文件）
    #取图像文件名（不含后缀）
    base_name = os.path.splitext(data['imagePath'])[0]
    #构造旋转标记
    suffix = f'_rotate{rotation_idx}_{int(angle)}'
    #新图像名
    new_img_name = f"{base_name}{suffix}.jpg"
    #新标注文件名
    new_json_name = f"{base_name}{suffix}.json"
    #保存旋转后的图像
    cv2.imwrite(os.path.join(output_img_dir,new_img_name),rotated_img)
    #构造新json数据（包含旋转后的图像信息，更新后的标注）
    new_data ={
        #保留原始标注版本
        "version":data.get("version",""),
        #保留原始标注标记
        "flags":data.get("flags",{}),
        #更新为旋转后的图像文件名
        "imagePath":new_img_name,
        "imageData":data.get("imageData",None),
        #更新为旋转后图像的高度
        "imageHeight":new_h,
        #更新为旋转后图像的宽度
        "imageWidth":new_w,
        #放入变换后的标注（坐标已更新）
        "shapes":transformed_data
    }
    with open(os.path.join(output_json_dir,new_json_name),'w',encoding='utf-8') as f:
        json.dump(new_data,f,ensure_ascii=False,indent=2)




if __name__ == '__main__':
    
    # ## 裁剪
    # original_dir = Path(r"E:\work\Data\QRDM\datasets_bac\dataset_cls3_20250818\raw")
    # original_images_dir = original_dir.joinpath("images")
    # original_jsons_dir = original_dir.joinpath("jsons")
    # cropped_img_dir = original_dir.joinpath("crop_images")#裁剪
    # cropped_jsons_dir = original_dir.joinpath("crop_jsons")
    # outward = 0.25  # 向外扩展的比例为(1+1/outward)    #2, 0.25
    # crop_and_adjust_jsons(original_images_dir,original_jsons_dir,cropped_img_dir,cropped_jsons_dir, outward, iou_threshold=0.5)
    
    
    #旋转
    raw_dir = Path(r"E:\work\Data\QRDM\datasets_bac\dataset_cls3_20250818\raw")
    raw_images_dir = raw_dir.joinpath("images")
    raw_jsons_dir = raw_dir.joinpath("jsons")
    output_img_dir = raw_dir.joinpath("rotate_images")
    output_json_dir = raw_dir.joinpath("rotate_jsons")
    #旋转角度列表
    x=[30]  # [30,60,-30,-60]
    rotate_and_adjust_jsons(raw_images_dir,raw_jsons_dir,output_img_dir,output_json_dir)
    