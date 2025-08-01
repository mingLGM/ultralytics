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

    #保存图片->标注信息列表的映射
    image_annotation_map={}
    
    #遍历原始json标注文件，读取对应图像，筛选出矩形标注，逐个调用process_single_crop完成裁剪与标注调整，输出新图像和新json
    for json_file in os.listdir(original_jsons_dir):
      if not json_file.endswith('.json'):
        continue
      json_path = os.path.join(original_jsons_dir,json_file)
      with open(json_path,'r',encoding='utf-8')as f:
        data =json.load(f)  

      img_name = data['imagePath']
      img_path = os.path.join(original_images_dir,img_name)
      img = cv2.imread(img_path)
      if img is None:
        print(f"无法读取图片:{img_path}")
        continue
    #记录当前图片关联的标注信息列表
      id = 0
      annotations=[]
      #遍历标注
      for idx,shape in enumerate(data['shapes']):
        if shape['shape_type'] not in ['rectangle','polygon']:
            continue
        #对每个矩形标注，执行裁剪+调整json标注逻辑
        process_result=process_single_crop(img,data,shape,idx,original_images_dir,cropped_img_dir,cropped_jsons_dir,iou_threshold, img_path, id, outward)
        if process_result:
            annotations.append(process_result)
        id = id +1
        image_annotation_map[img_path]=annotations
    with open('image_annptation_mapping.json','w',encoding='utf-8') as f:
        json.dump(image_annotation_map,f,ensure_ascii=False,indent=2)

        

#单矩形处理函数（针对单个矩形标注，完成图像裁剪，矩形标注坐标转换（矩形+关键点），输出裁剪后图像与新json标注）
def process_single_crop(img,original_data,box_shape,box_idx,original_images_dir,cropped_img_dir,cropped_jsons_dir,iou_threshold, img_path, id, outward):
    #适配polygon：取所有顶点计算包围盒
            points = np.array(box_shape['points'],dtype=np.float32)
            x1,y1 = np.min(points,axis=0)
            x2,y2 = np.max(points,axis=0)
           
            box_width=x2-x1
            box_height=y2-y1
            crop_size=max(box_width,box_height)*(1 + 1/outward)
            center_x=random.randint(int(x2-(crop_size/2)),int(x1+(crop_size/2)))
            center_y=random.randint(int(y2-(crop_size/2)),int(y1+(crop_size/2)))
            crop_x1=max(0,int(center_x-crop_size/2))
            crop_y1=max(0,int(center_y-crop_size/2))
            crop_x2=min(img.shape[1],int(center_x+crop_size/2))
            crop_y2=min(img.shape[0],int(center_y+crop_size/2))
        
            cropped_img=img[crop_y1:crop_y2,crop_x1:crop_x2]
            transform_matrix = np.array([[1,0,-crop_x1],[0,1,-crop_y1],[0,0,1]])
            new_shapes =[]
            # new_point =[]
            pts = []
            # a = 0
            for idx,shape in enumerate(original_data['shapes']):
            # for shape in original_data['shapes']:
                # if idx >= box_idx and idx < box_idx+5:
                #按shape_type分发处理（retangle/polygon/point...）
                if shape['shape_type'] in ['rectangle','polygon']:
                    new_shape=process_polygon_or_rectangle(shape,transform_matrix,(crop_x1,crop_y1,crop_x2,crop_y2),iou_threshold)
                    if new_shape:
                        new_shapes.append(new_shape)
                        #存顶点用于后续修正
                        pts.extend(new_shape['points'])
                elif shape['shape_type'] == 'point':
                    new_point=process_point(shape,transform_matrix,cropped_img.shape[:2])
                    if new_point:
                        new_shapes.append(new_point)
                        pts.append(new_point['points'][0])

            base_name = os.path.splitext(original_data['imagePath'])[0]
            suffix = f'_crop{id}_outward{outward}'
            cropped_img_name = f"{base_name}{suffix}.jpg"       
            cropped_json_name =f"{base_name}{suffix}.json"
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

                #返回当前标注处理后的关键信息，用于关联
            return{
                    "original_shape":box_shape,
                    "cropped_shape_info":new_shapes,
                    "cropped_img_path":os.path.join(cropped_img_dir,cropped_img_name),
                    "cropped_json_path":os.path.join(cropped_jsons_dir,cropped_json_name)
                }
        

                # if 1:
                #     if shape['shape_type'] =='rectangle':
                #         new_shape =process_rectangle(shape,transform_matrix,(crop_x1,crop_y1,crop_x2,crop_y2),iou_threshold)
                #         if new_shape:
                #             new_shapes.append(new_shape)
                #     elif shape['shape_type'] == 'point':
                #         # if a>3 :
                #         #     continue
                #         new_point =process_point(shape,transform_matrix,cropped_img.shape[:2],new_shapes)
                #         if new_point:
                #              new_shapes.append(new_point)
                #              pts.append(new_point['points'][0])
                #             #  a=a+1
                #         else:
                #             continue
                # else:
                #     continue
            
            #（可选）用所有点重新计算最小包围盒，覆盖原标注
            # if pts:
            #     min_x,min_y=np.min(pts,axis=0)
            #     max_x,max_y=np.max(pts,axis=0)
            #     #替换为新的包围盒
            #     new_bbox=[[min_x,min_y],[max_x,min_y],[max_x,max_y],[min_x,max_y]]
            #     for s in new_shapes:
            #         if s['shape_type'] in ['rectangle','polygon']:
            #             s['points']=new_bbox
            #             s['shape_type']='polygon'

            
            # if len(new_shapes) == 5:
            #     t = img_path
            #     new_x1,new_y1 =np.array(pts[0])
            #     new_x2,new_y2=np.array(pts[1])
            #     new_x3,new_y3 =np.array(pts[2])
            #     new_x4,new_y4 =np.array(pts[3])

            #     xmin=min(new_x1,new_x2,new_x3,new_x4) - 2
            #     xmax=max(new_x1,new_x2,new_x3,new_x4) + 2
            #     ymin=min(new_y1,new_y2,new_y3,new_y4) - 2
            #     ymax=max(new_y1,new_y2,new_y3,new_y4) + 2
            #     new_points =[[xmin,ymin],[xmax,ymax]]
            #     for shape in new_shapes:
            #         if shape['shape_type'] == 'rectangle':
            #            new_shapes[0]['points'] =new_points



                
#矩形标注转换（判断原始矩形标注与裁剪区域的交集，转换矩阵坐标到裁剪后图像，输出新矩阵标注）
def process_polygon_or_rectangle(shape,matirix,crop_region,iou_threshold):
    #统一用多边形处理（retangle可视为特殊的polygon）
    points = np.array(shape['points'])
    x1,y1 =np.min(points,axis=0)
    x2,y2 =np.max(points,axis=0)
    #构造原始多边形
    original_poly = Polygon([(x1,y1),(x2,y1),(x2,y2),(x1,y2)])
    crop_x1,crop_y1,crop_x2,crop_y2 = crop_region
    #新增面积检查
    if original_poly.area == 0:
        print(f"标注{shape['label']}对应的多边形面积为0，跳过该标注处理，顶点数据：{points}")
        return None
    #构造裁剪区域多边形（矩形）
    crop_poly=Polygon([(crop_x1,crop_y1),(crop_x2,crop_y1),(crop_x2,crop_y2),(crop_x1,crop_y2)])
    intersection=original_poly.intersection(crop_poly)
    if intersection.is_empty:
        return None
    #用面积计算iou
    iou =intersection.area / original_poly.area
    if iou <iou_threshold:
        return None
    
    #转换所有顶点坐标
    new_points=[]
    for x,y in points:
        pt = np.dot(matirix,[x,y,1])[:2]
        new_x =np.clip(pt[0],0,crop_x2-crop_x1)
        new_y =np.clip(pt[1],0,crop_y2-crop_y1)
        new_points.append([float(new_x),float(new_y)])
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
#def process_point(shape,matrix,img_size,valid_boxes):
def process_point(shape,matrix,img_size):
    x,y =shape['points'][0]
    pt=np.dot(matrix,[x,y,1])[:2]
    new_x=pt[0]
    new_y=pt[1]
    if not (0<= new_x <img_size[1] and 0<= new_y<img_size[0]):
        return None
    
    #  裁剪图内添加所有关键点
    return{
            "label":shape["label"],
            "points":[[new_x,new_y]],
            "group_id":None,
            "description": " ",
            "shape_type":"point",
            "flags":shape["flags"],
            "mask":None,
        } 

    # #  裁剪图内添加矩形框'rectangle'内关键点
    # for box in valid_boxes:
    #     if box['shape_type'] == 'rectangle':
    #         box_points = np.array(box['points'])
    #         x1,y1 =np.min(box_points,axis=0)
    #         x2,y2 =np.max(box_points,axis=0)
    #         if x1 <= new_x <= x2 and y1 <=new_y <= y2:
    #             return{
    #                 "label":shape["label"],
    #                 "points":[[new_x,new_y]],
    #                 "group_id":None,
    #                 "description": " ",
    #                 "shape_type":"point",
    #                 "flags":shape["flags"],
    #                 "mask":None,
    #             }  
    # return None

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
    center =(w/2,h/2)
    M=cv2.getRotationMatrix2D(center,angle,1.0)
    cos = np.abs(M[0,0])
    sin = np.abs(M[0,1])
    new_w =int((h*sin)+(w*cos))
    new_h =int((h*cos)+(w*sin))
    #调整旋转矩阵，让旋转后的图像居中显示
    M[0,2] += (new_w-w)/2
    M[1,2] += (new_h-h)/2
    rotated_img= cv2.warpAffine(img,M,(new_w,new_h),
    flags=cv2.INTER_LINEAR,
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=(127,127,127)
    )

    return rotated_img,M,new_w,new_h

def transform_polygon(points,M,img_size):
    #复用矩阵变换多边形，矩阵坐标变换
    points_homogeneous = np.hstack([np.array(points,dtype=np.float32),np.ones((len(points),1),dtype=np.float32)])
    # x1,y1=points[0]
    # x2,y2=points[1]
    # points=np.array([
    #     [x1,y1],[x2,y1],[x2,y2],[x1,y2]
    # ],dtype=np.float32)
    # ones = np.ones((4,1))
    # points_homogeneous = np.hstack([points,ones])
    transformed = np.dot(M,points_homogeneous.T).T
    x_coords = transformed[:,0]
    y_coords = transformed[:,1]
    #裁剪到图像尺寸范围内
    x_coords=np.clip(x_coords,0,img_size[0])+5
    y_coords=np.clip(y_coords,0,img_size[1])+5
    return [[float(x),float(y)] for x,y in zip(x_coords,y_coords)]

    # new_x1 =np.clip(np.min(x_coords),0,img_size[0])
    # new_y1 =np.clip(np.min(y_coords),0,img_size[1])
    # new_x2 =np.clip(np.max(x_coords),0,img_size[0])
    # new_y2 =np.clip(np.max(y_coords),0,img_size[1])
    # return [[new_x1,new_y1],[new_x2,new_y2]]

def transform_point(point,M):
    #单个点坐标变换
    homogeneous=np.array([point[0],point[1],1])
    transformed =np.dot(M,homogeneous)
    return transformed.tolist()

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
            new_points=transform_polygon(rect_points,M,(new_w,new_h))
            new_type="polygon"

        # #区分形状类型处理：矩形转多边形，或直接处理多边形
        # if cropped_type =='rectangle':
        #     #矩阵->转4顶点多边形
        #     x1,y1=cropped_points[0]
        #     x2,y2=cropped_points[1]
        #     rect_points=[[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
        #     #调用transform_rectangle，将矩形转成变换后的矩形
        #     new_points = transform_polygon(rect_points,M,(new_w,new_h))
        #     #从多边形还原矩阵（取最小最大坐标）
        #     min_x=min(p[0] for p in new_points)
        #     max_x=max(p[0] for p in new_points)
        #     min_y=min(p[1] for p in new_points)
        #     max_y=max(p[1] for p in new_points)
        #     new_points=[[min_x,min_y],[max_x,max_y]]
        #     #记录形状类型
        #     new_type ='rectangle'
        else:
            #非矩形（如多边形）：逐个点变换
            #对多边形的每个店调用transfoem_point做坐标变换
               new_points =transform_polygon(cropped_points,M,(new_w,new_h))
               new_type=cropped_type
        #过滤超出旋转后图像范围的点
        valid_points =[]
        for point in new_points:
            x,y=point
            #检查坐标是否在旋转后图像尺寸内
            if 0<= x<=new_w and 0 <= y<=new_h:
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
            # 'cropped_type':cropped_type,
            # #变换+过滤后的坐标
            # 'new_points':valid_points,
            # 'shape_type':new_type
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
    # #遍历变换后的标注，构造最终shape格式
    # for item in transformerd_data:
    #     new_shape ={
    #         #保留原始目标类别
    #         "label":item['label'],
    #         #旋转后的标注坐标
    #         "points":item['new_points'],
    #         #保留形状类型
    #         "shape_type":item['shape_type'],
    #         #形状标记
    #         "flags":item.get("flags",{})
    #     }
    #     #如果有分组ID，保留分组信息
    #     if 'group_id' in item:
    #         #将新shape加入标注
    #         new_shape['group_id']=item['group_id']
    #         new_data['shapes'].append(new_shape)
    with open(os.path.join(output_json_dir,new_json_name),'w',encoding='utf-8') as f:
        json.dump(new_data,f,ensure_ascii=False,indent=2)




if __name__ == '__main__':
    
    # original_dir = Path(r"D:\work\lzm\QRDM_dataset\dataset\raw")
    # original_images_dir = original_dir.joinpath("images")
    # original_jsons_dir = original_dir.joinpath("jsons")
    # cropped_img_dir = original_dir.joinpath("crop_images")#裁剪
    # cropped_jsons_dir = original_dir.joinpath("crop_jsons")
    # outward = 2  # 向外扩展的比例为(1+1/outward)    #2
    # crop_and_adjust_jsons(original_images_dir,original_jsons_dir,cropped_img_dir,cropped_jsons_dir, outward, iou_threshold=0.5)
    
    
    # #旋转
    raw_dir = Path(r"D:\work\lzm\QRDM_dataset\dataset\raw")
    raw_images_dir = raw_dir.joinpath("images")
    raw_jsons_dir = raw_dir.joinpath("jsons")
    output_img_dir = raw_dir.joinpath("rotate_images")
    output_json_dir = raw_dir.joinpath("rotate_jsons")
    #旋转角度列表
    x=[30,60]  # [30,60,-30,-60]
    rotate_and_adjust_jsons(raw_images_dir,raw_jsons_dir,output_img_dir,output_json_dir)
    