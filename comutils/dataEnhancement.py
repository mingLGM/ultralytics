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
      id = 0
      for idx,shape in enumerate(data['shapes']):
        if shape['shape_type'] !='rectangle':
            continue
        process_single_crop(img,data,shape,idx,original_images_dir,cropped_img_dir,cropped_jsons_dir,iou_threshold, img_path, id, outward)
        id = id +1
def process_single_crop(img,original_data,box_shape,box_idx,original_images_dir,cropped_img_dir,cropped_jsons_dir,iou_threshold, img_path, id, outward):
            points = np.array(box_shape['points'],dtype=np.float32)
            x1,y1 = np.min(points,axis=0)
            x2,y2 = np.max(points,axis=0)
           
            box_width=x2-x1
            box_height=y2-y1
            crop_size=max(box_width,box_height)*outward
            center_x=random.randint(int(x2-(crop_size/2)),int(x1+(crop_size/2)))
            center_y=random.randint(int(y2-(crop_size/2)),int(y1+(crop_size/2)))
            crop_x1=max(0,int(center_x-crop_size/2))
            crop_y1=max(0,int(center_y-crop_size/2))
            crop_x2=min(img.shape[1],int(center_x+crop_size/2))
            crop_y2=min(img.shape[0],int(center_y+crop_size/2))
        
            cropped_img=img[crop_y1:crop_y2,crop_x1:crop_x2]
            transform_matrix = np.array([[1,0,-crop_x1],[0,1,-crop_y1],[0,0,1]])
            new_shapes =[]
            new_point =[]
            pts = []
            # a = 0
            for idx,shape in enumerate(original_data['shapes']):
            # for shape in original_data['shapes']:
                # if idx >= box_idx and idx < box_idx+5:
                if 1:
                    if shape['shape_type'] =='rectangle':
                        new_shape =process_rectangle(shape,transform_matrix,(crop_x1,crop_y1,crop_x2,crop_y2),iou_threshold)
                        if new_shape:
                            new_shapes.append(new_shape)
                    elif shape['shape_type'] == 'point':
                        # if a>3 :
                        #     continue
                        new_point =process_point(shape,transform_matrix,cropped_img.shape[:2],new_shapes)
                        if new_point:
                             new_shapes.append(new_point)
                             pts.append(new_point['points'][0])
                            #  a=a+1
                else:
                    continue
            
            if len(new_shapes) == 5:
                t = img_path
                new_x1,new_y1 =np.array(pts[0])
                new_x2,new_y2=np.array(pts[1])
                new_x3,new_y3 =np.array(pts[2])
                new_x4,new_y4 =np.array(pts[3])

                xmin=min(new_x1,new_x2,new_x3,new_x4) - 2
                xmax=max(new_x1,new_x2,new_x3,new_x4) + 2
                ymin=min(new_y1,new_y2,new_y3,new_y4) - 2
                ymax=max(new_y1,new_y2,new_y3,new_y4) + 2
                new_points =[[xmin,ymin],[xmax,ymax]]
                for shape in new_shapes:
                    if shape['shape_type'] == 'rectangle':
                       new_shapes[0]['points'] =new_points



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
        

def process_rectangle(shape,matirix,crop_region,iou_threshold):
    points = np.array(shape['points'])
    x1,y1 =np.min(points,axis=0)
    x2,y2 =np.max(points,axis=0)
    
    original_poly = Polygon([(x1,y1),(x2,y1),(x2,y2),(x1,y2)])
    crop_x1,crop_y1,crop_x2,crop_y2 = crop_region
    crop_poly=Polygon([(crop_x1,crop_y1),(crop_x2,crop_y1),(crop_x2,crop_y2),(crop_x1,crop_y2)])
    intersection=original_poly.intersection(crop_poly)
    if intersection.is_empty:
        return None
    iou =intersection.area / original_poly.area
    if iou <iou_threshold:
        return None
    
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
        "shape_type":"rectangle",
        "flags":shape["flags"],
        "mask":None,
    }
def process_point(shape,matrix,img_size,valid_boxes):
    x,y =shape['points'][0]
    pt=np.dot(matrix,[x,y,1])[:2]
    new_x=pt[0]
    new_y=pt[1]
    if not (0<= new_x <img_size[1] and 0<= new_y<img_size[0]):
        return None

    for box in valid_boxes:
        if box['shape_type'] == 'rectangle':
            box_points = np.array(box['points'])
            x1,y1 =np.min(box_points,axis=0)
            x2,y2 =np.max(box_points,axis=0)
            if x1 <= new_x <= x2 and y1 <=new_y <= y2:
                return{
                    "label":shape["label"],
                    "points":[[new_x,new_y]],
                    "group_id":None,
                    "description": " ",
                    "shape_type":"point",
                    "flags":shape["flags"],
                    "mask":None,
                }  
    return None

def rotate_and_adjust_jsons(cropped_img_dir,cropped_jsons_dir,output_img_dir,output_json_dir):
    os.makedirs(output_img_dir,exist_ok=True)

    os.makedirs(output_json_dir,exist_ok=True)

    for json_file in os.listdir(cropped_jsons_dir) :
        if not json_file.endswith('.json'):
            continue
        cropped_path = os.path.join(cropped_jsons_dir,json_file,)
        with open(cropped_path,'r',encoding='utf-8')as f:
            cropped_data =json.load(f)

        cropped_img_name = cropped_data['imagePath']
        cropped_img_path = os.path.join(cropped_img_dir,cropped_img_name)
        cropped_img = cv2.imread(cropped_img_path)
        if cropped_img is None:
            print(f"警告，无法读取图像{cropped_img_path}")
            continue
        for i in x:
            angle =random.randint(-i, i)
            rotate_and_save(
                img=cropped_img,
                data=cropped_data,
                output_json_dir=output_json_dir,
                angle=angle,
                rotation_idx = i
            )

def rotate_img (img,angle):
    h,w = img.shape[:2]
    center =(w/2,h/2)
    M=cv2.getRotationMatrix2D(center,angle,1.0)
    cos = np.abs(M[0,0])
    sin = np.abs(M[0,1])
    new_w =int((h*sin)+(w*cos))
    new_h =int((h*cos)+(w*sin))
    M[0,2] += (new_w-w)/2
    M[1,2] += (new_h-h)/2
    rotated_img= cv2.warpAffine(img,M,(new_w,new_h),
    flags=cv2.INTER_LINEAR,
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=(127,127,127)
    )

    return rotated_img,M,new_w,new_h

def transform_rectangle(points,M,img_size):
    x1,y1=points[0]
    x2,y2=points[1]
    points=np.array([
        [x1,y1],[x2,y1],[x2,y2],[x1,y2]
    ],dtype=np.float32)
    ones = np.ones((4,1))
    points_homogeneous = np.hstack([points,ones])
    transformed = np.dot(M,points_homogeneous.T).T
    x_coords = transformed[:,0]
    y_coords = transformed[:,1]
    new_x1 =np.clip(np.min(x_coords),0,img_size[0])
    new_y1 =np.clip(np.min(y_coords),0,img_size[1])
    new_x2 =np.clip(np.max(x_coords),0,img_size[0])
    new_y2 =np.clip(np.max(y_coords),0,img_size[1])
    return [[new_x1,new_y1],[new_x2,new_y2]]

def transform_point(point,M):
    x,y=point
    homogeneous=np.array([x,y,1])
    transformed =np.dot(M,homogeneous)
    return transformed.tolist()

def process_annotations(data,M,new_w,new_h):
    transformed_data =[]
    for shape in data['shapes']:
        cropped_type = shape['shape_type']
        cropped_points = shape['points']

        if cropped_type =='rectangle':
            new_points = transform_rectangle(cropped_points,M,(new_w,new_h))
            new_type ='rectangle'
        else:
               new_points =[transform_point(p,M) for p in cropped_points]
               new_type=cropped_type
        valid_points =[]
        for point in new_points:
            x,y=point[0],point[1]
            if(0<= x<=new_w)and (0 <= y<=new_h):
                valid_points.append([float(x),float(y)])
        transformed_data.append({
            'label':shape["label"],
            'type':cropped_type,
            'cropped_type':cropped_type,
            'new_points':valid_points,
            'shape_type':new_type
        })
    return transformed_data
def rotate_and_save(img,data,output_json_dir,angle,rotation_idx):
    rotated_img,M,new_w,new_h = rotate_img(img,angle)
    transformerd_data =process_annotations(copy.deepcopy(data),M,new_w,new_h)
    base_name = os.path.splitext(data['imagePath'])[0]
    suffix = f'_rotate{rotation_idx}_{int(angle)}'
    new_img_name = f"{base_name}{suffix}.jpg"
    new_json_name = f"{base_name}{suffix}.json"
    cv2.imwrite(os.path.join(output_img_dir,new_img_name),rotated_img)
    new_data ={
        "version":data["version"],
        "flags":data["flags"],
        "imagePath":new_img_name,
        "imageData":None,
        "imageHeight":new_h,
        "imageWidth":new_w,
        "shapes":[]
    }
    for item in transformerd_data:
        new_shape ={
            "label":item['label'],
            "points":item['new_points'],
            "shape_type":item['shape_type'],
            "flags":{}
        }
        if'group_id' in item:
            new_shape['group_id']=item['group_id']
        new_data['shapes'].append(new_shape)
    with open(os.path.join(output_json_dir,new_json_name),'w')as f:
        json.dump(new_data,f,indent=2)




if __name__ == '__main__':
    
    original_dir = Path(r"E:\work\Data\QR\temp\raw")
    original_images_dir = original_dir.joinpath("images")
    original_jsons_dir = original_dir.joinpath("jsons")
    cropped_img_dir = original_dir.joinpath("crop_images")
    cropped_jsons_dir = original_dir.joinpath("crop_jsons")
    outward = 8
    crop_and_adjust_jsons(original_images_dir,original_jsons_dir,cropped_img_dir,cropped_jsons_dir, outward, iou_threshold=0.5)
    
    
    
    # raw_dir = Path(r"E:\work\Data\QR\temp\raw")
    # raw_images_dir = raw_dir.joinpath("images")
    # raw_jsons_dir = raw_dir.joinpath("jsons")
    # output_img_dir = raw_dir.joinpath("rotate_images")
    # output_json_dir = raw_dir.joinpath("rotate_jsons")
    # x=[20]  # [30,60,-30,-60]
    # rotate_and_adjust_jsons(raw_images_dir,raw_jsons_dir,output_img_dir,output_json_dir)
    