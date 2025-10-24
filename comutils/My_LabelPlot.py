import os
import csv
import cv2
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
import shutil

def visualize_yolo_detect_labels_PIL(input_folder, out_folder):
    """
    从 YOLO 格式的标签文件在图像上可视化边界框。功能不全，不建议使用
    Args:
        input_folder (str): 包含 'images' 和 'labels' 子文件夹的输入目录路径。
                            'images' 文件夹存放图像文件。
                            'labels' 文件夹存放 YOLO 格式的 .txt 标签文件。
        out_folder (str): 用于保存带有边界框的可视化图像的输出目录路径。
                          如果目录不存在，将会被自动创建。
    """
    
    # 查找images和labels文件夹
    image_dir = os.path.join(input_folder, 'images')
    label_dir = os.path.join(input_folder, 'labels')
    
    if not os.path.exists(image_dir):
        print(f"未找到images文件夹: {image_dir}")
        return
    
    if not os.path.exists(label_dir):
        print(f"未找到labels文件夹: {label_dir}")
        return
    
    """从YOLO标签文件可视化边界框"""
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
        print(f"文件夹 {out_folder} 已创建")
    else:
        print(f"文件夹 {out_folder} 已存在")
    
    # 支持的图像格式
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    # 定义颜色（RGB格式）
    colors = ["red", "green", "blue", "yellow", "purple", "orange"]
    
    # 遍历图像文件夹
    for filename in os.listdir(label_dir):
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext in image_extensions:
            # 构造对应的标签文件名
            label_filename = os.path.join(label_dir, os.path.splitext(filename)[0] + ".txt")
            
            if os.path.exists(label_filename):
                try:
                    # 读取图像
                    image_path = os.path.join(image_dir, filename)
                    image = Image.open(image_path)
                    draw = ImageDraw.Draw(image)

                    # 读取标签文件的内容并绘制边界框
                    with open(label_filename, 'r', encoding='utf-8') as f:
                        for line_num, line in enumerate(f):
                            line = line.strip()
                            if not line:
                                continue
                                
                            try:
                                parts = line.split()
                                if len(parts) >= 5:
                                    class_id, x_center, y_center, width, height = map(float, parts[:5])
                                    
                                    # 转换为绝对坐标
                                    x_center_abs = x_center * image.width
                                    y_center_abs = y_center * image.height
                                    width_abs = width * image.width
                                    height_abs = height * image.height
                                    
                                    # 计算边界框坐标
                                    x1 = x_center_abs - width_abs / 2
                                    y1 = y_center_abs - height_abs / 2
                                    x2 = x1 + width_abs
                                    y2 = y1 + height_abs
                                    
                                    # 确保坐标在图像范围内
                                    x1 = max(0, x1)
                                    y1 = max(0, y1)
                                    x2 = min(image.width, x2)
                                    y2 = min(image.height, y2)
                                    
                                    # 绘制边界框
                                    color = colors[class_id % len(colors)]
                                    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                                    
                                    # 添加类别标签（可选）
                                    label_text = f"Class {int(class_id)} ({int(x1)},{int(y1)})-({int(x2)},{int(y2)})"
                                    draw.text((x1, max(y1-20, 5)), label_text, fill=color)
                                    
                            except Exception as e:
                                print(f"处理标签文件 {label_filename} 第 {line_num+1} 行时出错: {e}")
                                continue

                    # 保存带结果的图像，保持原图格式和质量
                    output_filename = os.path.join(out_folder, filename)
                    image.save(output_filename, quality=95)
                    print(f"已处理: {filename}")
                    
                except Exception as e:
                    print(f"处理图像 {filename} 时出错: {e}")
            else:
                print(f"未找到标签文件: {os.path.basename(label_filename)}")

def visualize_yolo_detect_labels_from_csv(csv_file, input_folder, out_folder, has_header=False):
    # 查找images
    image_dir = os.path.join(input_folder, 'images')
    
    if not os.path.exists(image_dir):
        print(f"未找到images文件夹: {image_dir}")
        return
    
    """从CSV文件可视化边界框"""
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
        print(f"文件夹 {out_folder} 已创建")
    else:
        print(f"文件夹 {out_folder} 已存在")
    
    # 按图像文件名分组边界框，避免重复读取图像
    image_boxes = {}
    
    # 读取CSV文件并分组
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            csv_reader = csv.reader(f)
            
            if has_header:
                next(csv_reader)  # 跳过标题行
                
            for row_num, row in enumerate(csv_reader):
                if len(row) < 5:  # 至少需要文件名和4个坐标值
                    print(f"CSV第 {row_num+1} 行列数不足，跳过")
                    continue
                    
                try:
                    filename = row[0]
                    # 从第2列开始是坐标（跳过文件名和可能的类别ID）
                    x_center, y_center, width, height = map(float, row[1:5])
                    
                    if filename not in image_boxes:
                        image_boxes[filename] = []
                    
                    image_boxes[filename].append((x_center, y_center, width, height))
                    
                except Exception as e:
                    print(f"解析CSV第 {row_num+1} 行时出错: {e}")
                    continue
    except Exception as e:
        print(f"读取CSV文件时出错: {e}")
        return
    
    # 处理每个图像
    processed_count = 0
    for filename, boxes in image_boxes.items():
        try:
            # 查找图像文件（支持不同扩展名）
            image_path = None
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
            
            for ext in image_extensions:
                potential_path = os.path.join(image_dir, filename)
                # 如果文件名已经包含扩展名
                if os.path.splitext(filename)[1].lower() in image_extensions:
                    potential_path = os.path.join(image_dir, filename)
                else:
                    potential_path = os.path.join(image_dir, filename + ext)
                
                if os.path.exists(potential_path):
                    image_path = potential_path
                    break
            
            if image_path is None:
                print(f"未找到图像文件: {filename}")
                continue
            
            # 读取图像
            image = Image.open(image_path)
            draw = ImageDraw.Draw(image)
            
            # 绘制所有边界框
            for i, (x_center, y_center, width, height) in enumerate(boxes):
                # 根据图像大小调整坐标
                x_center_abs = x_center * image.width
                y_center_abs = y_center * image.height
                width_abs = width * image.width
                height_abs = height * image.height
                
                # 计算边界框坐标
                x1 = x_center_abs - width_abs / 2
                y1 = y_center_abs - height_abs / 2
                x2 = x1 + width_abs
                y2 = y1 + height_abs
                
                # 确保坐标在图像范围内
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(image.width, x2)
                y2 = min(image.height, y2)
                
                # 绘制边界框
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                
                # 添加序号标签（可选）
                label_text = f"Box {i+1}"
                draw.text((x1, y1-15), label_text, fill="red")
            
            # 保存结果图像，保持原图格式和质量
            output_filename = os.path.join(out_folder, os.path.basename(image_path))
            image.save(output_filename, quality=95)
            print(f"已处理: {os.path.basename(image_path)} (包含 {len(boxes)} 个边界框)")
            processed_count += 1
            
        except Exception as e:
            print(f"处理图像 {filename} 时出错: {e}")
    
    print(f"CSV处理完成，共处理 {processed_count} 个图像")

def visualize_yolo_detect_singlelabel_CV(image_path, label_path, output_path, colors):
    """
    使用OpenCV可视化单张图像的YOLO格式标签
    
    功能说明:
    - 读取图像和对应的YOLO标签文件
    - 将归一化的YOLO坐标转换为绝对像素坐标
    - 在图像上绘制彩色边界框和标签信息
    - 保持原图格式和质量保存可视化结果
    - 支持中文文件路径
    
    参数:
        image_path (str): 输入图像文件路径
        label_path (str): YOLO标签文件路径(.txt格式)
        output_path (str): 输出可视化图像保存路径
        colors (list): BGR格式颜色列表，用于不同目标的边界框
    
    返回:
        bool: 处理成功返回True，失败返回False
    
    标签格式说明:
        每行格式: class_id x_center y_center width height
        所有坐标值都是归一化到[0,1]的浮点数
    """
    
    # 读取图像
    image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), -1)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return False
    
    # 读取标签文件
    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            yolo_labels = f.readlines()
    except:
        print(f"无法读取标签文件: {label_path}")
        return False
    
    # 创建图像的副本，避免修改原图数据
    image_with_boxes = image.copy()
    
    # 解析并绘制每个边界框
    for i, line in enumerate(yolo_labels):
        line = line.strip()
        if not line:
            continue
            
        parts = line.split()
        if len(parts) < 5:
            continue
            
        try:
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            # 将归一化坐标转换为绝对坐标
            x_center_abs = int(x_center * image.shape[1])
            y_center_abs = int(y_center * image.shape[0])
            width_abs = int(width * image.shape[1])
            height_abs = int(height * image.shape[0])
            
            # 计算边界框的左上角和右下角坐标
            x1 = int(x_center_abs - width_abs / 2)
            y1 = int(y_center_abs - height_abs / 2)
            x2 = int(x_center_abs + width_abs / 2)
            y2 = int(y_center_abs + height_abs / 2)
            
            # 确保坐标在图像范围内
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.shape[1] - 1, x2)
            y2 = min(image.shape[0] - 1, y2)
            
            # 绘制边界框
            color = colors[i % len(colors)]
            cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), color, 2)
            
            # 添加类别和坐标信息
            label = f"ID:{class_id} ({x1},{y1})-({x2},{y2})"
            cv2.putText(image_with_boxes, label, (x1, max(y1-10, 10)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        except Exception as e:
            print(f"处理标签时出错 (文件: {label_path}, 行: {line}): {e}")
            continue
    
    # 保存处理后的图像，保持原图格式和质量
    try:
        # 获取原图扩展名
        original_ext = Path(image_path).suffix.lower()
        
        # 根据原图格式选择保存参数
        if original_ext in ['.jpg', '.jpeg']:
            # JPEG格式，使用高质量压缩
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
            success, encoded_image = cv2.imencode(original_ext, image_with_boxes, encode_param)
        elif original_ext in ['.png']:
            # PNG格式，无损压缩
            encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 0]
            success, encoded_image = cv2.imencode(original_ext, image_with_boxes, encode_param)
        else:
            # 其他格式，使用默认参数
            success, encoded_image = cv2.imencode(original_ext, image_with_boxes)
        
        if success:
            # 处理中文路径
            with open(output_path, 'wb') as f:
                encoded_image.tofile(f)
            return True
        else:
            print(f"图像编码失败: {output_path}")
            return False
            
    except Exception as e:
        print(f"保存图像失败: {output_path}, 错误: {e}")
        return False

def visualize_yolo_detect_labels_CV(input_folder, output_folder):
    """
    批量处理YOLO数据集文件夹的可视化
    
    功能说明:
    - 遍历指定文件夹中的images和labels子文件夹
    - 为每个有标签的图像生成可视化结果
    - 对无标签的图像进行标记并复制到输出文件夹
    - 自动创建输出文件夹并统计处理结果
    
    目录结构要求:
        input_folder/
        ├── images/      # 存放图像文件
        └── labels/      # 存放对应的YOLO标签文件(.txt)
    
    参数:
        input_folder (str): 输入数据集文件夹路径，包含images和labels子文件夹
        output_folder (str): 输出可视化结果保存文件夹路径
    
    返回:
        None
    
    输出说明:
        - 有标签图像: 保存为"原文件名_visualized.原扩展名"
        - 无标签图像: 保存为"原文件名_no_labels.原扩展名"
        - 控制台输出处理统计信息
    """
    
    # 定义颜色（BGR格式）
    colors = [(0, 255, 0),   # 绿色
              (0, 0, 255),   # 红色
              (255, 0, 0),   # 蓝色
              (0, 255, 255), # 黄色
              (255, 0, 255), # 粉色
              (255, 255, 0)] # 青色
    
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 查找images和labels文件夹
    images_dir = os.path.join(input_folder, 'images')
    labels_dir = os.path.join(input_folder, 'labels')
    
    if not os.path.exists(images_dir):
        print(f"未找到images文件夹: {images_dir}")
        return
    
    if not os.path.exists(labels_dir):
        print(f"未找到labels文件夹: {labels_dir}")
        return
    
    # 获取所有图像文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(images_dir).glob(f'*{ext}'))
        image_files.extend(Path(images_dir).glob(f'*{ext.upper()}'))
    
    print(f"找到 {len(image_files)} 个图像文件")
    
    processed_count = 0
    error_count = 0
    
    # 处理每个图像文件
    for image_path in image_files:
        # 构建对应的标签文件路径
        label_filename = image_path.stem + '.txt'
        label_path = Path(labels_dir) / label_filename
        
        # 构建输出文件路径，保持原图格式
        original_ext = image_path.suffix
        output_filename = image_path.stem + '_visualized' + original_ext
        output_path = Path(output_folder) / output_filename
        
        if label_path.exists():
            # 可视化标签并保存
            if visualize_yolo_detect_singlelabel_CV(image_path, label_path, output_path, colors):
                print(f"已处理: {image_path.name} -> {output_filename}")
                processed_count += 1
            else:
                print(f"处理失败: {image_path.name}")
                error_count += 1
        else:
            # 如果没有标签文件，直接复制原图到输出文件夹
            output_filename = image_path.stem + '_no_labels' + original_ext
            output_path = Path(output_folder) / output_filename
            shutil.copy2(image_path, output_path)
            print(f"无标签文件，已复制: {image_path.name} -> {output_filename}")
            processed_count += 1
    
    print(f"\n处理完成!")
    print(f"成功处理: {processed_count} 个文件")
    print(f"处理失败: {error_count} 个文件")
    print(f"输出文件夹: {output_folder}")


def main():
    # 定义路径
    image_dir = r"E:\项目\薄膜\bomo3-yolo数据格式\test"
    out_folder = r"E:\项目\薄膜\bomo3-yolo数据格式\test\out_result"
    csv_file = r"E:\项目\薄膜\testResult.csv"
    
    # ### 选择可视化方式
    # print("请选择可视化方式:")
    # print("1. 从YOLO标签文件可视化")
    # print("2. 从CSV文件可视化")
    # choice = input("请输入选择 (1 或 2): ").strip()
    choice = "1"

    if choice == "1":
        visualize_yolo_detect_labels_CV(image_dir, out_folder)
    elif choice == "2":
        visualize_yolo_detect_labels_PIL(image_dir, out_folder)
    elif choice == "3":
        # 询问CSV是否有标题行
        has_header = input("CSV文件是否有标题行? (y/n): ").strip().lower() == 'y'
        visualize_yolo_detect_labels_from_csv(csv_file, image_dir, out_folder, has_header)
    else:
        print("无效选择")

if __name__ == "__main__":
    main()