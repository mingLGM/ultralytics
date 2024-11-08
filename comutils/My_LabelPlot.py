import os
import csv
from PIL import Image, ImageDraw

# 定义图像文件夹路径和标签文件夹路径
image_folder = r"E:\项目\薄膜\bomo3-yolo数据格式\test\images"
label_folder = r"E:\项目\薄膜\bomo3-yolo数据格式\test\labels"
csv_file     = r"E:\项目\薄膜\testResult.csv"
out_folder   = r"E:\项目\薄膜\bomo3-yolo数据格式\test\out_result"

if not os.path.exists(out_folder):
    os.makedirs(out_folder)
    print(f"文件夹 {out_folder} 已创建")
else:
    print(f"文件夹 {out_folder} 已存在")


# # 遍历图像文件夹
# for filename in os.listdir(image_folder):
#     if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
#         # 构造对应的标签文件名
#         label_filename = os.path.join(label_folder, os.path.splitext(filename)[0] + ".txt")
#         if os.path.exists(label_filename):
#             # 读取图像
#             image_path = os.path.join(image_folder, filename)
#             image = Image.open(image_path)
#             draw = ImageDraw.Draw(image)

#             # 读取标签文件的内容并绘制边界框
#             with open(label_filename, 'r') as f:
#                 for line in f:
#                     class_id, x_center, y_center, width, height = map(float, line.split())
#                     x_center *= image.width
#                     y_center *= image.height
#                     width *= image.width
#                     height *= image.height
#                     x1 = (x_center - width / 2)
#                     y1 = (y_center - height / 2)
#                     x2 = x1 + width
#                     y2 = y1 + height
#                     draw.rectangle([x1, y1, x2, y2], outline="red", width=5)

#             # 保存带结果的图像
#             output_filename = os.path.join(out_folder, filename)
#             image.save(output_filename)



# 遍历CSV文件
with open(csv_file, 'r') as f:
    csv_reader = csv.reader(f)
    # next(csv_reader)  # 跳过标题行
    for row in csv_reader:
        filename = row[0]  # CSV 中的第一列是图像文件名
        # class_id = int(row[1])  # 第二列是类别ID
        x_center, y_center, width, height = map(float, row[1:])  # 第三列开始是坐标和尺寸信息

        # 读取图像
        image_path = os.path.join(image_folder, filename)
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)

        # 根据图像大小调整坐标
        width_img, height_img = image.size
        x_center *= image.width
        y_center *= image.height
        width *= image.width
        height *= image.height
        x1 = (x_center - width / 2)
        y1 = (y_center - height / 2)
        x2 = x1 + width
        y2 = y1 + height
        
        # width_img, height_img = image.size
        # x1 = x_center * width_img
        # y1 = y_center * height_img
        # x2 = width * width_img
        # y2 = height * height_img

        # 绘制边界框
        draw.rectangle([x1, y1, x2, y2], outline="red", width=5)

        # 保存带结果的图像
        output_filename = os.path.join(out_folder, filename)
        image.save(output_filename)


