import os
from PIL import Image
import shutil

def findSingleFile(path):
    num = 0
        # 创建 cutpictures 文件夹（先判断）
    cutp = os.path.join(path, "imgcrop")
        # 判断文件夹是否存在
    if os.path.exists(cutp):
        # 如果文件夹存在，先删除再创建
        # 递归删除文件夹
        shutil.rmtree(cutp)
        os.makedirs(cutp)
    else:
        # 如果文件夹不存在，直接创建
        os.makedirs(cutp)

    for filename in os.listdir(path):
        if not os.path.isdir(os.path.join(path,filename)):
            # print(filename)
            # 无后缀文件名 , 文件后缀
            filename_nosuffix, file_suffix = os.path.splitext(filename)
            # filename_nosuffix = filename.split(".")[0]
            # print(filename_nosuffix)
            # 文件后缀
            file_suffix = filename.split(".")[-1]
            # print(file_suffix)

            img_path = os.path.join(path,filename)
            label_path = img_path.replace('\\images\\', '\\labels\\').replace('.'+file_suffix, '.txt')
            # label_path = os.path.join(path,'labels',filename_nosuffix+".txt")

            # print(img_path)
            # print(label_path)
            # 生成裁剪图片（遍历 txt 每一行）eg: mask_0_1.jpg
            # 0 裁剪的图片序号 1 类别序号
            img = Image.open(img_path)
            w, h = img.size
            with open(label_path, 'r+', encoding='utf-8') as f:
                # 读取txt文件中的第一行，数据类型str
                lines = f.readlines()
                # 根据空格切割字符串，最后得到的是一个list
                for index, line in enumerate(lines):
                    msg = line.split(" ")
                    category = int(msg[0])
                    x_center = float(msg[1])
                    y_center = float(msg[2])
                    width = float(msg[3])
                    height = float(msg[4])
                    x1 = int((x_center - width / 2) * w)  # x_center - width/2
                    y1 = int((y_center - height / 2) * h)  # y_center - height/2
                    x2 = int((x_center + width / 2) * w)  # x_center + width/2
                    y2 = int((y_center + height / 2) * h)  # y_center + height/2
                    # print(x1, ",", y1, ",", x2, ",", y2, "," ,category)
                    # 保存图片
                    img_roi = img.crop((x1, y1, x2, y2))
                    cropimg_path = os.path.join(cutp, str(category))
                    if not os.path.exists(cropimg_path):
                        os.makedirs(cropimg_path)
                    save_path = os.path.join(cropimg_path, "{}_{}_{}.{}".format(filename_nosuffix, index, category, file_suffix))
                    img_roi.save(save_path)
                    num += 1

    print("裁剪图片数量：", num)
    print("裁剪图片存放目录：", cutp)


def main():
    # import argparse

    # # 创建 ArgumentParser 对象
    # parser = argparse.ArgumentParser(description='输入目标检测裁剪目录')
    # # 添加参数
    # parser.add_argument('--dir', help='目录名', required=True)
    # # 解析命令行参数
    # args = parser.parse_args()
    # dir = args.dir
    # # print('目录参数:', dir)

    dir = r'D:\lzm\work\ultralytics\datasets\zhitong\train\images'

    findSingleFile(dir)
    return


if __name__ == '__main__':
    main()

## python cutpictures.py --dir /home/hualiujie/baoxinshagnchuan/ultralytics-main-cgh/runs/detect/predict6

