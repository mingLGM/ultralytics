# 工具类
import os
import random
import shutil
from shutil import copy2

def data_set_split(src_data_folder, target_data_folder, train_scale=0.8, val_scale=0.2):
    '''
    读取源数据文件夹，生成划分好的文件夹，分为train、val两个文件夹进行
    :param src_data_folder: 源文件夹
    :param target_data_folder: 目标文件夹
    :param train_scale: 训练集比例
    :param val_scale: 验证集比例
    :return:
    '''
    print("开始数据集划分")
    class_names = os.listdir(src_data_folder)
    # 在目标目录下创建文件夹
    split_names = ['train', 'valid']
    for split_name in split_names:
        split_path = os.path.join(target_data_folder, split_name)
        if os.path.isdir(split_path):
            pass
        else:
            os.makedirs(split_path)
        # 然后在split_path的目录下创建类别文件夹
        for class_name in class_names:
            class_split_path = os.path.join(split_path, class_name)
            if os.path.isdir(class_split_path):
                pass
            else:
                os.makedirs(class_split_path)

    # 按照比例划分数据集，并进行数据图片的复制
    # 首先进行分类遍历
    for class_name in class_names:
        current_class_data_path = os.path.join(src_data_folder, class_name)
        current_all_data = os.listdir(current_class_data_path)
        current_data_length = len(current_all_data)
        current_data_index_list = list(range(current_data_length))
        random.shuffle(current_data_index_list)

        train_folder = os.path.join(os.path.join(target_data_folder, 'train'), class_name)
        val_folder = os.path.join(os.path.join(target_data_folder, 'val'), class_name)
        train_stop_flag = current_data_length * train_scale
        current_idx = 0
        train_num = 0
        val_num = 0
        for i in current_data_index_list:
            src_img_path = os.path.join(current_class_data_path, current_all_data[i])
            if current_idx <= train_stop_flag:
                copy2(src_img_path, train_folder)
                train_num = train_num + 1
            else:
                copy2(src_img_path, val_folder)
                val_num = val_num + 1

            current_idx = current_idx + 1

        print("*********************************{}*************************************".format(class_name))
        print("{}类按照{}：{}的比例划分完成，一共{}张图片".format(class_name, train_scale, val_scale, current_data_length))
        print("训练集{}：{}张".format(train_folder, train_num))
        print("验证集{}：{}张".format(val_folder, val_num))


def file_move(src_data_folder):
    '''
    读取源数据文件夹(包含image和label)，移动图像和对应标签
    :param src_data_folder: 源文件夹
    :return:
    '''
    label_folder = os.path.join(src_data_folder, "labels")
    images_folder = os.path.join(src_data_folder, "images")

    # 创建目标文件夹
    os.makedirs(label_folder, exist_ok=True)
    os.makedirs(images_folder, exist_ok=True)

    # 遍历源文件夹中的所有文件
    for file in os.listdir(src_data_folder):
        # 如果文件是txt文件
        if file.endswith(".txt"):
            txt_file_path = os.path.join(src_data_folder, file)
            # 移动txt文件到label文件夹
            shutil.move(txt_file_path, os.path.join(label_folder, file))

            # 获取对应的jpg文件路径
            jpg_file = file.replace(".txt", ".jpg")
            jpg_file_path = os.path.join(src_data_folder, jpg_file)

            # 如果对应的jpg文件存在
            if os.path.exists(jpg_file_path):
                # 移动jpg文件到images文件夹
                shutil.move(jpg_file_path, os.path.join(images_folder, jpg_file))
            # else:
            #     print(f"警告: 找不到对应的图像文件 {jpg_file}")
    print("文件移动完成！")

def split_dataset(source_folder, split_ratio=0.8):
    """
    将 source_folder 中的 images 和 labels 文件夹按比例划分为 train 和 test。
    
    :param source_folder: 源数据集文件夹路径
    :param split_ratio: 训练集划分比例（0-1），默认0.8
    """
    images_folder = os.path.join(source_folder, "images")
    labels_folder = os.path.join(source_folder, "labels")
    
    # 创建新的train和test目录
    train_images_folder = os.path.join(source_folder, "train", "images")
    train_labels_folder = os.path.join(source_folder, "train", "labels")
    test_images_folder = os.path.join(source_folder, "test", "images")
    test_labels_folder = os.path.join(source_folder, "test", "labels")

    os.makedirs(train_images_folder, exist_ok=True)
    os.makedirs(train_labels_folder, exist_ok=True)
    os.makedirs(test_images_folder, exist_ok=True)
    os.makedirs(test_labels_folder, exist_ok=True)
    
    # 获取所有的图像文件及其对应的标签文件
    image_files = [f for f in os.listdir(images_folder) if f.endswith(('.jpg', '.png'))]
    random.shuffle(image_files)  # 打乱顺序

    # 按照比例划分训练集和测试集
    split_index = int(len(image_files) * split_ratio)
    train_files = image_files[:split_index]
    test_files = image_files[split_index:]

    def move_files(file_list, src_images, src_labels, dest_images, dest_labels):
        for img_file in file_list:
            base_name = os.path.splitext(img_file)[0]
            label_file = base_name + ".txt"

            # 移动图片文件
            shutil.move(os.path.join(src_images, img_file), os.path.join(dest_images, img_file))

            # 移动对应的标签文件
            if os.path.exists(os.path.join(src_labels, label_file)):
                shutil.move(os.path.join(src_labels, label_file), os.path.join(dest_labels, label_file))

    # 将文件移动到对应的 train 和 test 目录
    move_files(train_files, images_folder, labels_folder, train_images_folder, train_labels_folder)
    move_files(test_files, images_folder, labels_folder, test_images_folder, test_labels_folder)

    print(f"数据集划分完成！训练集: {len(train_files)}，测试集: {len(test_files)}")


if __name__ == '__main__':
    
    # src_data_folder = "zhitong_cls/train_temp"
    # target_data_folder = "zhitong_cls"
    # data_set_split(src_data_folder, target_data_folder)


    src_data_folder = r"E:\铭\workspace\布缝\traindata\jiefeng20241126\images"
    # file_move(src_data_folder)

    split_ratio = 0.95  # 训练集占比
    split_dataset(src_data_folder, split_ratio)