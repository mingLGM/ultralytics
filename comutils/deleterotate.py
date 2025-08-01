
import os

def compare_and_clean(dir1, dir2):
    # 获取两个目录中的文件名（不含后缀）
    files1 = {os.path.splitext(f)[0] for f in os.listdir(dir1) if os.path.isfile(os.path.join(dir1, f))}
    files2 = {os.path.splitext(f)[0] for f in os.listdir(dir2) if os.path.isfile(os.path.join(dir2, f))}
    
    # 找出dir2中多余的文件（在dir2但不在dir1中的文件）
    extra_files = files2 - files1
    
    # 删除dir2中多余的文件
    for file in os.listdir(dir2):
        name, ext = os.path.splitext(file)
        if name in extra_files:
            file_path = os.path.join(dir2, file)
            os.remove(file_path)
            print(f"已删除: {file_path}")

# 使用示例
dir_a = "D:/work/lzm/QRDM_dataset/dataset/rotate/labels"  # 替换为第一个目录路径
dir_b = r"D:\work\lzm\QRDM_dataset\dataset\rotate\jsons"  # 替换为第二个目录路径
compare_and_clean(dir_a, dir_b)
