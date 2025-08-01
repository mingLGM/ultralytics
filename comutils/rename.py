import os
import shutil

def rename_paired_files(source_dir, target_dir, start_num=3096):
    """
    配对重命名图片和JSON文件到新文件夹，并更新JSON中的imagePath
    :param source_dir: 源文件夹路径
    :param target_dir: 目标文件夹路径
    :param start_num: 起始编号(默认2000)
    """
    # 创建目标文件夹
    os.makedirs(target_dir, exist_ok=True)
    
    # 支持的图片格式
    image_exts = ('.bmp', '.jpg', '.png', '.jpeg')
    all_files = os.listdir(source_dir)
    
    # 统计文件类型
    image_files = [f for f in all_files if os.path.splitext(f)[1].lower() in image_exts]
    json_files = [f for f in all_files if f.lower().endswith('.json')]
    
    print(f"发现 {len(image_files)} 个图片文件和 {len(json_files)} 个JSON文件")

    # 建立文件名基础名配对关系（不含扩展名）
    paired_files = {}
    unpaired_images = set()
    
    # 首先记录所有图片文件
    for img_file in image_files:
        basename = os.path.splitext(img_file)[0]
        unpaired_images.add(basename)
    
    # 建立配对关系
    for file in image_files + json_files:
        basename = os.path.splitext(file)[0]
        ext = os.path.splitext(file)[1].lower()
        
        if basename not in paired_files:
            paired_files[basename] = {'image': None, 'json': None}
        
        if ext in image_exts:
            paired_files[basename]['image'] = file
        elif ext == '.json':
            paired_files[basename]['json'] = file
            # 如果找到配对的JSON，从未配对集合中移除
            if basename in unpaired_images:
                unpaired_images.remove(basename)

    # 开始处理
    counter = start_num
    success_pairs = 0
    
    print("\n开始配对重命名...")
    for basename, files in paired_files.items():
        # 必须同时存在图片和JSON才处理
        if not files['image'] or not files['json']:
            continue
            
        new_name = f"QR-DM_{counter}"
        img_ext = os.path.splitext(files['image'])[1]
        
        # 处理图片文件
        shutil.copy2(
            os.path.join(source_dir, files['image']),
            os.path.join(target_dir, f"{new_name}{img_ext}")
        )
        
        # 处理JSON文件
        json_src = os.path.join(source_dir, files['json'])
        json_dest = os.path.join(target_dir, f"{new_name}.json")
        shutil.copy2(json_src, json_dest)
        
        # 更新JSON文件中的imagePath
        with open(json_dest, 'r+', encoding='utf-8') as f:
            try:
                data = json.load(f)
                data['imagePath'] = f"{new_name}{img_ext}"
                f.seek(0)
                json.dump(data, f, indent=4)
                f.truncate()
            except Exception as e:
                print(f"警告: 无法更新 {json_dest} 的imagePath - {str(e)}")
        
        print(f"配对重命名: {basename} -> {new_name}")
        counter += 1
        success_pairs += 1

    # 统计未被重命名的图片
    unpaired_image_files = []
    for basename in unpaired_images:
        for ext in image_exts:
            if os.path.exists(os.path.join(source_dir, f"{basename}{ext}")):
                unpaired_image_files.append(f"{basename}{ext}")
                break

    # 统计结果
    print("\n===== 处理结果 =====")
    print(f"成功配对组数: {success_pairs}")
    print(f"生成文件总数: {success_pairs * 2} (每组1图1JSON)")
    print(f"新文件命名范围: QR-DM_{start_num} 到 QR-DM_{counter-1}")
    print(f"目标文件夹: {target_dir}")
    
    if unpaired_image_files:
        print("\n===== 未重命名的图片 =====")
        for img in unpaired_image_files:
            print(f"- {img}")
    else:
        print("\n所有图片文件都已成功配对并重命名")

if __name__ == "__main__":
    # 配置路径（请修改为实际路径）
    source_folder = r"D:\work\lzm\QRDM码-数据集\dataset\images_BAR\images"  # 源文件夹路径
    target_folder = r"D:\work\lzm\QRDM码-数据集\dataset\image_temp"  # 新文件夹路径
    
    # 检查源文件夹
    if not os.path.exists(source_folder):
        print(f"错误: 源文件夹不存在 - {source_folder}")
        exit()
    
    # 执行重命名
    rename_paired_files(source_folder, target_folder)