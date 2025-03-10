import os
from pathlib import Path

def encrypt_files(xml_path, bin_path, hcov_path, key=0x55):
    """合并加密文件"""
    with open(xml_path, 'rb') as f_xml, open(bin_path, 'rb') as f_bin:
        xml_data = f_xml.read()
        bin_data = f_bin.read()
        
        # 文件头结构：[4]HCIR+ [4]xml长度 + [4]bin长度
        header = b'HCIR' + len(xml_data).to_bytes(4, 'little') + len(bin_data).to_bytes(4, 'little')
        
        # 异或加密
        encrypted = bytes([b ^ key for b in header + xml_data + bin_data])
        
        with open(hcov_path, 'wb') as f_out:
            f_out.write(encrypted)

def decrypt_files(hcov_path, out_xml, out_bin, key=0x55):
    """解密拆分文件"""
    with open(hcov_path, 'rb') as f:
        encrypted = f.read()
        
        # 异或解密
        decrypted = bytes([b ^ key for b in encrypted])
        
        # 解析文件头
        xml_len = int.from_bytes(decrypted[4:8], 'little')
        bin_len = int.from_bytes(decrypted[8:12], 'little')
        
        # 拆分文件
        with open(out_xml, 'wb') as f_xml:
            f_xml.write(decrypted[12:12+xml_len])
        with open(out_bin, 'wb') as f_bin:
            f_bin.write(decrypted[12+xml_len:12+xml_len+bin_len])
            
            
if __name__ == '__main__':
    
    xml_path_file = Path(r'E:\work\code\ultralytics\project\QR_yolo11m640\exp\weights\ov\t\detect_检测形状.xml')
    bin_path_file = xml_path_file.with_suffix('.bin')
    hcov_path_file = xml_path_file.with_suffix('.hcir')
    # encrypt_files(xml_path_file, bin_path_file, hcov_path_file)
    decrypt_files(hcov_path_file, xml_path_file, bin_path_file)