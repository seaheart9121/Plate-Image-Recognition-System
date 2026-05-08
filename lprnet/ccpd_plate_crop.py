import os
import cv2
import numpy as np
from tqdm import tqdm

PROVINCES = [
    "皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑",
    "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
    "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁",
    "新", "警", "学", "O"
]
ADS = [
    'A','B','C','D','E','F','G','H','J','K','L','M',
    'N','P','Q','R','S','T','U','V','W','X','Y','Z',
    '0','1','2','3','4','5','6','7','8','9','O'
]

def four_point_transform(image, pts):
    width, height = 94, 24
    dst = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(image, M, (width, height))

def safe_split(s):
    """兼容所有可能的分隔符：& , _"""
    return s.replace(',', '&').replace('_', '&').split('&')

def convert_ccpd_to_lpr(ccpd_root_list, save_root):
    os.makedirs(os.path.join(save_root, 'train'), exist_ok=True)
    os.makedirs(os.path.join(save_root, 'val'), exist_ok=True)

    image_files = []
    for root_path in ccpd_root_list:
        if not os.path.exists(root_path):
            print(f"⚠️ 路径不存在，跳过：{root_path}")
            continue
        for root, _, files in os.walk(root_path):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_files.append(os.path.join(root, f))

    split = int(len(image_files) * 0.9)
    train_files = image_files[:split]
    val_files = image_files[split:]

    def process(files, subset):
        for file_path in tqdm(files, desc=f"Processing {subset}"):
            try:
                filename = os.path.basename(file_path)
                parts = filename.split('-')
                if len(parts) < 5:
                    print(f"⛔ 文件名格式异常: {filename}")
                    continue

                # 解析车牌号
                plate_indices = parts[4].split('_')
                if len(plate_indices) < 2:
                    continue
                plate_str = PROVINCES[int(plate_indices[0])] + ADS[int(plate_indices[1])]
                for idx_str in plate_indices[2:]:
                    plate_str += ADS[int(idx_str)]
                plate_str = plate_str[:8]  # 适配蓝牌7位和绿牌8位

                # 解析坐标（使用升级版 safe_split）
                vertices_str = parts[3]
                v_coords = vertices_str.split('_')
                pts = []
                for v in v_coords:
                    coord = safe_split(v)
                    if len(coord) >= 2:
                        pts.append([int(coord[0]), int(coord[1])])
                if len(pts) < 4:
                    print(f"⛔ 坐标解析失败: {filename}")
                    continue
                pts = np.array([pts[2], pts[3], pts[0], pts[1]], dtype="float32")

                img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                crop = four_point_transform(img, pts)

                save_name = f"{plate_str}_{os.path.splitext(filename)[0]}.jpg"
                save_path = os.path.join(save_root, subset, save_name)
                cv2.imencode('.jpg', crop)[1].tofile(save_path)

            except Exception as e:
                print(f"⛔ 处理失败: {filename} → {str(e)}")
                continue

    process(train_files, 'train')
    process(val_files, 'val')
    print(f"✅ 完成！蓝牌和绿牌均已转换至：{save_root}")

if __name__ == "__main__":
    # 蓝牌路径 和 绿牌路径 分别配置
    convert_ccpd_to_lpr(
        ccpd_root_list=["../qwertyu", "../zxcvbnm"],  # 改成您实际的蓝牌和绿牌目录
        save_root="ccpd_plate_crop_dataset5"
    )