import os
import cv2
import glob
import albumentations as A
from tqdm import tqdm

# 确保同时处理真图和假图，保证测试集平衡！
INPUT_REAL = "datasets/test3_Ojha/test/guided/0_real"
INPUT_FAKE = "datasets/test3_Ojha/test/guided/1_fake"

LEVELS = ["Level_1", "Level_2", "Level_3"]
OUTPUT_BASE = {
    "Level_1": "./test_data_level_1/guided", 
    "Level_2": "./test_data_level_2/guided", 
    "Level_3": "./test_data_level_3/guided"
}

# 修正后的梯度退化参数 (拒绝降维打击，保留微弱信号)
transforms = {
    "Level_1": A.Compose([
        A.ImageCompression(quality_lower=80, quality_upper=90, p=1.0), 
        A.GaussianBlur(blur_limit=(3, 3), p=1.0)
    ]),
    "Level_2": A.Compose([
        A.Resize(height=180, width=180, p=1.0), # 缩放至约 0.7x
        A.ImageCompression(quality_lower=60, quality_upper=70, p=1.0),
        A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        A.Resize(height=256, width=256, p=1.0)  # 缩放回原尺寸
    ]),
    "Level_3": A.Compose([
        A.Resize(height=128, width=128, p=1.0), # 缩放至 0.5x，不再是 0.25x
        A.ImageCompression(quality_lower=40, quality_upper=50, p=1.0),
        A.GaussianBlur(blur_limit=(5, 7), p=1.0),
        A.Resize(height=256, width=256, p=1.0)
    ])
}

for level in LEVELS:
    os.makedirs(os.path.join(OUTPUT_BASE[level], "0_real"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_BASE[level], "1_fake"), exist_ok=True)

def process_folder(input_dir, class_name):
    image_paths = glob.glob(os.path.join(input_dir, "*.png")) + glob.glob(os.path.join(input_dir, "*.jpg"))
    print(f"正在处理 {class_name} 的 {len(image_paths)} 张图片...")
    for img_path in tqdm(image_paths):
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is None: continue
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for level in LEVELS:
            degraded = transforms[level](image=image)['image']
            out_path = os.path.join(OUTPUT_BASE[level], class_name, img_name)
            cv2.imwrite(out_path, cv2.cvtColor(degraded, cv2.COLOR_RGB2BGR))

process_folder(INPUT_REAL, "0_real")
process_folder(INPUT_FAKE, "1_fake")
print("✅ 梯度退化数据生成完毕！真假图已完全平衡！")