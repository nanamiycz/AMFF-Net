import os
from PIL import Image
from pathlib import Path
from tqdm import tqdm

def create_resized_dataset(src_dir, dest_dir, scale):
    src_path = Path(src_dir)
    dest_path = Path(dest_dir)
    
    # 找到所有的 png 和 jpg
    images = list(src_path.rglob('*.png')) + list(src_path.rglob('*.jpg'))
    
    for img_path in tqdm(images, desc=f"Generating {scale}x dataset"):
        # 保持原有的目录结构 (如 test/dalle, test/guided 等)
        rel_path = img_path.relative_to(src_path)
        out_path = dest_path / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                new_w, new_h = int(w * scale), int(h * scale)
                
                # 核心降质：先缩小（物理丢失高频），再放大回原尺寸（双三次插值抹平）
                img_resized = img.resize((new_w, new_h), Image.BICUBIC)
                img_restored = img_resized.resize((w, h), Image.BICUBIC)
                
                # 保存为纯净的破坏后图像
                img_restored.save(out_path)
        except Exception as e:
            pass

if __name__ == "__main__":
    print("开始生成 0.5x 缩放破坏数据集...")
    create_resized_dataset("datasets/test3_Ojha", "datasets/test3_Ojha_scale05", 0.5)
    
    print("\n开始生成 0.25x 极限缩放破坏数据集...")
    create_resized_dataset("datasets/test3_Ojha", "datasets/test3_Ojha_scale025", 0.25)
    
    print("\n✅ 全部生成完毕！可以开始测试了！")