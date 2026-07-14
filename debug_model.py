import sys, timm
sys.path.append('.')
import models.safe_mb
import models.resnet

models_found = timm.list_models('*safe*') + timm.list_models('*SAFE*')
print("🔍 找到的注册模型:", models_found)

if not models_found:
    print("❌ 模型根本没有成功注册，可能是装饰器没写对！")
    sys.exit(1)

model_name = models_found[0]
print(f"\n⚙️ 正在尝试实例化 {model_name}...")

configs_to_try = [
    {'ablation_mode': 'ours', 'num_classes': 2, 'pretrained': False},
    {'ablation_mode': 'ours', 'nb_classes': 2, 'pretrained': False},
    {'num_classes': 2, 'pretrained': False},
    {'nb_classes': 2, 'pretrained': False}
]

for kwargs in configs_to_try:
    try:
        model = timm.create_model(model_name, **kwargs)
        print(f"✅ 成功！正确的参数组合是: {kwargs}")
        sys.exit(0)
    except Exception as e:
        print(f"❌ 失败 {kwargs} | 报错原因: {type(e).__name__} - {e}")
