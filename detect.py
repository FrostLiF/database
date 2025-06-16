from ultralytics import YOLO

# 加载你训练好的模型
model = YOLO("runs/train/anime_facial_parts/weights/best.pt")

# 推理一张图像，显示检测结果
model.predict(
    source="test_image.jpg",  # ← 替换为你想检测的动漫人物图
    conf=0.25,
    show=True,                # 自动弹出窗口显示图像
    save=True                 # 也可以加上保存预测图像
)
