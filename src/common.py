import os
import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional


# -------------------------- 配置类（补全 clip_model_path 字段） --------------------------
@dataclass
class AppConfig:
    """应用配置类（集中管理所有配置，支持外部注入）"""
    user_behavior_file: str = "user_behavior.json"
    image_folder: str = "test_Images"
    image_index_path: str = "image_index.pkl"
    text_index_path: str = "text_index.pkl"
    style_csv_path: str = "test_styles.csv"
    default_recommend_num: int = 12
    max_history_len: int = 20
    recent_behavior_cnt: int = 3
    top_k_recommend: int = 5
    image_size: Tuple[int, int] = (300, 400)
    placeholder_color: str = "#f0f0f0"
    clip_model_path: str = "D:\graduation\Clip\Clip\clip-vit-base-patch32"  # Windows绝对路
    clip_device: Optional[str] = None  # 新增：单独配置CLIP设备（可选）


# -------------------------- 日志工具（解耦日志配置） --------------------------
def init_logger(name: str = "FashionCLIPApp") -> logging.Logger:
    """初始化日志工具（独立配置，可复用）"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(name)


# -------------------------- 工具函数（解耦通用逻辑） --------------------------
def get_all_image_paths(folder: str) -> List[str]:
    """获取文件夹下所有图片路径（通用工具，解耦业务逻辑）"""
    if not os.path.exists(folder):
        return []

    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')
    image_paths = []

    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(image_extensions):
                image_paths.append(os.path.abspath(os.path.join(root, file)))

    return image_paths