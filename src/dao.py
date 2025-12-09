import os
import json
from typing import Dict, List, Optional
from PIL import Image
from PIL import UnidentifiedImageError

from .common import AppConfig, init_logger, get_all_image_paths

logger = init_logger("DAO")


# -------------------------- 用户行为DAO（单一职责：用户行为数据操作） --------------------------
class UserBehaviorDAO:
    def __init__(self, config: AppConfig):
        self.config = config
        self.behavior: Dict[str, List[str]] = self._load_behavior()

    def _load_behavior(self) -> Dict[str, List[str]]:
        """加载用户行为（解耦文件操作与业务逻辑）"""
        default_behavior = {"search_history": [], "click_history": [], "analyze_history": []}

        absolute_file_path = os.path.abspath(self.config.user_behavior_file)
        logger.debug(f"用户行为文件路径: {absolute_file_path}")
        if not os.path.exists(absolute_file_path):
            # 确保父目录存在（即使是根目录，os.makedirs也不会报错）
            parent_dir = os.path.dirname(absolute_file_path)
            if parent_dir:  # 父目录不为空时才创建（避免空路径错误）
                os.makedirs(parent_dir, exist_ok=True)
            # 直接创建空文件（避免后续保存失败）
            with open(absolute_file_path, 'w', encoding='utf-8') as f:
                json.dump(default_behavior, f, ensure_ascii=False, indent=2)
            logger.info(f"已自动创建用户行为文件: {absolute_file_path}")
            return default_behavior

        try:
            with open(self.config.user_behavior_file, 'r', encoding='utf-8') as f:
                behavior = json.load(f)
            # 兼容字段缺失
            for key, val in default_behavior.items():
                if key not in behavior:
                    behavior[key] = val
            return behavior
        except Exception as e:
            logger.error(f"加载用户行为失败: {str(e)}")
            return default_behavior

    def _save_behavior(self, behavior: Dict[str, List[str]]) -> None:
        """保存用户行为（解耦文件操作）"""
        try:
            os.makedirs(os.path.dirname(self.config.user_behavior_file), exist_ok=True)
            with open(self.config.user_behavior_file, 'w', encoding='utf-8') as f:
                json.dump(behavior, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存用户行为失败: {str(e)}")

    def add_behavior(self, behavior_type: str, value: str) -> None:
        """添加用户行为（解耦业务规则与数据存储）"""
        if behavior_type not in self.behavior or not value.strip():
            return

        value = value.strip()
        if value not in self.behavior[behavior_type]:
            self.behavior[behavior_type].append(value)
            # 限制历史长度
            if len(self.behavior[behavior_type]) > self.config.max_history_len:
                self.behavior[behavior_type].pop(0)
            self._save_behavior(self.behavior)

    def get_behavior(self) -> Dict[str, List[str]]:
        """获取用户行为（只读，避免外部修改）"""
        return self.behavior.copy()


# -------------------------- 图片DAO（单一职责：图片加载操作） --------------------------
class ImageDAO:
    def __init__(self, config: AppConfig):
        self.config = config
        self.image_paths = get_all_image_paths(config.image_folder)
        logger.info(f"加载图片数量: {len(self.image_paths)}")

    def load_image(self, path: str) -> Optional[Image.Image]:
        """加载单张图片（解耦图片处理与业务逻辑）"""
        try:
            with Image.open(path) as img:
                img.thumbnail(self.config.image_size, Image.Resampling.LANCZOS)
                return img.convert("RGB").copy() if img.mode != "RGB" else img.copy()
        except UnidentifiedImageError:
            logger.error(f"图片格式错误: {path}")
        except Exception as e:
            logger.error(f"加载图片失败: {str(e)}")
        return self.get_placeholder()

    def get_placeholder(self) -> Image.Image:
        """生成占位图（解耦占位图逻辑）"""
        return Image.new('RGB', self.config.image_size, self.config.placeholder_color)

    def get_random_images(self, count: int) -> List[Image.Image]:
        """获取随机图片（解耦随机逻辑与业务）"""
        if len(self.image_paths) == 0:
            return [self.get_placeholder() for _ in range(count)]

        import random
        random_paths = random.sample(self.image_paths, min(count, len(self.image_paths)))
        return [self.load_image(path) or self.get_placeholder() for path in random_paths]

    def get_image_paths(self) -> List[str]:
        """获取所有图片路径（只读）"""
        return self.image_paths.copy()


# -------------------------- 索引DAO（单一职责：索引加载/构建） --------------------------
class IndexDAO:
    def __init__(self, config: AppConfig, clip_matcher):
        self.config = config
        self.clip_matcher = clip_matcher  # 依赖注入，解耦CLIP实现

    def load_or_build_indexes(self) -> None:
        """加载或构建索引（解耦索引操作与业务逻辑）"""
        self._load_or_build_image_index()
        self._load_or_build_text_index()

    def _load_or_build_image_index(self) -> None:
        if os.path.exists(self.config.image_index_path):
            try:
                self.clip_matcher.load_image_index(self.config.image_index_path)
                logger.info("加载图像索引成功")
                return
            except Exception as e:
                logger.error(f"加载图像索引失败: {str(e)}")

        logger.info("构建图像索引...")
        if os.path.exists(self.config.image_folder):
            self.clip_matcher.build_image_index(self.config.image_folder)
        else:
            logger.error("图像文件夹不存在，无法构建索引")

    def _load_or_build_text_index(self) -> None:
        if os.path.exists(self.config.text_index_path):
            try:
                self.clip_matcher.load_text_index(self.config.text_index_path)
                logger.info("加载文本索引成功")
                return
            except Exception as e:
                logger.error(f"加载文本索引失败: {str(e)}")

        logger.info("构建文本索引...")
        if os.path.exists(self.config.style_csv_path):
            self.clip_matcher.build_text_index(self.config.style_csv_path)
        else:
            logger.warning("文本CSV文件不存在，跳过文本索引构建")