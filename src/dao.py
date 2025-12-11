import os
import json
import sqlite3
from typing import Dict, List, Optional

import pandas as pd
from PIL import Image
from PIL import UnidentifiedImageError
from .db_utils import get_db_connection
from .common import AppConfig, init_logger, get_all_image_paths

logger = init_logger("DAO")


# -------------------------- 用户行为DAO（单一职责：用户行为数据操作） --------------------------
class UserBehaviorDAO:
    def __init__(self, config: AppConfig):
        self.config = config

    def add_behavior(self, user_id: int, behavior_type: str, value: str) -> None:
        """添加用户行为（强制传入 user_id）"""
        if user_id is None:
            return

        conn = get_db_connection()
        cursor = conn.cursor()

        try:
            if behavior_type == "click_history":
                # 记录点击行为
                cursor.execute(
                    "INSERT INTO user_clicks (user_id, image_path) VALUES (?, ?)",
                    (user_id, value.strip())
                )
                logger.debug(f"用户 {user_id} 记录点击: {value}")

            elif behavior_type == "search_history":
                # 记录搜索行为
                cursor.execute(
                    "INSERT INTO user_searches (user_id, query_text) VALUES (?, ?)",
                    (user_id, value.strip())
                )
                logger.debug(f"用户 {user_id} 记录搜索: {value}")

            conn.commit()
        except sqlite3.OperationalError as e:
            logger.error(f"无法记录行为: 数据库操作错误，请检查表结构。错误信息: {e}")
        except Exception as e:
            logger.error(f"记录行为失败: {e}")
        finally:
            conn.close()

    def get_behavior(self, user_id: Optional[int]) -> Dict[str, List[str]]:
        """获取用于推荐系统的最新行为（保持不变）"""
        if user_id is None:
            return {"search_history": [], "click_history": []}

        conn = get_db_connection()
        cursor = conn.cursor()

        try:
            # 1. 获取点击历史 (最新 N 条)
            cursor.execute(
                "SELECT image_path FROM user_clicks WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?",
                (user_id, self.config.max_history_len)
            )
            click_history = [row[0] for row in cursor.fetchall()]

            # 2. 获取搜索历史 (最新 N 条)
            cursor.execute(
                "SELECT query_text FROM user_searches WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?",
                (user_id, self.config.max_history_len)
            )
            search_history = [row[0] for row in cursor.fetchall()]

        except sqlite3.OperationalError as e:
            logger.error(f"读取行为失败: 数据库操作错误，可能缺少表。错误信息: {e}")
            search_history = []
            click_history = []
        except Exception as e:
            logger.error(f"读取行为失败: {e}")
            search_history = []
            click_history = []
        finally:
            conn.close()

        return {"search_history": search_history, "click_history": click_history}

    def get_full_activity_history(self, user_id: int) -> List[Dict]:
        """
        🚨 NEW FUNCTION: 检索所有用户活动记录（搜索和点击），按时间排序，用于可视化。
        """
        if user_id is None:
            return []

        conn = get_db_connection()
        cursor = conn.cursor()

        history = []

        try:
            # 1. 获取点击历史 (image_path, timestamp)
            cursor.execute(
                "SELECT image_path, timestamp FROM user_clicks WHERE user_id = ? ORDER BY timestamp DESC",
                (user_id,)
            )
            for path, timestamp in cursor.fetchall():
                history.append({
                    "type": "click",
                    "value": path,
                    "timestamp": timestamp
                })

            # 2. 获取搜索历史 (query_text, timestamp)
            cursor.execute(
                "SELECT query_text, timestamp FROM user_searches WHERE user_id = ? ORDER BY timestamp DESC",
                (user_id,)
            )
            for query, timestamp in cursor.fetchall():
                history.append({
                    "type": "search",
                    "value": query,
                    "timestamp": timestamp
                })

        except sqlite3.OperationalError as e:
            logger.error(f"无法读取完整的活动记录: {e}")
        finally:
            conn.close()

        # 按照时间戳排序 (降序)
        history.sort(key=lambda x: x['timestamp'], reverse=True)

        return history

    def delete_all_behavior(self, user_id: int) -> bool:
        """🚨 NEW FUNCTION: 删除某个用户的所有搜索和点击行为记录"""
        if user_id is None:
            return False

        conn = get_db_connection()
        cursor = conn.cursor()
        success = False

        try:
            # 1. 删除点击历史
            cursor.execute(
                "DELETE FROM user_clicks WHERE user_id = ?",
                (user_id,)
            )
            # 2. 删除搜索历史
            cursor.execute(
                "DELETE FROM user_searches WHERE user_id = ?",
                (user_id,)
            )
            conn.commit()
            success = True
            logger.info(f"成功删除用户 {user_id} 的所有行为记录。")
        except Exception as e:
            logger.error(f"删除用户行为记录失败 (ID: {user_id}): {e}")
            conn.rollback()
        finally:
            conn.close()

        return success

# -------------------------- 图片DAO（单一职责：图片加载操作） --------------------------
class ImageDAO:
    def __init__(self, config: AppConfig):
        self.config = config
        self.image_paths = get_all_image_paths(config.image_folder)
        logger.info(f"加载图片数量: {len(self.image_paths)}")
        self.caption_map: Dict[str, str] = self._load_captions()

    def _load_captions(self) -> Dict[str, str]:
        """从 CSV 文件加载图片名到 Caption 的映射"""
        caption_map = {}
        if not os.path.exists(self.config.style_csv_path):
            logger.warning(f"Caption file not found: {self.config.style_csv_path}")
            return caption_map

        try:
            df = pd.read_csv(self.config.style_csv_path)
            # 假设 CSV 有两列: 'image' (文件名) 和 'caption' (描述)
            if 'image' in df.columns and 'caption' in df.columns:
                # 遍历 DataFrame
                for index, row in df.iterrows():
                    image_filename = row['image']
                    # 构建完整的绝对路径，确保匹配 ImageDAO.image_paths 中的格式
                    full_path = os.path.abspath(os.path.join(self.config.image_folder, image_filename))
                    caption_map[full_path] = row['caption']
                logger.info(f"Loaded {len(caption_map)} Caption mappings.")
        except Exception as e:
            logger.error(f"Failed to load captions from CSV: {str(e)}")

        return caption_map

        # 🚨 NEW: 提供根据图片路径查询 Caption 的接口
    def get_caption_by_path(self, path: str) -> str:
            """根据图片绝对路径获取其 Caption，找不到则返回默认值"""
            # 确保路径是绝对路径以便匹配 map 中的 key
            abs_path = os.path.abspath(path)
            # 🚨 默认英文描述
            return self.caption_map.get(abs_path, "No description available")

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