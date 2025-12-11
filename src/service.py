import os
import tempfile
from typing import List, Tuple, Optional, Dict

import bcrypt
import numpy as np
from PIL import Image

from .auth_dao import UserAuthDAO
from .common import AppConfig, init_logger
from .dao import UserBehaviorDAO, ImageDAO

logger = init_logger("Service")

# -------------------------- 认证服务（单一职责：搜索业务逻辑）- 关键调整 --------------------------
class AuthService:
    def __init__(self, auth_dao: UserAuthDAO):
        self.auth_dao = auth_dao

    def register_user(self, username: str, password: str) -> bool:
        """注册新用户，返回是否成功"""
        if not username or len(password) < 6:
            return False  # 简单校验

        # 生成盐并哈希密码
        password_bytes = password.encode('utf-8')
        salt = bcrypt.gensalt()
        hashed_password = bcrypt.hashpw(password_bytes, salt)

        user_id = self.auth_dao.add_user(username, hashed_password)
        return user_id is not None

    def login_user(self, username: str, password: str) -> Optional[int]:
        """用户登录，成功返回用户ID，失败返回None"""
        user_data = self.auth_dao.get_user_data(username)
        if not user_data:
            return None  # 用户不存在

        user_id, password_hash = user_data

        # 验证密码
        password_bytes = password.encode('utf-8')
        if bcrypt.checkpw(password_bytes, password_hash):
            return user_id  # 登录成功，返回用户ID
        else:
            return None  # 密码错误


# -------------------------- 搜索服务（单一职责：搜索业务逻辑）- 关键调整 --------------------------
class SearchService:
    def __init__(self, config: AppConfig, clip_matcher, image_dao: ImageDAO, behavior_dao: UserBehaviorDAO):
        self.config = config
        self.clip_matcher = clip_matcher
        self.image_dao = image_dao
        self.behavior_dao = behavior_dao

    def text_search(self, query: str, top_k: int, user_id: Optional[int] = None) -> List[Tuple[Image.Image, str]]:
        """文本搜索（保持不变）"""
        if not query.strip() or top_k < 1:
            return []

        self.behavior_dao.add_behavior(user_id, "search_history", query.strip())

        try:
            results = self.clip_matcher.search_images_by_text(query.strip(), top_k=top_k)

            output = []
            for path, _ in results:
                img = self.image_dao.load_image(path) or self.image_dao.get_placeholder()
                caption = self.image_dao.get_caption_by_path(path)
                output.append((img, caption))
            return output
        except Exception as e:
            logger.error(f"文本搜索失败: {str(e)}")
            return []

    def image_search(self, query_image: Image.Image, top_k: int, user_id: Optional[int] = None) -> List[
        Tuple[Image.Image, str]]:
        """图像搜索（优化：执行单次搜索并记录行为）"""
        if not query_image or top_k < 1:
            return []

        tmp_path = None
        try:
            # 1. 保存临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                query_image.convert("RGB").save(tmp_file, format='JPEG', quality=95)
                tmp_path = tmp_file.name

            # 2. 执行检索 (只执行一次检索)
            # 假设 self.clip_matcher.search_images_by_image 返回：List[Tuple[path, similarity_score]]
            results = self.clip_matcher.search_images_by_image(
                query_image_path=tmp_path,
                top_k=top_k
            )

            output = []
            best_caption_to_record = None

            for i, (path, _) in enumerate(results):
                img = self.image_dao.load_image(path) or self.image_dao.get_placeholder()
                caption = self.image_dao.get_caption_by_path(path)

                # 3. 🚨 行为记录：使用排名第一的商品的描述
                if user_id is not None and i == 0 and caption:
                    best_caption_to_record = caption

                output.append((img, caption))

            # 4. 记录行为（放在循环外执行，确保只记录一次）
            if user_id is not None and best_caption_to_record:
                search_description = "[图搜]" + best_caption_to_record
                self.behavior_dao.add_behavior(user_id, "search_history", search_description)
                logger.info(f"记录图像搜索行为 (User {user_id}): {search_description[:40]}...")

            return output

        except Exception as e:
            logger.error(f"图像搜索失败: {str(e)}")
            return []

        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)  # 确保删除临时文件


# -------------------------- 推荐服务（单一职责：推荐业务逻辑）- 关键调整 --------------------------
class RecommendService:
    def __init__(self, config: AppConfig, clip_matcher, image_dao: ImageDAO, behavior_dao: UserBehaviorDAO):
        self.config = config
        self.clip_matcher = clip_matcher
        self.image_dao = image_dao
        self.behavior_dao = behavior_dao
        self._last_recommendation_cache: Dict[int, Tuple[List[Tuple[Image.Image, str]], str]] = {}

    def _build_user_interest_vector(self, behavior: dict) -> Optional[np.ndarray]:
        # ... (_build_user_interest_vector 方法体保持不变) ...
        vectors = []

        # 1. 历史点击商品向量 (图像特征)
        recent_clicks = behavior["click_history"][-self.config.recent_behavior_cnt:]
        if recent_clicks:
            try:
                img_features = self.clip_matcher.encode_images(recent_clicks)
                if img_features.size > 0:
                    vectors.append(img_features)
            except Exception as e:
                logger.error(f"图像向量编码失败: {e}")

        # 2. 历史搜索关键词向量 (文本特征)
        recent_searches = behavior["search_history"][-self.config.recent_behavior_cnt:]
        if recent_searches:
            try:
                text_features = self.clip_matcher.encode_texts(recent_searches)
                if text_features.size > 0:
                    vectors.append(text_features)
            except Exception as e:
                logger.error(f"文本向量编码失败: {e}")

        if not vectors:
            return None

        # 3. 平均化所有向量
        all_vectors = np.vstack(vectors)
        user_vector = np.mean(all_vectors, axis=0)

        # 4. 再次归一化并转换为 FAISS 要求的格式
        user_vector = user_vector / np.linalg.norm(user_vector)
        return user_vector.astype('float32').reshape(1, -1)

    def _get_random_recommendation(self) -> List[Tuple[Image.Image, str]]:
        """辅助函数: 获取随机推荐并添加 Caption"""
        random_images = self.image_dao.get_random_images(self.config.default_recommend_num)

        enriched_list = []
        for img in random_images:
            enriched_list.append((img, "随机精选商品"))

        return enriched_list

    def get_personalized_recommend(self, user_id: Optional[int]) -> Tuple[List[Tuple[Image.Image, str]], str]:
        """个性化推荐（使用用户兴趣向量进行单步高性能检索）"""

        if user_id is None:
            # 对于未登录用户，不缓存，直接返回结果
            return self._get_random_recommendation(), "📱 请先登录以获取个性化推荐。"

        behavior = self.behavior_dao.get_behavior(user_id)
        has_behavior = any([len(behavior["search_history"]) > 0, len(behavior["click_history"]) > 0])

        if not has_behavior:
            # 同样，对于无历史用户，不缓存
            return self._get_random_recommendation(), "✨ 您的账户暂无历史记录，为您推荐热门商品。"

            # 1. 构建用户兴趣向量 (User Interest Vector)
        user_vector = self._build_user_interest_vector(behavior)

        if user_vector is None:
            return self._get_random_recommendation(), "⚠️ 无法构建用户画像，已转为热门商品推荐。"

            # 2. 单步高性能检索
        results = self.clip_matcher.search_images_by_vector(user_vector, top_k=self.config.default_recommend_num)

        # 3. 数据封装
        enriched_recommendations = []
        for path, _ in results:
            img = self.image_dao.load_image(path) or self.image_dao.get_placeholder()
            caption = self.image_dao.get_caption_by_path(path)
            enriched_recommendations.append((img, caption))

        # 4. 补充不足数量
        while len(enriched_recommendations) < self.config.default_recommend_num:
            placeholder_img = self.image_dao.get_placeholder()
            placeholder_caption = "占位商品"
            enriched_recommendations.append((placeholder_img, placeholder_caption))

        reason = self._generate_reason(behavior)

        # 🚨 NEW: 缓存结果
        self._last_recommendation_cache[user_id] = (
        enriched_recommendations[:self.config.default_recommend_num], reason)

        return self._last_recommendation_cache[user_id]

    def _generate_recommendation_paths(self, user_id: int) -> List[str]:
        # ... (_generate_recommendation_paths 方法体保持不变) ...
        """
        生成当前用户兴趣向量搜索结果的路径列表（用于行为跟踪）。
        """
        behavior = self.behavior_dao.get_behavior(user_id)

        user_vector = self._build_user_interest_vector(behavior)
        if user_vector is None:
            # 无法构建向量，则退化到获取所有路径（作为随机候选集）
            return self.image_dao.get_image_paths()[:self.config.default_recommend_num]

        # 使用用户向量进行搜索
        results = self.clip_matcher.search_images_by_vector(user_vector, top_k=self.config.default_recommend_num)

        return [path for path, _ in results]

    def _generate_reason(self, behavior: dict) -> str:
        """生成推荐理由（翻译中文）"""
        reasons = []
        if len(behavior["search_history"]) > 0:
            reasons.append("搜索记录")
        if len(behavior["click_history"]) > 0:
            reasons.append("点击偏好")
        return f"🎯 个性化推荐（基于您的{('和'.join(reasons))}）"

# -------------------------- 行为跟踪服务（单一职责：行为跟踪逻辑）- 关键调整 --------------------------
class BehaviorTrackService:
    def __init__(self, config: AppConfig, behavior_dao: UserBehaviorDAO, recommend_service: RecommendService):
        self.config = config
        self.behavior_dao = behavior_dao
        self.recommend_service = recommend_service
        self.caption_max_display_length = 50  # 截断长度常量

    def track_recommend_click(self, user_id: Optional[int], click_index: int) -> Tuple[
        List[Tuple[Image.Image, str]], str]:
        # ... (方法体保持不变) ...
        if user_id is None:
            return self.recommend_service.get_personalized_recommend(None)
        if click_index < 0:
            return self.recommend_service.get_personalized_recommend(user_id)
        # 获取当前推荐的候选路径 (通过重用 _generate_recommendation_paths)
        candidate_paths = self.recommend_service._generate_recommendation_paths(user_id)

        if 0 <= click_index < len(candidate_paths):
            self.behavior_dao.add_behavior(user_id, "click_history", candidate_paths[click_index])
            logger.info(f"用户 {user_id} 跟踪点击: {candidate_paths[click_index]}")

        # 刷新推荐
        return self.recommend_service.get_personalized_recommend(user_id)

    def get_user_activity_history(self, user_id: Optional[int]) -> List[str]:
        """
        🚨 NEW FUNCTION: 获取并格式化用户活动时间线列表 (字符串形式，用于 UI 可视化)。
        """
        if user_id is None:
            return ["请先登录以查看您的活动记录。"]

        # 调用 DAO 方法
        raw_history = self.behavior_dao.get_full_activity_history(user_id)

        if not raw_history:
            return ["您目前没有活动记录。请尝试搜索或点击推荐商品。"]

        formatted_list = []

        for item in raw_history:
            # 格式化时间戳 (去除毫秒)
            timestamp_str = item['timestamp'].split('.')[0]
            value = item['value']

            if item['type'] == 'search':
                # 搜索记录
                formatted_list.append(f"[{timestamp_str}] 🔎 **搜索**: “{value}”")
            elif item['type'] == 'click':
                # 点击记录，需要查找 Caption
                caption = self.recommend_service.image_dao.get_caption_by_path(value)

                # 截断 Caption
                display_caption = caption
                if len(caption) > self.caption_max_display_length:
                    display_caption = caption[:self.caption_max_display_length] + '...'

                formatted_list.append(f"[{timestamp_str}] ✨ **点击**: “{display_caption}”")

        return formatted_list

    def delete_user_history(self, user_id: Optional[int]) -> bool:
        """调用 DAO 删除用户的全部行为记录"""
        if user_id is None:
            logger.warning("尝试删除历史记录失败：用户未登录。")
            return False
        return self.behavior_dao.delete_all_behavior(user_id)