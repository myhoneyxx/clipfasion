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


# -------------------------- 认证服务 --------------------------
class AuthService:
    def __init__(self, auth_dao: UserAuthDAO):
        self.auth_dao = auth_dao

    def register_user(self, username: str, password: str) -> bool:
        """注册新用户，返回是否成功"""
        if not username or len(password) < 6:
            return False

        password_bytes = password.encode('utf-8')
        salt = bcrypt.gensalt()
        hashed_password = bcrypt.hashpw(password_bytes, salt)

        user_id = self.auth_dao.add_user(username, hashed_password)
        return user_id is not None

    def login_user(self, username: str, password: str) -> Optional[int]:
        """用户登录，成功返回用户ID，失败返回None"""
        user_data = self.auth_dao.get_user_data(username)
        if not user_data:
            return None

        user_id, password_hash = user_data
        password_bytes = password.encode('utf-8')
        if bcrypt.checkpw(password_bytes, password_hash):
            return user_id
        else:
            return None


# -------------------------- 搜索服务 --------------------------
class SearchService:
    def __init__(self, config: AppConfig, clip_matcher, image_dao: ImageDAO, behavior_dao: UserBehaviorDAO):
        self.config = config
        self.clip_matcher = clip_matcher
        self.image_dao = image_dao
        self.behavior_dao = behavior_dao
        # 🚨 NEW: 搜索结果缓存 {user_id: [path1, path2, ...]} 用于点击跟踪
        self._last_search_cache: Dict[int, List[str]] = {}

    def text_search(self, query: str, top_k: int, user_id: Optional[int] = None) -> List[Tuple[Image.Image, str]]:
        if not query.strip() or top_k < 1:
            return []

        self.behavior_dao.add_behavior(user_id, "search_history", query.strip())

        try:
            results = self.clip_matcher.search_images_by_text(query.strip(), top_k=top_k)

            # 🚨 NEW: 缓存本次搜索结果的路径列表
            if user_id is not None:
                current_paths = [path for path, _ in results]
                self._last_search_cache[user_id] = current_paths

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
        if not query_image or top_k < 1:
            return []

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                query_image.convert("RGB").save(tmp_file, format='JPEG', quality=95)
                tmp_path = tmp_file.name

            results = self.clip_matcher.search_images_by_image(
                query_image_path=tmp_path,
                top_k=top_k
            )

            # 🚨 NEW: 缓存本次搜索结果的路径列表
            if user_id is not None:
                current_paths = [path for path, _ in results]
                self._last_search_cache[user_id] = current_paths

            output = []
            best_caption_to_record = None

            for i, (path, _) in enumerate(results):
                img = self.image_dao.load_image(path) or self.image_dao.get_placeholder()
                caption = self.image_dao.get_caption_by_path(path)

                if user_id is not None and i == 0 and caption:
                    best_caption_to_record = caption

                output.append((img, caption))

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
                os.unlink(tmp_path)

    # 🚨 NEW: 提供获取缓存路径的接口
    def get_cached_path(self, user_id: int, index: int) -> Optional[str]:
        if user_id in self._last_search_cache:
            paths = self._last_search_cache[user_id]
            if 0 <= index < len(paths):
                return paths[index]
        return None


# -------------------------- 推荐服务 --------------------------
class RecommendService:
    def __init__(self, config: AppConfig, clip_matcher, image_dao: ImageDAO, behavior_dao: UserBehaviorDAO):
        self.config = config
        self.clip_matcher = clip_matcher
        self.image_dao = image_dao
        self.behavior_dao = behavior_dao
        self._last_recommendation_cache: Dict[int, Tuple[List[Tuple[Image.Image, str]], str]] = {}

    # 🚨 核心修正：接收 user_id，使用混合历史数据构建向量
    def _build_user_interest_vector(self, user_id: int) -> Optional[np.ndarray]:
        """
        [修正版] 构建用户向量：使用统一时间窗口，最近的行为（无论搜索还是点击）权重最高
        """
        # 1. 获取混合历史 (例如最近 3 条)
        limit = self.config.recent_behavior_cnt
        # 调用 DAO 中新增的混合历史接口
        recent_items = self.behavior_dao.get_recent_combined_behavior(user_id, limit)

        if not recent_items:
            return None

        vectors = []

        # 2. 分别编码
        clicks = [item['value'] for item in recent_items if item['type'] == 'click']
        searches = [item['value'] for item in recent_items if item['type'] == 'search']

        if clicks:
            try:
                img_features = self.clip_matcher.encode_images(clicks)
                if img_features.size > 0:
                    vectors.append(img_features)
            except Exception as e:
                logger.error(f"图像向量编码失败: {e}")

        if searches:
            try:
                # 过滤掉 "[图搜]" 前缀
                clean_searches = [s.replace("[图搜]", "").strip() for s in searches]
                text_features = self.clip_matcher.encode_texts(clean_searches)
                if text_features.size > 0:
                    vectors.append(text_features)
            except Exception as e:
                logger.error(f"文本向量编码失败: {e}")

        if not vectors:
            return None

        # 3. 聚合
        all_vectors = np.vstack(vectors)
        user_vector = np.mean(all_vectors, axis=0)
        # 4. 归一化
        user_vector = user_vector / np.linalg.norm(user_vector)

        return user_vector.astype('float32').reshape(1, -1)

    def _get_random_recommendation(self) -> List[Tuple[Image.Image, str]]:
        """辅助函数: 获取随机推荐"""
        random_images = self.image_dao.get_random_images(self.config.default_recommend_num)
        enriched_list = []
        for img in random_images:
            enriched_list.append((img, "随机精选商品"))
        return enriched_list

    def _perform_partitioned_search(self, user_vector: np.ndarray) -> List[Tuple[str, float]]:
        """
        [修正版] 执行策略一：分片索引混合检索
        动态分配召回数量，确保总数能够填满 UI 列表
        """
        candidates = []

        # 获取目标展示数量 (例如 12)
        target_num = self.config.default_recommend_num

        # 💡 策略配置：动态分配召回配额
        # 总召回数设为目标的 ~1.3 倍，保证有足够数量供排序，同时容错
        # 1. 服饰 (Apparel): 核心品类，占 50%
        k_apparel = int(target_num * 0.5) + 2  # (12*0.5)+2 = 8

        # 2. 鞋履 (Footwear): 搭配品类，占 30%
        k_footwear = int(target_num * 0.3) + 1  # (12*0.3)+1 = 4

        # 3. 其他 (Others): 稀疏品类，占 20% (但至少保底 3 个)
        k_others = max(3, int(target_num * 0.2) + 1)  # max(3, 3) = 3

        # A. 核心品类 [Apparel]
        res_apparel = self.clip_matcher.search_in_partition(user_vector, "apparel", top_k=k_apparel)
        candidates.extend(res_apparel)

        # B. 次要品类 [Footwear]
        res_footwear = self.clip_matcher.search_in_partition(user_vector, "footwear", top_k=k_footwear)
        candidates.extend(res_footwear)

        # C. 稀疏品类 [Others]
        res_others = self.clip_matcher.search_in_partition(user_vector, "others", top_k=k_others)
        candidates.extend(res_others)

        # D. 结果排序
        # 将所有来源的商品混合，按相似度(score)降序排列
        candidates.sort(key=lambda x: x[1], reverse=True)

        return candidates

    def get_personalized_recommend(self, user_id: Optional[int]) -> Tuple[List[Tuple[Image.Image, str]], str]:
        """个性化推荐入口"""

        if user_id is None:
            return self._get_random_recommendation(), "📱 请先登录以获取个性化推荐。"

        # 保留旧的 get_behavior 仅用于判断“是否为空白用户”和生成“推荐理由”
        behavior = self.behavior_dao.get_behavior(user_id)
        has_behavior = any([len(behavior["search_history"]) > 0, len(behavior["click_history"]) > 0])

        if not has_behavior:
            return self._get_random_recommendation(), "✨ 您的账户暂无历史记录，为您推荐热门商品。"

        # 1. 构建用户兴趣向量 (🚨 修正：传入 user_id)
        user_vector = self._build_user_interest_vector(user_id)

        if user_vector is None:
            return self._get_random_recommendation(), "⚠️ 无法构建用户画像，已转为热门商品推荐。"

        # 2. 分片混合检索
        candidates = self._perform_partitioned_search(user_vector)

        # 3. 数据封装
        enriched_recommendations = []
        final_results = candidates[:self.config.default_recommend_num]

        for path, _ in final_results:
            img = self.image_dao.load_image(path) or self.image_dao.get_placeholder()
            caption = self.image_dao.get_caption_by_path(path)
            enriched_recommendations.append((img, caption))

        # 4. 补充不足数量
        while len(enriched_recommendations) < self.config.default_recommend_num:
            placeholder_img = self.image_dao.get_placeholder()
            enriched_recommendations.append((placeholder_img, "更多精选"))

        reason = self._generate_reason(behavior)
        self._last_recommendation_cache[user_id] = (enriched_recommendations, reason)

        return enriched_recommendations, reason

    def _generate_recommendation_paths(self, user_id: int) -> List[str]:
        """
        生成当前用户兴趣向量搜索结果的路径列表（用于行为跟踪）。
        """
        # 🚨 修正：传入 user_id
        user_vector = self._build_user_interest_vector(user_id)

        if user_vector is None:
            return self.image_dao.get_image_paths()[:self.config.default_recommend_num]

        candidates = self._perform_partitioned_search(user_vector)
        final_results = candidates[:self.config.default_recommend_num]

        return [path for path, _ in final_results]

    def _generate_reason(self, behavior: dict) -> str:
        reasons = []
        if len(behavior["search_history"]) > 0:
            reasons.append("搜索记录")
        if len(behavior["click_history"]) > 0:
            reasons.append("点击偏好")
        return f"🎯 个性化推荐（基于您的{('和'.join(reasons))}）"


# -------------------------- 行为跟踪服务 --------------------------
class BehaviorTrackService:
    # 🚨 修正 __init__，注入 SearchService
    def __init__(self, config: AppConfig, behavior_dao: UserBehaviorDAO,
                 recommend_service: RecommendService, search_service: SearchService):
        self.config = config
        self.behavior_dao = behavior_dao
        self.recommend_service = recommend_service
        self.search_service = search_service  # 依赖注入
        self.caption_max_display_length = 50

    def track_recommend_click(self, user_id: Optional[int], click_index: int) -> Tuple[
        List[Tuple[Image.Image, str]], str]:
        """记录推荐列表的点击"""
        if user_id is None:
            return self.recommend_service.get_personalized_recommend(None)

        if click_index < 0:
            return self.recommend_service.get_personalized_recommend(user_id)

        candidate_paths = self.recommend_service._generate_recommendation_paths(user_id)

        if 0 <= click_index < len(candidate_paths):
            self.behavior_dao.add_behavior(user_id, "click_history", candidate_paths[click_index])
            logger.info(f"用户 {user_id} 跟踪推荐点击: {candidate_paths[click_index]}")

        return self.recommend_service.get_personalized_recommend(user_id)

    def track_search_click(self, user_id: Optional[int], click_index: int) -> str:
        """
        🚨 NEW: 记录用户在搜索结果中的点击
        """
        if user_id is None:
            return "请先登录"

        # 从 SearchService 获取缓存的搜索结果路径
        path = self.search_service.get_cached_path(user_id, click_index)

        if path:
            self.behavior_dao.add_behavior(user_id, "click_history", path)
            logger.info(f"用户 {user_id} 点击搜索结果: {path}")
            return f"已记录点击: {os.path.basename(path)}"
        return "点击无效 (索引越界或未找到缓存)"

    def get_user_activity_history(self, user_id: Optional[int]) -> List[str]:
        if user_id is None:
            return ["请先登录以查看您的活动记录。"]

        raw_history = self.behavior_dao.get_full_activity_history(user_id)
        if not raw_history:
            return ["您目前没有活动记录。请尝试搜索或点击推荐商品。"]

        formatted_list = []
        for item in raw_history:
            timestamp_str = item['timestamp'].split('.')[0]
            value = item['value']

            if item['type'] == 'search':
                formatted_list.append(f"[{timestamp_str}] 🔎 **搜索**: “{value}”")
            elif item['type'] == 'click':
                caption = self.recommend_service.image_dao.get_caption_by_path(value)
                display_caption = caption
                if len(caption) > self.caption_max_display_length:
                    display_caption = caption[:self.caption_max_display_length] + '...'
                formatted_list.append(f"[{timestamp_str}] ✨ **点击**: “{display_caption}”")

        return formatted_list

    def delete_user_history(self, user_id: Optional[int]) -> bool:
        if user_id is None:
            return False
        return self.behavior_dao.delete_all_behavior(user_id)