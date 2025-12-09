import tempfile
import os
from typing import List, Tuple, Optional
from PIL import Image

from .common import AppConfig, init_logger
from .dao import UserBehaviorDAO, ImageDAO, IndexDAO

logger = init_logger("Service")


# -------------------------- 搜索服务（单一职责：搜索业务逻辑）- 关键调整 --------------------------
class SearchService:
    def __init__(self, config: AppConfig, clip_matcher, image_dao: ImageDAO, behavior_dao: UserBehaviorDAO):
        self.config = config
        self.clip_matcher = clip_matcher  # 依赖注入，解耦CLIP实现
        self.image_dao = image_dao  # 依赖注入，解耦图片操作
        self.behavior_dao = behavior_dao  # 依赖注入，解耦行为操作

    def text_search(self, query: str, top_k: int) -> List[Image.Image]:
        """文本搜索（解耦搜索逻辑与数据操作）- 无调整（CLIPMatcher接口匹配）"""
        if not query.strip() or top_k < 1:
            return []

        # 记录行为（业务规则：搜索后记录）
        self.behavior_dao.add_behavior("search_history", query.strip())

        try:
            # CLIPMatcher.search_images_by_text 返回 (路径, 分数) 列表，接口匹配
            results = self.clip_matcher.search_images_by_text(query.strip(), top_k=top_k)
            return [self.image_dao.load_image(path) or self.image_dao.get_placeholder() for path, _ in results]
        except Exception as e:
            logger.error(f"文本搜索失败: {str(e)}")
            return []

    def image_search(self, query_image: Image.Image, top_k: int) -> List[Image.Image]:
        """图像搜索（关键调整：CLIPMatcher要求传入图片路径，需临时保存PIL对象）"""
        if not query_image or top_k < 1:
            return []

        try:
            # 关键调整：CLIPMatcher.search_images_by_image 需要传入图片路径，而非PIL对象
            # 临时保存PIL图片为文件
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                query_image.convert("RGB").save(tmp_file, format='JPEG', quality=95)
                tmp_path = tmp_file.name

            # 调用CLIPMatcher接口（传入临时文件路径）
            results = self.clip_matcher.search_images_by_image(
                query_image_path=tmp_path,
                top_k=top_k
            )

            # 加载结果图片
            images = [self.image_dao.load_image(path) or self.image_dao.get_placeholder() for path, _ in results]

            # 清理临时文件
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

            return images
        except Exception as e:
            logger.error(f"图像搜索失败: {str(e)}")
            return []


# -------------------------- 推荐服务（单一职责：推荐业务逻辑）- 关键调整 --------------------------
class RecommendService:
    def __init__(self, config: AppConfig, clip_matcher, image_dao: ImageDAO, behavior_dao: UserBehaviorDAO):
        self.config = config
        self.clip_matcher = clip_matcher
        self.image_dao = image_dao
        self.behavior_dao = behavior_dao

    def get_personalized_recommend(self) -> Tuple[List[Image.Image], str]:
        """个性化推荐（解耦推荐逻辑与数据操作）"""
        behavior = self.behavior_dao.get_behavior()
        has_behavior = any([len(behavior["search_history"]) > 0, len(behavior["click_history"]) > 0])

        # 冷启动：无行为时返回随机推荐
        if not has_behavior:
            return self.image_dao.get_random_images(self.config.default_recommend_num), "📱 暂无个性化行为，为你推荐热门商品～"

        # 生成推荐候选集（业务核心逻辑）
        candidate_paths = self._generate_candidates(behavior)
        # 加载图片（解耦图片加载）
        recommendations = [self.image_dao.load_image(path) or self.image_dao.get_placeholder() for path in
                           candidate_paths]
        # 补充不足数量
        while len(recommendations) < self.config.default_recommend_num:
            recommendations.append(self.image_dao.get_placeholder())

        return recommendations[:self.config.default_recommend_num], self._generate_reason(behavior)

    def _generate_candidates(self, behavior: dict) -> List[str]:
        """生成推荐候选集（关键调整：基于点击历史的图像推荐适配CLIPMatcher接口）"""
        candidates = set()

        # 1. 基于搜索历史（无调整，接口匹配）
        recent_searches = behavior["search_history"][-self.config.recent_behavior_cnt:]
        for keyword in recent_searches:
            try:
                results = self.clip_matcher.search_images_by_text(keyword, top_k=self.config.top_k_recommend)
                candidates.update([path for path, _ in results])
            except Exception as e:
                logger.error(f"搜索历史推荐失败: {str(e)}")

        # 2. 基于点击历史（关键调整：CLIPMatcher需要传入图片路径）
        recent_clicks = behavior["click_history"][-self.config.recent_behavior_cnt:]
        for path in recent_clicks:
            try:
                # 直接传入点击商品的路径（无需临时文件，因为已经是文件路径）
                results = self.clip_matcher.search_images_by_image(
                    query_image_path=path,
                    top_k=self.config.top_k_recommend
                )
                candidates.update([p for p, _ in results])
            except Exception as e:
                logger.error(f"点击历史推荐失败（路径: {path}）: {str(e)}")

        # 3. 补充随机图片（保证多样性）
        candidate_list = list(candidates)
        if len(candidate_list) < self.config.default_recommend_num:
            all_paths = self.image_dao.get_image_paths()
            supplement = [p for p in all_paths if p not in candidates][
                         :self.config.default_recommend_num - len(candidate_list)]
            candidate_list.extend(supplement)

        return candidate_list

    def _generate_reason(self, behavior: dict) -> str:
        """生成推荐理由（解耦理由生成逻辑）"""
        reasons = []
        if len(behavior["search_history"]) > 0:
            reasons.append("搜索记录")
        if len(behavior["click_history"]) > 0:
            reasons.append("点击偏好")
        return f"🎯 个性化推荐（基于你的{('、'.join(reasons))}）"


# -------------------------- 行为跟踪服务（单一职责：行为跟踪逻辑）- 无改动 --------------------------
class BehaviorTrackService:
    def __init__(self, config: AppConfig, behavior_dao: UserBehaviorDAO, recommend_service: RecommendService):
        self.config = config
        self.behavior_dao = behavior_dao
        self.recommend_service = recommend_service  # 依赖注入，解耦推荐服务

    def track_recommend_click(self, click_index: int) -> Tuple[List[Image.Image], str]:
        """跟踪推荐点击并刷新推荐（解耦跟踪逻辑与推荐逻辑）"""
        if click_index < 0:
            return self.recommend_service.get_personalized_recommend()

        # 获取当前推荐的候选路径（复用推荐服务逻辑，避免重复）
        behavior = self.behavior_dao.get_behavior()
        candidate_paths = self.recommend_service._generate_candidates(behavior)

        if 0 <= click_index < len(candidate_paths):
            self.behavior_dao.add_behavior("click_history", candidate_paths[click_index])
            logger.info(f"跟踪推荐点击: {candidate_paths[click_index]}")

        # 刷新推荐
        return self.recommend_service.get_personalized_recommend()