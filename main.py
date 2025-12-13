import logging
from typing import Tuple, List, Optional
from PIL import Image

from src.common import AppConfig, init_logger
from src.dao import UserBehaviorDAO, ImageDAO, IndexDAO
from src.service import SearchService, RecommendService, BehaviorTrackService, AuthService
from src.ui import FashionUI
from src.clip_matcher import CLIPMatcher
from src.auth_dao import UserAuthDAO
from src.db_utils import init_db

logger = init_logger("Main")


def main():
    """应用入口（负责组件组装，解耦模块依赖）"""
    try:
        # 1. 初始化配置和数据库
        config = AppConfig()
        init_db()  # 确保数据库表结构已创建
        logger.info("应用配置初始化完成")

        # 2. 初始化第三方依赖（CLIP匹配器）
        logger.info(f"正在初始化CLIP匹配器（模型路径：{config.clip_model_path}）")
        clip_matcher = CLIPMatcher(
            model_path=config.clip_model_path,
            device=config.clip_device  # 使用配置中的设备设置
        )
        logger.info("CLIP匹配器初始化完成")

        # 3. 初始化数据访问层（DAO）
        user_behavior_dao = UserBehaviorDAO(config)
        image_dao = ImageDAO(config)
        index_dao = IndexDAO(config, clip_matcher)
        auth_dao = UserAuthDAO()

        # 加载所有索引（包含全局索引和推荐系统的分片索引）
        # 注意：IndexDAO.load_or_build_indexes 已包含分片索引加载逻辑
        index_dao.load_or_build_indexes()
        logger.info("数据访问层初始化完成")

        # 4. 初始化业务逻辑层（Service）
        auth_service = AuthService(auth_dao)
        search_service = SearchService(config, clip_matcher, image_dao, user_behavior_dao)
        recommend_service = RecommendService(config, clip_matcher, image_dao, user_behavior_dao)

        # 🚨 修改点 1: BehaviorTrackService 注入 search_service
        behavior_track_service = BehaviorTrackService(
            config,
            user_behavior_dao,
            recommend_service,
            search_service  # 新增依赖：用于追踪搜索结果点击
        )
        logger.info("业务逻辑层初始化完成")

        # 5. 定义业务函数（适配UI的回调格式，解耦UI与Service）

        # 搜索包装函数
        def text_search_wrapper(query: str, top_k: int, user_id: Optional[int]) -> List[Tuple[Image.Image, str]]:
            return search_service.text_search(query, top_k, user_id)

        def image_search_wrapper(query_img: Image.Image, top_k: int, user_id: Optional[int]) -> List[
            Tuple[Image.Image, str]]:
            return search_service.image_search(query_img, top_k, user_id)

        # 推荐和跟踪包装函数
        def refresh_recommend_wrapper(user_id: Optional[int]) -> Tuple[List[Tuple[Image.Image, str]], str]:
            return recommend_service.get_personalized_recommend(user_id)

        def track_click_wrapper(user_id: Optional[int], click_index: int) -> Tuple[List[Tuple[Image.Image, str]], str]:
            # 这是"推荐列表"的点击
            return behavior_track_service.track_recommend_click(user_id, click_index)

        # 🚨 修改点 2: 新增"搜索结果"点击的包装函数
        def track_search_click_wrapper(user_id: Optional[int], index: int) -> str:
            return behavior_track_service.track_search_click(user_id, index)

        # 用户中心功能包装函数
        def get_activity_history_wrapper(user_id: Optional[int]) -> List[str]:
            return behavior_track_service.get_user_activity_history(user_id)

        def delete_history_wrapper(user_id: Optional[int]) -> bool:
            return behavior_track_service.delete_user_history(user_id)

        # 6. 初始化界面层（UI）- 依赖注入业务函数
        fashion_ui = FashionUI(config)
        interface = fashion_ui.create_interface(
            text_search_fn=text_search_wrapper,
            image_search_fn=image_search_wrapper,
            refresh_recommend_fn=refresh_recommend_wrapper,
            track_click_fn=track_click_wrapper,
            auth_service=auth_service,
            get_activity_history_fn=get_activity_history_wrapper,
            delete_history_fn=delete_history_wrapper,
            # 🚨 修改点 3: 传入搜索点击回调
            track_search_click_fn=track_search_click_wrapper
        )
        logger.info("界面层初始化完成")

        # 7. 启动应用
        logger.info("启动应用...")
        interface.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            debug=False,
            show_error=True,
            inbrowser=True
        )

    except KeyboardInterrupt:
        logger.info("应用被用户中断")
    except Exception as e:
        logger.error(f"应用启动失败: {str(e)}", exc_info=True)
    finally:
        logger.info("应用退出")


if __name__ == "__main__":
    main()