import logging
from typing import Tuple, List
from PIL import Image

from src.common import AppConfig, init_logger
from src.dao import UserBehaviorDAO, ImageDAO, IndexDAO
from src.service import SearchService, RecommendService, BehaviorTrackService
from src.ui import FashionUI
from src.clip_matcher import CLIPMatcher  # CLIP匹配器也在src目录

logger = init_logger("Main")

def main():
    """应用入口（负责组件组装，解耦模块依赖）"""
    try:
        # 1. 初始化配置（全局唯一，可外部传入修改）
        config = AppConfig()
        logger.info("应用配置初始化完成")

        # 2. 初始化第三方依赖（CLIP匹配器）- 关键调整：传入模型路径和设备
        logger.info(f"正在初始化CLIP匹配器（模型路径：{config.clip_model_path}）")
        clip_matcher = CLIPMatcher(
            model_path=config.clip_model_path,
            device=None  # 自动选择GPU/CPU
        )
        logger.info("CLIP匹配器初始化完成")

        # 3. 初始化数据访问层（DAO）- 依赖配置和CLIP
        user_behavior_dao = UserBehaviorDAO(config)
        image_dao = ImageDAO(config)
        index_dao = IndexDAO(config, clip_matcher)
        # 加载/构建索引
        index_dao.load_or_build_indexes()
        logger.info("数据访问层初始化完成")

        # 4. 初始化业务逻辑层（Service）- 依赖注入DAO和配置
        search_service = SearchService(config, clip_matcher, image_dao, user_behavior_dao)
        recommend_service = RecommendService(config, clip_matcher, image_dao, user_behavior_dao)
        behavior_track_service = BehaviorTrackService(config, user_behavior_dao, recommend_service)
        logger.info("业务逻辑层初始化完成")

        # 5. 定义业务函数（适配UI的回调格式，解耦UI与Service）
        def text_search_wrapper(query: str, top_k: int) -> List[Image.Image]:
            return search_service.text_search(query, top_k)

        def image_search_wrapper(query_img: Image.Image, top_k: int) -> List[Image.Image]:
            return search_service.image_search(query_img, top_k)

        def refresh_recommend_wrapper() -> Tuple[List[Image.Image], str]:
            return recommend_service.get_personalized_recommend()

        def track_click_wrapper(click_index: int) -> Tuple[List[Image.Image], str]:
            return behavior_track_service.track_recommend_click(click_index)

        # 6. 初始化界面层（UI）- 依赖注入业务函数
        fashion_ui = FashionUI(config)
        interface = fashion_ui.create_interface(
            text_search_fn=text_search_wrapper,
            image_search_fn=image_search_wrapper,
            refresh_recommend_fn=refresh_recommend_wrapper,
            track_click_fn=track_click_wrapper
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