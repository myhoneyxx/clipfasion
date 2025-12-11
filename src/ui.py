from typing import Callable, Tuple, List, Optional
from PIL import Image

import gradio as gr

from .common import AppConfig, init_logger

logger = init_logger("UI")

# 🚨 全局常量，定义 Caption 的最大展示长度 (保留，但 Python 代码中不再用于截断)
CAPPTION_MAX_DISPLAY_LENGTH = 50


class FashionUI:
    def __init__(self, config: AppConfig):
        self.config = config
        self.interface: Optional[gr.Blocks] = None

    def create_interface(
            self,
            # 改造: Service 层返回 List[Tuple[Image.Image, str]] (Image, Caption)
            text_search_fn: Callable[[str, int, Optional[int]], List[Tuple[Image.Image, str]]],
            image_search_fn: Callable[[Image.Image, int, Optional[int]], List[Tuple[Image.Image, str]]],

            # 改造: Service 层返回 Tuple[List[Tuple[Image.Image, str]], str]
            refresh_recommend_fn: Callable[[Optional[int]], Tuple[List[Tuple[Image.Image, str]], str]],
            track_click_fn: Callable[[Optional[int], int], Tuple[List[Tuple[Image.Image, str]], str]],

            # 新增: 认证服务对象
            auth_service,
            # 🚨 NEW FUNCTION: 获取活动记录
            get_activity_history_fn: Callable[[Optional[int]], List[str]],
            # 🚨 NEW FUNCTION: 删除活动记录
            delete_history_fn: Callable[[Optional[int]], bool]
    ) -> gr.Blocks:
        """创建界面（Amazon 极简风格重构，包含用户中心）"""
        css = self._get_css()

        with gr.Blocks(css=css, title="FashionAI - 智能服装电商平台") as interface:

            # 🚨 1. 核心状态组件：存储 (user_id, username) 或 None
            logged_in_user = gr.State(value=None)

            # --------------------- 用户中心 Modal (必须先定义) ---------------------
            history_markdown_output, user_center_modal = self._add_user_center_section(
                get_activity_history_fn, delete_history_fn, logged_in_user
            )

            # --------------------- 认证区域 (Auth) ---------------------
            with gr.Column(visible=True, elem_id="auth-container") as auth_column:
                gr.HTML("<div class='banner'><h1>欢迎登录 FashionAI</h1><p>请登录或注册以体验个性化推荐</p></div>")
                login_output = gr.Markdown("")

                with gr.Row():
                    with gr.Tab("登录", elem_id="login-tab"):
                        login_username = gr.Textbox(label="用户名", placeholder="请输入用户名")
                        login_password = gr.Textbox(label="密码", type="password", placeholder="请输入密码")
                        login_btn = gr.Button("立即登录", variant="primary")

                    with gr.Tab("注册", elem_id="register-tab"):
                        reg_username = gr.Textbox(label="新用户名", placeholder="用户名至少包含一个字母")
                        reg_password = gr.Textbox(label="新密码", type="password", placeholder="密码长度至少6位")
                        reg_btn = gr.Button("创建账户")
                        reg_output = gr.Markdown("")

            # --------------------- 主应用区域 (App) ---------------------
            with gr.Column(visible=False, elem_id="app-container") as app_column:

                # 顶部栏
                user_center_btn = self._add_navigation()

                # 搜索框区域
                search_section_results = self._add_search_section(
                    text_search_fn=text_search_fn,
                    image_search_fn=image_search_fn,
                    logged_in_user=logged_in_user
                )
                search_results_gallery = search_section_results[0]

                # 核心内容区 - 个性化推荐区域
                recommendations_gallery, recommend_reason_md = self._add_recommend_section(
                    refresh_recommend_fn=refresh_recommend_fn,
                    track_click_fn=track_click_fn,
                    logged_in_user=logged_in_user,
                    is_visible=True
                )

                # 底部栏
                self._add_footer()

            # --------------------- 用户中心 Modal 绑定 ---------------------
            def show_user_center(user_info):
                """加载历史记录并显示用户中心 Modal"""
                user_id = user_info[0] if user_info else None

                history_list = get_activity_history_fn(user_id)
                history_markdown = self._format_history_markdown(user_id, history_list)

                return gr.Markdown.update(value=history_markdown), gr.Column.update(visible=True)

            user_center_btn.click(
                fn=show_user_center,
                inputs=[logged_in_user],
                outputs=[history_markdown_output, user_center_modal]
            )

            # --------------------- 认证逻辑绑定 ---------------------

            def handle_login(username, password):
                """处理登录请求"""
                user_id = auth_service.login_user(username, password)
                if user_id:
                    user_info = (user_id, username)
                    welcome_msg = f"<div class='banner'><h1>欢迎回来, {username}</h1><p>已为您切换至个性化推荐</p></div>"

                    init_imgs_enriched, init_reason = refresh_recommend_fn(user_id)

                    return (
                        user_info,
                        gr.Column.update(visible=False),  # 隐藏 Auth
                        gr.Column.update(visible=True),  # 显示 App
                        gr.Gallery.update(visible=False, value=None),  # 隐藏搜索结果
                        gr.Gallery.update(value=self._format_product_gallery(init_imgs_enriched)),  # 刷新推荐结果
                        gr.Markdown.update(value=f"<p class='recommend-desc'>{init_reason}</p>"),  # 刷新推荐理由
                        gr.Button.update(visible=True),  # 显示个人中心按钮
                        gr.Markdown.update(value="✅ 登录成功！")  # login_output
                    )
                else:
                    return gr.State.update(), gr.Column.update(), gr.Column.update(), gr.Gallery.update(), gr.Gallery.update(), gr.Markdown.update(), gr.Button.update(
                        visible=False), gr.Markdown.update(value="❌ 登录失败：用户名或密码错误")

            def handle_register(username, password):
                """处理注册请求"""
                if auth_service.register_user(username, password):
                    return gr.Markdown.update(value="✅ 注册成功，请切换到登录页进行登录")
                else:
                    return gr.Markdown.update(value="❌ 注册失败：用户名已存在或密码长度不足6位")

            # 绑定认证按钮事件
            login_btn.click(
                fn=handle_login,
                inputs=[login_username, login_password],
                outputs=[logged_in_user, auth_column, app_column, search_results_gallery, recommendations_gallery,
                         recommend_reason_md, user_center_btn, login_output]
            )
            reg_btn.click(
                fn=handle_register,
                inputs=[reg_username, reg_password],
                outputs=[reg_output]
            )

            # 初始加载内容 (未登录状态)
            init_imgs_enriched, _ = refresh_recommend_fn(None)
            recommendations_gallery.value = self._format_product_gallery(init_imgs_enriched)
            recommend_reason_md.value = "<p class='recommend-desc'>请先登录或进行搜索以获取个性化推荐。</p>"

        self.interface = interface
        return interface

    def _get_css(self) -> str:
        """获取CSS样式（已修复：Gallery Modal 和全宽展示）"""
        return """
        /* 全局样式 */
        .gradio-container {
            font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
            background-color: #fafafa;
            max-width: 100% !important; 
            margin: 0 !important;        
            padding: 0 !important;
        }

        /* 导航栏样式 (Amazon Navy) */
        .nav-container {
            background-color: #232f3e; 
            color: white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            padding: 8px 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            position: sticky;
            top: 0;
            z-index: 100;
            max-width: 1200px; 
            margin: 0 auto; 
        }
        .logo {
            font-size: 20px;
            font-weight: 700;
            color: #ff9900; /* Amazon Orange */
            text-decoration: none;
        }
        .nav-actions {
            display: flex;
            gap: 15px;
            align-items: center;
        }
        .nav-btn {
            background: none;
            border: none;
            color: white;
            font-size: 14px;
            cursor: pointer;
            transition: color 0.3s ease;
            white-space: nowrap;
        }
        .nav-btn:hover {
            color: #ff9900;
        }

        /* 横幅/欢迎区 (Auth 页面使用) */
        .banner {
            background: none;
            color: #333;
            padding: 20px 40px;
            text-align: center;
            margin-bottom: 20px;
        }
        .banner h1 {
            font-size: 28px;
            margin: 0 0 8px 0;
        }

        /* 搜索区域样式 (核心区域) */
        .search-container {
            background: #fff;
            padding: 30px 40px;
            margin: 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        .search-content {
            max-width: 1000px;
            margin: 0 auto;
            align-items: center;
        }
        .search-input {
            height: 45px;
            border-color: #ff9900; /* Amazon 强调色 */
        }
        .search-btn {
            background-color: #ff9900 !important;
            border: none !important;
            color: #232f3e !important;
            font-weight: 700 !important;
            padding: 12px 24px !important;
            border-radius: 4px !important;
        }

        /* FIX: 修复 Gallery Modal 的关闭按钮超出问题 */
        .modal-close {
            position: fixed !important; 
            top: 10px !important;
            right: 10px !important;
            z-index: 10000 !important; 
            cursor: pointer;
            background-color: rgba(255, 255, 255, 0.9); 
            border-radius: 50%;
            padding: 5px;
            box-shadow: 0 0 5px rgba(0,0,0,0.2);
        }

        /* 推荐区域 */
        .recommend-container {
            max-width: 1200px; 
            margin: 30px auto; 
            padding: 20px 0;
        }
        .section-title {
            font-size: 24px;
            margin-bottom: 15px;
            color: #333;
        }
        .product-gallery {
            gap: 20px !important;
            padding: 0 !important;
        }
        .product-card {
            /* 🚨 淘宝优化基础样式 */
            border: 1px solid #ddd;
            border-radius: 4px;
            box-shadow: none;
            transition: box-shadow 0.2s;
            overflow: hidden;
            background: white;
            display: flex;
            flex-direction: column;
            justify-content: space-between; 
            position: relative; 
        }
        .product-card:hover {
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        .product-image {
            height: 300px !important;
        }

        .wishlist-icon {
            display: none; /* 隐藏心愿单图标 */
        }

        .product-info {
            /* 🚨 优化信息区，仅保留名称，最小化高度 */
            padding: 8px 10px; 
            height: auto; 
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
        }
        .product-name {
            /* 🚨 FIX: 移除所有强制截断属性 */
            font-size: 14px;
            font-weight: 500;
            white-space: normal; /* 允许换行 */
            overflow: visible;  /* 允许溢出（即允许显示所有行） */
            text-overflow: clip; /* 允许溢出 */
            display: block; /* 覆盖 -webkit-box */
            -webkit-line-clamp: unset; 
            -webkit-box-orient: unset;
            margin-bottom: 0; 
            height: auto; /* 自动调整高度以容纳所有文本 */
            color: #333;
        }
        /* 🚨 移除价格和状态栏相关 CSS */
        .product-price, .product-sales, .product-status-bar {
            display: none; 
        }

        /* 🚨 NEW: 用户中心 Modal 样式 */
        .user-center-modal {
            position: fixed !important;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 90%;
            max-width: 1000px;
            max-height: 80vh;
            background: white;
            border-radius: 8px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.5);
            z-index: 2000;
            padding: 30px;
            overflow-y: auto;
        }
        .modal-close-btn {
            position: absolute !important;
            top: 10px !important;
            right: 10px !important;
            background: none !important;
            border: none !important;
            color: #e63946 !important;
            font-weight: 600 !important;
            cursor: pointer;
            z-index: 2001;
            padding: 5px;
            font-size: 14px !important;
        }

        /* 页脚样式 */
        .footer {
            background-color: #232f3e;
            color: white;
            padding: 30px 40px;
            margin-top: 50px;
        }
        """

    def _add_navigation(self) -> gr.Button:
        """添加顶部导航栏（Amazon 风格，返回个人中心按钮）"""
        with gr.Blocks():
            gr.HTML("""
            <div class="nav-container">
                <a href="#" class="logo">FashionAI</a>
                <div class="nav-actions">
                    <button class="nav-btn">❤️ 收藏</button>
                    <button class="nav-btn">🛒 购物车</button>
                </div>
            </div>
            """)
            user_center_btn = gr.Button("👤 个人中心", elem_classes="nav-btn", visible=False, scale=0)

        return user_center_btn

    def _add_banner(self) -> gr.Markdown:
        """
        Amazon 风格不显示大 Banner，此函数仅返回一个占位 Markdown
        """
        return gr.Markdown(value="", visible=False, show_label=False)

    def _add_search_section(
            self,
            # 改造: Service 层返回 List[Tuple[Image.Image, str]] (Image, Caption)
            text_search_fn: Callable[[str, int, Optional[int]], List[Tuple[Image.Image, str]]],
            image_search_fn: Callable[[Image.Image, int, Optional[int]], List[Tuple[Image.Image, str]]],
            logged_in_user: gr.State
    ) -> Tuple[gr.Gallery]:
        """🚨 核心：Amazon 搜索优先，搜索框放大居中"""

        # 搜索结果 Gallery 必须在外部定义以便返回和后续更新
        with gr.Column(elem_classes="search-container") as search_block:

            gr.Markdown("<h2>搜索时尚单品</h2>", elem_classes="section-title")  # 标题突出搜索

            # 搜索标签切换
            with gr.Row(elem_classes="search-tabs"):
                text_tab = gr.Button("关键词搜索", elem_classes=["search-tab", "selected"], elem_id="text-tab")
                image_tab = gr.Button("识图找同款", elem_classes="search-tab", elem_id="image-tab")

            # 搜索内容区域
            with gr.Row(elem_classes="search-content"):
                # 关键词搜索内容
                with gr.Column(visible=True, elem_id="text-search-content", scale=4) as text_search_col:
                    with gr.Row():
                        text_query = gr.Textbox(
                            placeholder="搜索 T恤、连衣裙、鞋子等...",
                            lines=1,
                            elem_classes="search-input",
                            show_label=False
                        )
                        text_search_btn = gr.Button("搜索商品", elem_classes="search-btn", min_width=150)

                    with gr.Row(elem_classes="search-params"):
                        gr.Markdown("展示数量：", elem_classes="search-label", min_width=100)
                        text_top_k = gr.Slider(
                            minimum=3, maximum=18, value=9, step=3,
                            elem_classes="search-slider", label=None
                        )

                # 图像搜索内容（默认隐藏）
                with gr.Column(visible=False, elem_id="image-search-content", scale=4) as image_search_col:
                    with gr.Row():
                        image_query = gr.Image(
                            label="上传服装照片",
                            type="pil",
                            height=200,
                            elem_classes="upload-preview",
                            show_download_button=False,
                            info="支持JPG/PNG格式"
                        )
                        image_search_btn = gr.Button("查找同款", elem_classes="search-btn", min_width=150)

                    with gr.Row(elem_classes="search-params"):
                        gr.Markdown("展示数量：", elem_classes="search-label", min_width=100)
                        image_top_k = gr.Slider(
                            minimum=3, maximum=18, value=9, step=3,
                            elem_classes="search-slider", label=None
                        )

            # 搜索结果 Gallery
            search_results = gr.Gallery(
                label="相关商品",
                show_label=False,
                elem_classes="product-gallery",
                columns=3,
                height="auto",
                visible=False  # 搜索结果默认隐藏
            )

            # 标签切换逻辑 (修复 Gradio TypeError)
            def switch_to_text_tab():
                return [gr.Button.update(), gr.Button.update(), gr.Column.update(visible=True),
                        gr.Column.update(visible=False)]

            def switch_to_image_tab():
                return [gr.Button.update(), gr.Button.update(), gr.Column.update(visible=False),
                        gr.Column.update(visible=True)]

            # 绑定标签切换事件
            text_tab.click(fn=switch_to_text_tab, outputs=[text_tab, image_tab, text_search_col, image_search_col])
            image_tab.click(fn=switch_to_image_tab, outputs=[text_tab, image_tab, text_search_col, image_search_col])

            # -------------------------- 搜索事件绑定 --------------------------

            def handle_text_search(user_info, query_str, top_k_num):
                if not isinstance(query_str, str) or len(query_str.strip()) == 0:
                    return gr.Gallery.update(visible=True, value=self._format_product_gallery([]))

                user_id = user_info[0] if user_info else None
                result_images_enriched = text_search_fn(query_str.strip(), top_k_num, user_id)
                return gr.Gallery.update(visible=True, value=self._format_product_gallery(result_images_enriched))

            def handle_image_search(user_info, img_obj, top_k_num):
                if not img_obj:
                    return gr.Gallery.update(visible=True, value=self._format_product_gallery([]))

                user_id = user_info[0] if user_info else None
                result_images_enriched = image_search_fn(img_obj, top_k_num, user_id)
                return gr.Gallery.update(visible=True, value=self._format_product_gallery(result_images_enriched))

            # 绑定搜索按钮事件
            text_search_btn.click(fn=handle_text_search, inputs=[logged_in_user, text_query, text_top_k],
                                  outputs=search_results)
            image_search_btn.click(fn=handle_image_search, inputs=[logged_in_user, image_query, image_top_k],
                                   outputs=search_results)

            return (search_results,)  # 返回搜索结果 Gallery

    def _add_recommend_section(
            self,
            # 改造: Service 层返回 Tuple[List[Tuple[Image.Image, str]], str]
            refresh_recommend_fn: Callable[[Optional[int]], Tuple[List[Tuple[Image.Image, str]], str]],
            track_click_fn: Callable[[Optional[int], int], Tuple[List[Tuple[Image.Image, str]], str]],
            logged_in_user: gr.State,
            is_visible: bool  # 🚨 参数控制可见性
    ) -> Tuple[gr.Gallery, gr.Markdown]:
        """添加电商风格个性化推荐区域（翻译中文）"""
        with gr.Column(visible=is_visible, elem_classes="recommend-container") as recommend_block:
            with gr.Row(class_name="recommend-header"):
                gr.Markdown("<h3 class='recommend-title'>为你推荐 (个人中心)</h3>")
                refresh_btn = gr.Button("刷新推荐", class_name="refresh-btn")

                # 核心组件 1
            recommend_reason = gr.Markdown(
                value="<p class='recommend-desc'>请先登录或进行搜索以获取个性化推荐。</p>",
                show_label=False
            )

            # 核心组件 2
            recommendations = gr.Gallery(
                label="推荐商品",
                show_label=False,
                elem_classes="product-gallery",
                columns=4,
                height="auto"
            )

            # 绑定事件
            def refresh_and_format(user_info):
                user_id = user_info[0] if user_info else None
                imgs_enriched, reason = refresh_recommend_fn(user_id)
                return [self._format_product_gallery(imgs_enriched), f"<p class='recommend-desc'>{reason}</p>"]

            def track_click_and_format(user_info, evt: gr.SelectData):
                user_id = user_info[0] if user_info else None
                imgs_enriched, reason = track_click_fn(user_id, evt.index)
                return [self._format_product_gallery(imgs_enriched), f"<p class='recommend-desc'>{reason}</p>"]

            refresh_btn.click(
                fn=refresh_and_format,
                inputs=[logged_in_user],
                outputs=[recommendations, recommend_reason]
            )
            recommendations.select(
                fn=track_click_and_format,
                inputs=[logged_in_user],
                outputs=[recommendations, recommend_reason]
            )

            return recommendations, recommend_reason

    def _format_history_markdown(self, user_id: Optional[int], history_list: List[str]) -> str:
        """格式化活动记录列表为 Markdown 字符串"""

        if not user_id:
            return "### ⚠️ 请先登录以查看您的个人信息和活动记录。"

        history_markdown = "### 👤 用户活动记录 (最新至最旧)\n\n"

        if not history_list or (len(history_list) == 1 and history_list[0].startswith("您目前没有")):
            history_markdown += "您目前没有活动记录。请尝试搜索或点击推荐商品。"
        else:
            history_markdown += "\n".join([f"* {item}" for item in history_list])

        return history_markdown

    def _add_user_center_section(self, get_activity_history_fn, delete_history_fn, logged_in_user) -> Tuple[
        gr.Markdown, gr.Column]:
        """
        🚨 NEW FUNCTION: 创建用户中心的模态窗口 (Modal)
        """

        def handle_delete(user_info):
            """处理清空历史记录请求"""
            user_id = user_info[0] if user_info else None

            if delete_history_fn(user_id):
                # 成功删除后，需要重新获取空的记录列表
                new_history = get_activity_history_fn(user_id)
                history_markdown = self._format_history_markdown(user_id, new_history)
                return gr.Markdown.update(value=history_markdown), gr.Markdown.update(
                    value="✅ 记录已清空，推荐模型将重新学习您的偏好。"), gr.Column.update(visible=True)
            else:
                return gr.Markdown.update(), gr.Markdown.update(value="❌ 清空失败或用户未登录。"), gr.Column.update(
                    visible=True)

        # 1. Define Modal Structure
        with gr.Column(visible=False, elem_classes="user-center-modal") as user_center_modal:
            gr.HTML("<h3>👤 个人中心</h3>")

            # 2. Activity History Output
            history_markdown_output = gr.Markdown(label="活动时间线")

            # 3. Action Buttons
            with gr.Row():
                delete_btn = gr.Button("🗑️ 清空所有行为记录", elem_classes="delete-btn", min_width=200)
                delete_status_output = gr.Markdown(value="", show_label=False)
                close_btn = gr.Button("关闭", elem_classes="modal-close-btn", min_width=100)

            # 4. Bind delete logic
            delete_btn.click(
                fn=handle_delete,
                inputs=[logged_in_user],
                outputs=[history_markdown_output, delete_status_output, user_center_modal]
            )

            # 5. Bind close logic
            close_btn.click(
                fn=lambda: gr.Column.update(visible=False),
                outputs=[user_center_modal]
            )

        return history_markdown_output, user_center_modal

    def _add_footer(self) -> None:
        """添加电商网站页脚（翻译中文）"""
        gr.HTML("""
        <div class="footer">
            <div class="footer-content">
                <div class="footer-column">
                    <h3>关于我们</h3>
                    <ul>
                        <li><a href="#">品牌故事</a></li>
                        <li><a href="#">联系我们</a></li>
                        <li><a href="#">招贤纳士</a></li>
                        <li><a href="#">门店地址</a></li>
                    </ul>
                </div>
                <div class="footer-column">
                    <h3>客户服务</h3>
                    <ul>
                        <li><a href="#">购物指南</a></li>
                        <li><a href="#">支付方式</a></li>
                        <li><a href="#">配送说明</a></li>
                        <li><a href="#">售后政策</a></li>
                        <li><a href="#">常见问题</a></li>
                    </ul>
                </div>
                <div class="footer-column">
                    <h3>会员中心</h3>
                    <ul>
                        <li><a href="#">会员注册</a></li>
                        <li><a href="#">会员权益</a></li>
                        <li><a href="#">积分兑换</a></li>
                        <li><a href="#">订单查询</a></li>
                    </ul>
                </div>
                <div class="footer-column">
                    <h3>关注我们</h3>
                    <ul>
                        <li><a href="#">微信公众号</a></li>
                        <li><a href="#">微博</a></li>
                        <li><a href="#">抖音</a></li>
                        <li><a href="#">小红书</a></li>
                    </ul>
                </div>
            </div>
            <div class="footer-bottom">
                <p>© 2025 FashionAI 智能服装电商平台 | 版权所有</p>
            </div>
        </div>
        """)

    def _format_product_gallery(self, enriched_images: List[Tuple[Image.Image, str]]) -> List[Tuple[Image.Image, str]]:
        """
        🚨 核心优化：格式化商品展示（移除购物车按钮，极简布局）。
        """

        CAPTION_MAX_DISPLAY_LENGTH = 50

        formatted_gallery = []
        # 遍历传入的 (Image, Caption) 元组
        for img, full_caption in enriched_images:
            # 1. 移除截断逻辑，直接使用完整的 full_caption

            # 2. 构建商品卡片HTML (极简结构)
            html = f"""
            <div class="product-card">
                <img src="{img}" class="product-image" />

                <div class="product-info">
                    <div class="product-name">{full_caption}</div> 
                </div>
            </div>
            """
            # Gradio Gallery 需要 (Image.Image, HTML/Caption) 格式
            formatted_gallery.append((img, html))

        return formatted_gallery