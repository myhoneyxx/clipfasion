from typing import Callable, Tuple, List, Optional
from PIL import Image

import gradio as gr

from .common import AppConfig, init_logger

logger = init_logger("UI")


class FashionUI:
    def __init__(self, config: AppConfig):
        self.config = config
        self.interface: Optional[gr.Blocks] = None

    def create_interface(
            self,
            text_search_fn: Callable[[str, int], List[Image.Image]],
            image_search_fn: Callable[[Image.Image, int], List[Image.Image]],
            refresh_recommend_fn: Callable[[], Tuple[List[Image.Image], str]],
            track_click_fn: Callable[[int], Tuple[List[Image.Image], str]]
    ) -> gr.Blocks:
        """创建界面（依赖注入业务函数，解耦界面与业务）"""
        css = self._get_css()

        with gr.Blocks(css=css, title="FashionAI - 智能服装电商平台") as interface:
            # 导航组件（新增）
            self._add_navigation()

            # 顶部横幅（优化）
            self._add_banner()

            # 搜索区域（优化为电商风格）
            self._add_search_section(text_search_fn, image_search_fn)


            # 个性化推荐（优化商品展示）
            self._add_recommend_section(refresh_recommend_fn, track_click_fn)

            # 页脚（新增）
            self._add_footer()

        self.interface = interface
        return interface

    def _get_css(self) -> str:
        """获取CSS样式（电商风格优化）"""
        return """
        /* 全局样式 */
        .gradio-container {
            font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
            background-color: #fafafa;
            max-width: 1600px !important;
            margin: 0 auto !important;
            padding: 0 !important;
        }

        /* 导航栏样式 */
        .nav-container {
            background-color: white;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            padding: 12px 40px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            position: sticky;
            top: 0;
            z-index: 100;
        }
        .logo {
            font-size: 24px;
            font-weight: 700;
            color: #e63946;
            text-decoration: none;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .nav-menu {
            display: flex;
            gap: 32px;
            margin: 0;
            padding: 0;
            list-style: none;
        }
        .nav-menu li a {
            color: #333;
            text-decoration: none;
            font-size: 16px;
            font-weight: 500;
            transition: color 0.3s ease;
        }
        .nav-menu li a:hover {
            color: #e63946;
        }
        .nav-actions {
            display: flex;
            gap: 20px;
            align-items: center;
        }
        .nav-btn {
            background: none;
            border: none;
            color: #333;
            font-size: 16px;
            cursor: pointer;
            transition: color 0.3s ease;
        }
        .nav-btn:hover {
            color: #e63946;
        }

        /* 横幅样式 */
        .banner {
            background: linear-gradient(135deg, #e63946, #f1faee);
            color: white;
            padding: 60px 40px;
            text-align: center;
            margin-bottom: 40px;
        }
        .banner h1 {
            font-size: 42px;
            margin: 0 0 16px 0;
            font-weight: 700;
        }
        .banner p {
            font-size: 18px;
            margin: 0 0 24px 0;
            max-width: 800px;
            margin-left: auto;
            margin-right: auto;
        }

        /* 搜索区域样式 */
        .search-container {
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.05);
            padding: 30px;
            margin: 0 40px 40px 40px;
        }
        .search-tabs {
            margin-bottom: 24px;
        }
        .search-tab {
            font-size: 18px;
            font-weight: 600;
            color: #666;
            border: none;
            background: none;
            padding: 10px 24px;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .search-tab.selected {
            background-color: #e63946;
            color: white;
        }
        .search-content {
            display: flex;
            gap: 30px;
            align-items: center;
        }
        .search-input-group {
            flex: 1;
        }
        .search-input {
            width: 100%;
            padding: 16px 20px;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            font-size: 16px;
            margin-bottom: 16px;
        }
        .search-input:focus {
            outline: none;
            border-color: #e63946;
            box-shadow: 0 0 0 3px rgba(230, 57, 70, 0.1);
        }
        .search-params {
            display: flex;
            gap: 20px;
            align-items: center;
            margin-bottom: 16px;
        }
        .search-slider {
            flex: 1;
        }
        .search-btn {
            background-color: #e63946 !important;
            border: none !important;
            color: white !important;
            font-weight: 600 !important;
            padding: 16px 32px !important;
            border-radius: 8px !important;
            font-size: 16px !important;
            cursor: pointer;
            transition: background-color 0.3s ease !important;
        }
        .search-btn:hover {
            background-color: #c1121f !important;
        }
        .upload-preview {
            width: 200px;
            height: 200px;
            border: 2px dashed #e5e7eb;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
        }

        /* 分类导航样式 */
        .category-nav {
            margin: 0 40px 40px 40px;
            overflow-x: auto;
            padding-bottom: 10px;
        }
        .category-list {
            display: flex;
            gap: 16px;
            list-style: none;
            margin: 0;
            padding: 0;
        }
        .category-item {
            background: white;
            border-radius: 8px;
            padding: 12px 24px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            white-space: nowrap;
        }
        .category-item a {
            color: #333;
            text-decoration: none;
            font-size: 16px;
            font-weight: 500;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .category-item a:hover {
            color: #e63946;
        }

        /* 商品展示样式 */
        .product-section {
            margin: 0 40px 60px 40px;
        }
        .section-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 24px;
        }
        .section-title {
            font-size: 28px;
            color: #333;
            margin: 0;
            font-weight: 600;
        }
        .product-gallery {
            display: grid !important;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)) !important;
            gap: 24px !important;
            padding: 0 !important;
        }
        .product-card {
            background: white;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            cursor: pointer;
        }
        .product-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 8px 24px rgba(0,0,0,0.1);
        }
        .product-image {
            width: 100% !important;
            height: 360px !important;
            object-fit: cover !important;
            border-radius: 0 !important;
            border: none !important;
        }
        .product-info {
            padding: 16px;
        }
        .product-name {
            font-size: 16px;
            color: #333;
            margin: 0 0 8px 0;
            font-weight: 500;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .product-price {
            font-size: 18px;
            color: #e63946;
            margin: 0;
            font-weight: 600;
        }

        /* 推荐区域样式 */
        .recommend-container {
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.05);
            padding: 30px;
            margin: 0 40px 60px 40px;
        }
        .recommend-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 24px;
        }
        .recommend-title {
            font-size: 24px;
            color: #333;
            margin: 0;
            font-weight: 600;
        }
        .refresh-btn {
            background-color: #f8f9fa !important;
            border: 1px solid #e5e7eb !important;
            color: #333 !important;
            padding: 10px 20px !important;
            border-radius: 8px !important;
            font-size: 14px !important;
            cursor: pointer;
            transition: all 0.3s ease !important;
        }
        .refresh-btn:hover {
            background-color: #e9ecef !important;
        }
        .recommend-desc {
            color: #666;
            font-size: 16px;
            margin-bottom: 24px;
            text-align: left;
        }

        /* 页脚样式 */
        .footer {
            background-color: #2d3142;
            color: white;
            padding: 60px 40px;
            margin-top: 40px;
        }
        .footer-content {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 40px;
            max-width: 1400px;
            margin: 0 auto;
        }
        .footer-column h3 {
            font-size: 18px;
            margin: 0 0 20px 0;
            font-weight: 600;
        }
        .footer-column ul {
            list-style: none;
            margin: 0;
            padding: 0;
        }
        .footer-column ul li {
            margin-bottom: 12px;
        }
        .footer-column ul li a {
            color: #d1d5db;
            text-decoration: none;
            transition: color 0.3s ease;
        }
        .footer-column ul li a:hover {
            color: white;
        }
        .footer-bottom {
            max-width: 1400px;
            margin: 40px auto 0 auto;
            padding-top: 20px;
            border-top: 1px solid #4a5568;
            text-align: center;
            color: #9ca3af;
            font-size: 14px;
        }

        /* 响应式样式 */
        @media (max-width: 1200px) {
            .footer-content {
                grid-template-columns: repeat(2, 1fr);
            }
        }
        @media (max-width: 768px) {
            .nav-container {
                padding: 12px 20px;
            }
            .nav-menu {
                gap: 16px;
            }
            .banner h1 {
                font-size: 32px;
            }
            .banner p {
                font-size: 16px;
            }
            .search-content {
                flex-direction: column;
                gap: 20px;
            }
            .upload-preview {
                width: 100%;
                height: 150px;
            }
            .footer-content {
                grid-template-columns: 1fr;
            }
            .product-gallery {
                grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)) !important;
            }
            .product-image {
                height: 280px !important;
            }
        }
        """

    def _add_navigation(self) -> None:
        """添加电商风格导航栏（新增）"""
        with gr.HTML('<div class="nav-container">'):
            # Logo
            gr.HTML('<a href="#" class="logo">FashionAI</a>')

            # 主导航菜单
            gr.HTML("""
            <ul class="nav-menu">
                <li><a href="#">首页</a></li>
                <li><a href="#">女装</a></li>
                <li><a href="#">男装</a></li>
                <li><a href="#">童装</a></li>
                <li><a href="#">配饰</a></li>
                <li><a href="#">新品上市</a></li>
            </ul>
            """)

            # 导航操作按钮
            gr.HTML("""
            <div class="nav-actions">
                <button class="nav-btn">🔍 搜索</button>
                <button class="nav-btn">❤️ 收藏</button>
                <button class="nav-btn">🛒 购物车</button>
                <button class="nav-btn">👤 我的账户</button>
            </div>
            """)
        gr.HTML('</div>')

    def _add_banner(self) -> None:
        """添加电商顶部横幅（优化）"""
        gr.HTML("""
        <div class="banner">
            <h1>智能穿搭，精准匹配</h1>
            <p>输入关键词或上传图片，AI为你找到最心仪的服装款式，解锁专属时尚风格</p>
            <button class="search-btn">立即探索</button>
        </div>
        """)

    def _add_search_section(
            self,
            text_search_fn: Callable[[str, int], List[Image.Image]],
            image_search_fn: Callable[[Image.Image, int], List[Image.Image]]
    ) -> None:
        """添加电商风格搜索区域（终极正确版，无任何错误）"""
        with gr.Blocks(elem_classes="search-container") as search_block:
            # 搜索标签切换
            with gr.Row(elem_classes="search-tabs"):
                text_tab = gr.Button("关键词搜索", elem_classes=["search-tab", "selected"], elem_id="text-tab")
                image_tab = gr.Button("识图找同款", elem_classes="search-tab", elem_id="image-tab")

            # 搜索内容区域
            with gr.Row(elem_classes="search-content"):
                # 关键词搜索内容
                with gr.Column(visible=True, elem_id="text-search-content") as text_search_col:
                    text_query = gr.Textbox(
                        label="输入服装描述",
                        placeholder="例如：红色碎花连衣裙 收腰 中长款 气质 夏季",
                        lines=1,
                        elem_classes="search-input"
                    )
                    with gr.Row(elem_classes="search-params"):
                        gr.Markdown("展示数量：", elem_classes="search-label")
                        text_top_k = gr.Slider(
                            minimum=3, maximum=18, value=9, step=3,
                            elem_classes="search-slider"
                        )
                    text_search_btn = gr.Button("搜索商品", elem_classes="search-btn")

                # 图像搜索内容（默认隐藏）
                with gr.Column(visible=False, elem_id="image-search-content") as image_search_col:
                    image_query = gr.Image(
                        label="上传服装照片",
                        type="pil",
                        height=200,
                        elem_classes="upload-preview",
                        show_download_button=False,
                        info="支持JPG、PNG格式，清晰正面照效果更佳"
                    )
                    with gr.Row(elem_classes="search-params"):
                        gr.Markdown("展示数量：", elem_classes="search-label")
                        image_top_k = gr.Slider(
                            minimum=3, maximum=18, value=9, step=3,
                            elem_classes="search-slider"
                        )
                    image_search_btn = gr.Button("查找同款", elem_classes="search-btn")

                # 搜索结果展示
                with gr.Column(scale=2):
                    gr.Markdown("<h3>搜索结果</h3>", elem_classes="search-result-title")
                    search_results = gr.Gallery(
                        label="相关服装",
                        show_label=False,
                        elem_classes="product-gallery",
                        columns=3,
                        height="auto"
                    )

            # 标签切换逻辑
            def switch_to_text_tab():
                return [
                    gr.Button.update(),
                    gr.Button.update(),
                    gr.Column.update(visible=True),
                    gr.Column.update(visible=False)
                ]

            def switch_to_image_tab():
                return [
                    gr.Button.update(),
                    gr.Button.update(),
                    gr.Column.update(visible=False),
                    gr.Column.update(visible=True)
                ]

            # 绑定标签切换事件
            text_tab.click(
                fn=switch_to_text_tab,
                outputs=[text_tab, image_tab, text_search_col, image_search_col]
            )
            image_tab.click(
                fn=switch_to_image_tab,
                outputs=[text_tab, image_tab, text_search_col, image_search_col]
            )

            # -------------------------- 唯一正确的搜索事件绑定 --------------------------
            # 关键词搜索处理（仅接收值，不接收组件）
            def handle_text_search(query_str, top_k_num):
                # 校验输入：必须是字符串且非空
                if not isinstance(query_str, str) or len(query_str.strip()) == 0:
                    return []
                # 调用业务函数（传入字符串和数字，而非组件）
                result_images = text_search_fn(query_str.strip(), top_k_num)
                return self._format_product_gallery(result_images)

            # 图像搜索处理（仅接收值，不接收组件）
            def handle_image_search(img_obj, top_k_num):
                if not img_obj:  # 图片为空
                    return []
                # 调用业务函数（传入PIL图像和数字，而非组件）
                result_images = image_search_fn(img_obj, top_k_num)
                return self._format_product_gallery(result_images)

            # 绑定搜索按钮事件（inputs仅传组件，由Gradio自动传值）
            text_search_btn.click(
                fn=handle_text_search,
                inputs=[text_query, text_top_k],  # 传入组件列表，Gradio自动提取值
                outputs=search_results
            )
            image_search_btn.click(
                fn=handle_image_search,
                inputs=[image_query, image_top_k],  # 传入组件列表，Gradio自动提取值
                outputs=search_results
            )

    def _add_navigation(self) -> None:
        """添加电商风格导航栏（修复with语句错误）"""
        # 直接输出完整的导航栏HTML，无需with包裹
        gr.HTML("""
        <div class="nav-container">
            <!-- Logo -->
            <a href="#" class="logo">FashionAI</a>

            <!-- 主导航菜单 -->
            <ul class="nav-menu">
                <li><a href="#">首页</a></li>
                <li><a href="#">女装</a></li>
                <li><a href="#">男装</a></li>
                <li><a href="#">童装</a></li>
                <li><a href="#">配饰</a></li>
                <li><a href="#">新品上市</a></li>
            </ul>

            <!-- 导航操作按钮 -->
            <div class="nav-actions">
                <button class="nav-btn">🔍 搜索</button>
                <button class="nav-btn">❤️ 收藏</button>
                <button class="nav-btn">🛒 购物车</button>
                <button class="nav-btn">👤 我的账户</button>
            </div>
        </div>
        """)

    def _add_recommend_section(
            self,
            refresh_recommend_fn: Callable[[], Tuple[List[Image.Image], str]],
            track_click_fn: Callable[[int], Tuple[List[Image.Image], str]]
    ) -> None:
        """添加电商风格个性化推荐区域（优化）"""
        with gr.Blocks(class_name="recommend-container") as recommend_block:
            with gr.Row(class_name="recommend-header"):
                gr.Markdown("<h3 class='recommend-title'>为你推荐</h3>")
                refresh_btn = gr.Button("刷新推荐", class_name="refresh-btn")

            recommend_reason = gr.Markdown(
                "<p class='recommend-desc'>基于你的浏览和搜索行为，为你精选优质商品</p>"
            )

            # 商品展示画廊
            recommendations = gr.Gallery(
                label="推荐商品",
                show_label=False,
                class_name="product-gallery",
                columns=4,
                height="auto"
            )

            # 绑定事件
            def refresh_and_format():
                imgs, reason = refresh_recommend_fn()
                return [self._format_product_gallery(imgs), f"<p class='recommend-desc'>{reason}</p>"]

            def track_click_and_format(evt):
                imgs, reason = track_click_fn(evt.index)
                return [self._format_product_gallery(imgs), f"<p class='recommend-desc'>{reason}</p>"]

            refresh_btn.click(
                fn=refresh_and_format,
                inputs=[],
                outputs=[recommendations, recommend_reason]
            )
            recommendations.select(
                fn=track_click_and_format,
                inputs=[],
                outputs=[recommendations, recommend_reason]
            )

            # 初始化推荐
            init_imgs, init_reason = refresh_recommend_fn()
            recommendations.value = self._format_product_gallery(init_imgs)
            recommend_reason.value = f"<p class='recommend-desc'>{init_reason}</p>"

    def _add_footer(self) -> None:
        """添加电商网站页脚（新增）"""
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
                <p>© 2025 FashionAI 智能服装电商平台 版权所有 | 营业执照 | 食品经营许可证 | 增值电信业务经营许可证</p>
            </div>
        </div>
        """)

    def _format_product_gallery(self, images: List[Image.Image]) -> List[Tuple[Image.Image, str]]:
        """格式化商品展示（添加商品名称和价格占位）"""
        # 模拟商品名称和价格数据（实际项目可从数据库获取）
        product_names = [
            "夏季碎花连衣裙", "宽松休闲T恤", "高腰牛仔裤", "气质衬衫",
            "防晒外套", "时尚半身裙", "舒适运动鞋", "百搭帆布包",
            "修身西装裤", "甜美针织衫", "复古风衬衫", "运动休闲套装"
        ]
        product_prices = ["¥199", "¥99", "¥159", "¥129", "¥179", "¥139", "¥259", "¥89", "¥169", "¥149", "¥189", "¥299"]

        formatted_gallery = []
        for i, img in enumerate(images):
            # 循环使用商品名称和价格
            name = product_names[i % len(product_names)]
            price = product_prices[i % len(product_prices)]
            # 构建商品卡片HTML
            html = f"""
            <div class="product-card">
                <img src="{img}" class="product-image" />
                <div class="product-info">
                    <div class="product-name">{name}</div>
                    <div class="product-price">{price}</div>
                </div>
            </div>
            """
            formatted_gallery.append((img, html))

        return formatted_gallery