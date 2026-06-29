import contextlib
import os

from  fastapi import  FastAPI
from starlette.middleware.cors import CORSMiddleware
from app.src.response.exception.global_exception import GlobalReOrExHandler
from app.src.common.config.prosgresql_config import async_db_manager
from app.src.response.response_middleware import ResponseMiddleware
from app.src.utils import get_logger
from app.src.controller import account_router, model_config_router, chat_router, conversation_router, tongue_analysis_router
from app.src.middleware.auth_middleware import AuthContextMiddleware

from app.src.common.config.prosgresql_config import create_db_tables

# 🚀 企业级优化：引入 Redis 和 LLM 客户端池
from app.src.common.config.redis_config import redis_manager
from app.src.common.config.llm_client_pool import llm_client_pool
from app.src.core.language_model.llm_provider import normalize_gitcc_gateway_url

# 创建日志记录器
logger = get_logger("app")







def add_middleware(app: FastAPI):
    # 先注册的内层先处理请求；CORS 放在最后注册 = 最外层，保证含 4xx/5xx 的响应也带上跨域头
    app.add_middleware(
        ResponseMiddleware,
        enable_tracing=True,
        enable_request_id=True,
    )
    app.add_middleware(AuthContextMiddleware)
    # allow_credentials=True 时不能使用 allow_origins=["*"]，浏览器会拦截
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )




async  def init_resource():
      """初始化资源 - 企业级优化"""
      logger.info("正在注册数据库")

      # 初始化 PostgreSQL 配置
      await async_db_manager.init()
      logger.info("PostgreSQL 初始化完成")

      try:
          await create_db_tables()
          logger.info("数据库表结构已检查/创建")
      except Exception as e:
          logger.warning(f"自动建表失败（若已手动迁移可忽略）: {e}")
      
      # 初始化 Redis 缓存（可选）
      try:
          redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
          redis_enabled = os.getenv("REDIS_ENABLED", "true").lower() == "true"
          await redis_manager.init(redis_url=redis_url, enabled=redis_enabled)
          logger.info("Redis 缓存初始化完成")
      except Exception as e:
          logger.warning(f"Redis 初始化失败，降级为无缓存模式: {e}")
      
      # 🔥 预热 LLM 客户端（从数据库加载已配置的供应商）
      await warmup_llm_clients()
      
      logger.info("注册数据库完成")


async def warmup_llm_clients():
    """预热 LLM 客户端 - 在应用启动时预创建常用连接"""
    try:
        from app.src.model.model_config_models import SystemModelProvider, UserProviderConfig
        from app.src.utils.auth_utils import decrypt_api_key
        from sqlmodel import select
        
        async with async_db_manager.get_session() as session:
            # 查询所有启用的用户配置（包含 API Key）
            stmt = select(UserProviderConfig, SystemModelProvider).join(
                SystemModelProvider,
                UserProviderConfig.provider_id == SystemModelProvider.id
            ).where(
                UserProviderConfig.is_enabled == True,
                UserProviderConfig.api_key.isnot(None)
            ).limit(10)  # 限制预热数量，避免启动过慢
            
            result = await session.exec(stmt)
            configs_with_providers = result.all()
            
            # 获取配置信息
            warmup_configs = []
            for config, provider in configs_with_providers:
                try:
                    api_key = decrypt_api_key(config.api_key)
                    base_url = normalize_gitcc_gateway_url(
                        config.base_url_override or provider.default_base_url
                    )
                    if api_key and base_url:
                        warmup_configs.append({
                            'provider_id': str(provider.id),
                            'api_key': api_key,
                            'base_url': base_url
                        })
                        logger.debug(f"准备预热: {provider.label}")
                except Exception as e:
                    logger.debug(f"解密 API Key 失败 [{provider.label}]: {e}")
            
            # 执行预热
            if warmup_configs:
                warmed = await llm_client_pool.warmup(warmup_configs)
                logger.info(f"LLM 客户端预热完成，共 {warmed} 个")
            else:
                logger.info("无可预热的 LLM 客户端配置")
                
    except Exception as e:
        logger.warning(f"LLM 客户端预热失败: {e}")





def register_routers(app: FastAPI) -> None:
    """注册业务路由（同步，在 create_app 中调用）。

    原先仅在 lifespan 里注册，若进程未完整跑过启动流程或热重载未加载最新 controller，
    会出现 OpenAPI 缺路由、部分业务 GET 返回 404。
    """
    logger.info("正在注册路由")
    app.include_router(account_router)
    app.include_router(model_config_router)
    app.include_router(chat_router)
    app.include_router(conversation_router)
    app.include_router(tongue_analysis_router)
    logger.info("注册路由完成")






@contextlib.asynccontextmanager
async def life_span(app:FastAPI):
    """应用生命周期管理 - 企业级优化"""
    logger.info("正在启动 FastAPI 应用")
    try:
         # 初始化数据库和缓存（路由已在 create_app 中注册）
         await init_resource()
         logger.info("应用启动完成，准备就绪")
         yield
    finally:
         logger.info("正在关闭应用...")
         # 关闭 Redis
         await redis_manager.close()
         # 关闭 LLM 客户端池
         await llm_client_pool.close_all()
         # 关闭数据库
         await async_db_manager.close()
         logger.info("应用关闭完成")

def create_app():
    logger.info("创建 FastAPI 应用实例")
    
    app = FastAPI(
        title="zhongyi-agentic",
        description="多智能体中医问诊 API",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=life_span
    )
    logger.info("正在注册全局异常管理器")
    GlobalReOrExHandler(app)
    logger.info("注册全局异常管理器成功")
    logger.info("正在注册中间件")
    add_middleware(app)
    logger.info("注册中间件成功")
    register_routers(app)
    return app



