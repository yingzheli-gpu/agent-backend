"""
从 GitCC（New API）公开定价接口拉取模型元数据，写入 system_model_definitions（供应商 name=gitcc）。

定价页: http://api.gitcc.com/pricing
数据接口: http://api.gitcc.com/api/pricing

用法（在 backend 目录）:
  uv run python scripts/import_gitcc_pricing_models.py
  uv run python scripts/import_gitcc_pricing_models.py --all
  uv run python scripts/import_gitcc_pricing_models.py --only gpt-4o,deepseek-chat
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
from sqlalchemy import text

backend_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_path))

from app.src.common.config.prosgresql_config import async_db_manager

DEFAULT_PRICING_URL = "http://api.gitcc.com/api/pricing"

# 默认仅导入「基础常用」模型（须在定价接口中存在）；--all 则导入接口返回的全部条目
DEFAULT_BASIC_MODEL_NAMES: frozenset[str] = frozenset(
    {
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-nano",
        "deepseek-chat",
        "deepseek-reasoner",
        "qwen-plus",
        "qwen-turbo",
        "qwen-max",
        "qwen-flash",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-3-flash-preview",
        "text-embedding-3-small",
        "text-embedding-3-large",
        "o3-mini",
    }
)


def _infer_model_type(item: dict[str, Any]) -> str:
    name = (item.get("model_name") or "").lower()
    tags = (item.get("tags") or "").lower()
    ep = item.get("supported_endpoint_types") or []

    if "image-generation" in ep:
        return "image"
    if "embedding" in name or name.startswith("text-embedding"):
        return "embedding"
    if "rerank" in name:
        return "rerank"
    if "whisper" in name or "tts" in name or "speech" in name:
        return "audio"
    if "sora" in name or "video" in tags:
        return "video"
    if (
        "-vl-" in name
        or name.endswith("-vl")
        or "vl-plus" in name
        or "vl-max" in name
        or "multimodal" in tags
        or "多模态" in (item.get("tags") or "")
    ):
        return "multimodal"
    if "coder" in name or name.startswith("code") or "-code-" in name:
        return "code"
    return "llm"


def _default_features(model_type: str) -> list[str]:
    if model_type == "llm":
        return ["structured_output", "tool_call"]
    if model_type == "multimodal":
        return ["image_input", "tool_call", "structured_output"]
    if model_type == "embedding":
        return ["dense", "batch"]
    if model_type == "rerank":
        return ["rerank", "batch"]
    if model_type == "image":
        return ["text2img"]
    if model_type == "audio":
        return ["speech2text", "tts"]
    if model_type == "video":
        return ["text2video"]
    if model_type == "code":
        return ["completion", "generation", "coding"]
    return []


_CTX_RE = re.compile(r"(\d+(?:\.\d+)?)\s*([KkMm])")


def _parse_context_window(item: dict[str, Any]) -> int:
    text = f"{item.get('tags') or ''} {item.get('description') or ''}"
    best = 4096
    for m in _CTX_RE.finditer(text):
        val = float(m.group(1))
        u = m.group(2).upper()
        if u == "M":
            n = int(val * 1_000_000)
        else:
            n = int(val * 1000)
        if 4096 <= n <= 10_000_000:
            best = max(best, n)
    return best


def _label(item: dict[str, Any]) -> str:
    desc = (item.get("description") or "").strip()
    name = item.get("model_name") or ""
    if desc and len(desc) <= 100:
        return desc
    pretty = name.replace("-", " ").replace("_", " ").strip()
    return (pretty[:100] or name)[:100]


def _pricing_blob(item: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "model_ratio",
        "model_price",
        "completion_ratio",
        "quota_type",
        "vendor_id",
        "supported_endpoint_types",
    )
    return {k: item[k] for k in keys if k in item}


async def _fetch_pricing(url: str) -> list[dict[str, Any]]:
    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        r = await client.get(url)
        r.raise_for_status()
        body = r.json()
    if not body.get("success"):
        raise RuntimeError(f"定价接口 success!=true: {body!r:.200}")
    data = body.get("data")
    if not isinstance(data, list):
        raise RuntimeError("定价接口缺少 data 数组")
    return data


async def _import_rows(
    rows: list[dict[str, Any]],
    *,
    provider_name: str,
) -> None:
    await async_db_manager.init()
    try:
        async with async_db_manager.async_engine.begin() as conn:
            res = await conn.execute(
                text(
                    "SELECT id FROM system_model_providers WHERE LOWER(TRIM(name)) = LOWER(TRIM(:n))"
                ),
                {"n": provider_name},
            )
            pid = res.scalar()
            if not pid:
                raise RuntimeError(
                    f"未找到供应商 name={provider_name!r}，请先启动应用执行 create_db_tables 或运行 import_builtin_models_2026.py"
                )

            for pos, item in enumerate(rows, start=1):
                model_name = item.get("model_name")
                if not model_name or not isinstance(model_name, str):
                    continue
                model_name = model_name.strip()[:100]
                mtype = _infer_model_type(item)
                ctx = _parse_context_window(item)
                max_tok = min(max(ctx // 4, 1024), 131072)
                label = _label(item)
                desc = (item.get("description") or "").strip() or None
                if desc and len(desc) > 2000:
                    desc = desc[:2000]
                feats = _default_features(mtype)
                pricing_json = json.dumps(_pricing_blob(item), ensure_ascii=False)

                r2 = await conn.execute(
                    text(
                        """
                        SELECT id FROM system_model_definitions
                        WHERE provider_id = :pid AND model_name = :mname
                        """
                    ),
                    {"pid": pid, "mname": model_name},
                )
                mid = r2.scalar()

                params = {
                    "provider_id": pid,
                    "model_name": model_name,
                    "label": label[:100],
                    "description": desc,
                    "model_type": mtype,
                    "context_window": ctx,
                    "default_max_tokens": max_tok,
                    "features": json.dumps(feats, ensure_ascii=False),
                    "default_parameters": json.dumps({}, ensure_ascii=False),
                    "pricing": pricing_json,
                    "position": pos,
                    "is_enabled": True,
                }

                if not mid:
                    params["id"] = uuid4()
                    await conn.execute(
                        text(
                            """
                            INSERT INTO system_model_definitions
                            (id, provider_id, model_name, label, description, model_type,
                             context_window, default_max_tokens, features, default_parameters,
                             pricing, position, is_enabled, owner_id, created_at, updated_at)
                            VALUES
                            (:id, :provider_id, :model_name, :label, :description, :model_type,
                             :context_window, :default_max_tokens, CAST(:features AS json),
                             CAST(:default_parameters AS json), CAST(:pricing AS json),
                             :position, :is_enabled, NULL, NOW(), NOW())
                            """
                        ),
                        params,
                    )
                    print(f"  + {model_name} ({mtype})")
                else:
                    params["id"] = mid
                    await conn.execute(
                        text(
                            """
                            UPDATE system_model_definitions
                            SET label = :label,
                                description = :description,
                                model_type = :model_type,
                                context_window = :context_window,
                                default_max_tokens = :default_max_tokens,
                                features = CAST(:features AS json),
                                pricing = CAST(:pricing AS json),
                                position = :position,
                                is_enabled = :is_enabled,
                                updated_at = NOW()
                            WHERE id = :id
                            """
                        ),
                        params,
                    )
                    print(f"  . {model_name} ({mtype})")
    finally:
        await async_db_manager.close()


async def main_async() -> None:
    ap = argparse.ArgumentParser(description="从 GitCC /api/pricing 导入模型到数据库")
    ap.add_argument("--url", default=DEFAULT_PRICING_URL, help="定价 JSON 地址")
    ap.add_argument("--provider", default="gitcc", help="system_model_providers.name")
    ap.add_argument(
        "--all",
        action="store_true",
        help="导入接口中的全部模型（数量多、耗时更长）",
    )
    ap.add_argument(
        "--only",
        type=str,
        default="",
        help="逗号分隔的 model_name 列表；指定时忽略默认基础集",
    )
    args = ap.parse_args()

    print(f"拉取定价数据: {args.url}")
    data = await _fetch_pricing(args.url)
    by_name = {r["model_name"]: r for r in data if r.get("model_name")}

    if args.only.strip():
        names = [x.strip() for x in args.only.split(",") if x.strip()]
        selected = [by_name[n] for n in names if n in by_name]
        missing = [n for n in names if n not in by_name]
        if missing:
            print(f"警告: 以下模型在定价接口中不存在，已跳过: {', '.join(missing)}")
    elif args.all:
        selected = data
    else:
        selected = [by_name[n] for n in sorted(DEFAULT_BASIC_MODEL_NAMES) if n in by_name]
        skipped = DEFAULT_BASIC_MODEL_NAMES - set(by_name.keys())
        if skipped:
            print(f"提示: 下列预设名在接口中不存在，已跳过: {', '.join(sorted(skipped))}")

    if not selected:
        print("没有可导入的模型，退出。")
        return

    print(f"准备写入 {len(selected)} 条到供应商 {args.provider!r} …")
    await _import_rows(selected, provider_name=args.provider)
    print("完成。")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
