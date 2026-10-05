"""质控测温盲区查询：在一致快照上加载货物上下文并识别盲区。

只读查询，整次查询基于一个**一致的已提交数据视图**：PostgreSQL 下以
REPEATABLE READ 快照事务执行（SQLite 单连接读本身串行，无需提升级别）。
晚到读数 / 装卸窗口更正 / 探头故障撤销在本次查询事务开始前已提交即按
**当前状态**反映，提交后才能被下一次查询看到；查询本身不改任何记录。
"""
from __future__ import annotations

from datetime import timedelta

from app import coverage, timeline
from app.crud import ApiError
from app.models import Cargo
from app.timeutil import ensure_utc


def query_coverage_gaps(db, cargo_id: str, start_at, end_at, max_gap: timedelta):
    """查询一票货物在查询窗内的测温盲区（含保管方片段）。

    :param max_gap: 最大允许无有效测温时长（正时长；调用方负责解析与校验）。
    :returns: ``(cargo, gaps)``——货物行（归一化到 UTC）与
        :class:`app.coverage.CoverageGap` 列表。
    """
    if end_at <= start_at:
        raise ApiError(
            422,
            "invalid_query_window",
            f"查询窗非法：终点 {end_at.isoformat()} 必须严格晚于起点 "
            f"{start_at.isoformat()}",
        )
    if max_gap <= timedelta(0):
        raise ApiError(
            422,
            "invalid_max_gap",
            "最大允许无有效测温时长 max_gap_seconds 必须为正数",
        )

    # 一致快照：在该 Session 的第一条 SQL 前指定事务隔离级别。
    bind = db.get_bind()
    if bind.dialect.name == "postgresql":
        db.connection(execution_options={"isolation_level": "REPEATABLE READ"})

    # load_cargo_context 对未知货物抛 404 cargo_not_found
    events, loadings, readings, effective, quarantine, standards = (
        timeline.load_cargo_context(db, cargo_id)
    )
    cargo = db.get(Cargo, cargo_id)
    cargo.started_at = ensure_utc(cargo.started_at)

    # 完整合并时间线（含交接点与被取代 / 被隔离的车载点）：
    # 有效测温=adopted 的测温点；交接点供保管方片段拆分；温度是否越限无关。
    points = timeline.merged_timeline(
        events,
        readings,
        loadings,
        effective[0],
        effective[1],
        quarantine,
        standards,
    )
    gaps = coverage.coverage_gaps(
        points,
        cargo.initial_holder,
        start_at,
        end_at,
        max_gap,
    )
    return cargo, gaps
