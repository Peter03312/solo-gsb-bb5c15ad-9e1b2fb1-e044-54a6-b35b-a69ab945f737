"""质控：测温盲区（测温覆盖间隔）识别与保管方片段拆分（纯逻辑）。

盲区口径
--------
质控人员按货物编号、带时区查询窗 ``[start_at, end_at)`` 与**最大允许无有效
测温时长** ``max_gap`` 查看测温盲区。有效测温点取自货物的合并读数时间线
（见 app/timeline.py，:class:`app.chain.TimelinePoint`）：

* 货物**自有测温**恒为有效点（与是否在车无关）；
* 车载读数仅当读数时刻落在该货物**当前装卸窗口**（左闭右开
  ``[loaded_at, unloaded_at)``）、同一车次，且**未被生效探头故障隔离**时
  才出现在时间线上；同刻存在自有测温时该车载点 ``adopted=False``（被取代）。

故“有效测温点” = 时间线上 ``is_measurement and adopted`` 的点；温度是否
越限与覆盖无关（被采用的越温点照样覆盖）。

以**查询边界**与窗内（``start_at <= t < end_at``）相邻有效测温时刻划分
间隔：边界序列为 ``[start_at] + 窗内有效测温时刻 + [end_at]``，逐对相邻
时刻 ``(a, b)`` 判定，**严格长于**阈值（``b - a > max_gap``）的间隔列为
盲区；窗内无任何有效测温时，整窗 ``[start_at, end_at)`` 按同一规则处理。
窗外读数（即使紧邻边界）不参与，盲区端点读数来源在边界来自查询窗时为空。

保管方片段
----------
每个盲区按查询时刻落在盲区内的**交接事件**（自有交接，``a < h < b``）切分
保管方片段：``[片段起, 交接时刻)`` 归交出方，**交接时刻起归接收方**——
盲区起点恰为交接时刻时，该点已归接收方（与责任拆分“交接点归接收方”一致）；
交接恰在盲区终点时不产生交出方片段。不产生零长度片段。

晚到读数 / 装卸窗口更正 / 故障撤销均在查询时按当前状态重建（本模块只处理
重建后的时间线，不感知存储）；测温值更正只改温度、不改变点的时刻与
``adopted``，故不影响覆盖。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from app.chain import SOURCE_OWN, SOURCE_VEHICLE, TimelinePoint


@dataclass(frozen=True)
class CoverageReading:
    """盲区端点处的有效测温读数来源。"""

    occurred_at: object
    source: str                  # own=货物自有测温；vehicle=车载读数
    source_id: str | None = None  # own=event_id；vehicle=reading_id
    trip_id: str | None = None    # 仅车载点有值

    @property
    def event_id(self) -> str | None:
        return self.source_id if self.source == SOURCE_OWN else None

    @property
    def reading_id(self) -> str | None:
        return self.source_id if self.source == SOURCE_VEHICLE else None


@dataclass(frozen=True)
class CustodySlice:
    """盲区 [start, end) 内的一段保管方区间（左闭右开，正长度）。"""

    start: object
    end: object
    party: str


@dataclass(frozen=True)
class CoverageGap:
    """一段测温盲区：[start, end) 无有效测温，且时长严格大于阈值。"""

    start: object
    end: object
    duration_seconds: float
    # 造成该间隔的两端有效读数（间隔 (start,end) 的外侧锚点）；
    # 端点来自查询边界（窗外无锚点）时为 None
    start_reading: CoverageReading | None
    end_reading: CoverageReading | None
    custody_segments: tuple[CustodySlice, ...]


def _is_covered_point(p: TimelinePoint) -> bool:
    """该时间线点是否构成一次有效测温覆盖。

    交接点不测温；车载点在被同刻自有测温取代（``adopted=False``）或被生效
    探头故障隔离（``adopted=False``）时不构成覆盖；是否越温不影响。
    """
    return p.is_measurement and p.adopted


def _reading_of(p: TimelinePoint) -> CoverageReading:
    return CoverageReading(
        occurred_at=p.occurred_at,
        source=p.source,
        source_id=p.source_id,
        trip_id=p.trip_id,
    )


def holder_after(points, initial_holder: str, at) -> str:
    """at 时刻（该时刻的交接已生效后）的保管方。

    只沿自有交接点推进（测温 / 车载点不改变保管方）；用于确定盲区起点的
    保管方，故交接恰在 at 时取接收方（交接点归接收方）。
    """
    holder = initial_holder
    for p in points:
        if not p.is_measurement and p.occurred_at <= at:
            holder = p.receiver
    return holder


def custody_slices(
    points,
    initial_holder: str,
    gap_start,
    gap_end,
) -> tuple[CustodySlice, ...]:
    """把一个盲区 [gap_start, gap_end) 按内部交接时刻切成保管方正长度片段。

    仅统计严格位于盲区内部的交接（``gap_start < h < gap_end``）：交接在
    盲区起点时，起点已归接收方（经 :func:`holder_after` 计入）；交接在
    盲区终点时，交出方片段为零长、不产生。
    """
    internal_handovers = sorted(
        (
            p
            for p in points
            if not p.is_measurement and gap_start < p.occurred_at < gap_end
        ),
        key=lambda p: (p.occurred_at, p.source_id or ""),
    )

    slices: list[CustodySlice] = []
    current = gap_start
    party = holder_after(points, initial_holder, gap_start)
    for h in internal_handovers:
        if current < h.occurred_at:
            slices.append(CustodySlice(current, h.occurred_at, party))
        current = h.occurred_at
        party = h.receiver
    if current < gap_end:
        slices.append(CustodySlice(current, gap_end, party))
    return tuple(slices)


def coverage_gaps(
    points,
    initial_holder: str,
    start_at,
    end_at,
    max_gap: timedelta,
) -> list[CoverageGap]:
    """在合并读数时间线上识别查询窗内的测温盲区（按时间升序）。

    :param points: 货物的**完整合并时间线**
        （:class:`app.chain.TimelinePoint`，含交接点；窗外点也要给全——
        窗外读数不参与锚点，但窗外 / 窗内交接参与保管方判定）。
    :param initial_holder: 货物起始保管方。
    :param start_at / end_at: 查询窗（左闭右开，调用方保证 start < end）。
    :param max_gap: 最大允许无有效测温时长（正的 timedelta；间隔严格长于
        该值才列为盲区，相等不算）。
    """
    if max_gap <= timedelta(0):
        raise ValueError("max_gap 必须为正时长")

    ordered = sorted(
        points,
        key=lambda p: (
            p.occurred_at,
            0 if p.is_measurement else 1,
            {SOURCE_OWN: 0, SOURCE_VEHICLE: 1}.get(p.source, 9),
            p.source_id or "",
        ),
    )
    # 仅窗内 [start_at, end_at) 的有效测温点充当间隔锚点
    anchors = [
        p
        for p in ordered
        if _is_covered_point(p) and start_at <= p.occurred_at < end_at
    ]

    boundaries: list = [start_at]
    readings: list[CoverageReading | None] = [None]
    for p in anchors:
        boundaries.append(p.occurred_at)
        readings.append(_reading_of(p))
    boundaries.append(end_at)
    readings.append(None)

    gaps: list[CoverageGap] = []
    for i in range(len(boundaries) - 1):
        a, b = boundaries[i], boundaries[i + 1]
        duration = b - a
        if duration <= max_gap:
            continue
        gaps.append(
            CoverageGap(
                start=a,
                end=b,
                duration_seconds=duration.total_seconds(),
                start_reading=readings[i],
                end_reading=readings[i + 1],
                custody_segments=custody_slices(
                    ordered, initial_holder, a, b
                ),
            )
        )
    return gaps
