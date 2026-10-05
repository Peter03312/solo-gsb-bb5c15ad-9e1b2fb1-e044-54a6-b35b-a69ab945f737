"""质控：测温盲区查询（纯逻辑）。

覆盖：
* 以查询边界与窗内相邻有效测温时刻划分间隔，严格长于阈值才为盲区（相等不算）；
* 窗内无有效测温时整窗按同一规则处理；
* 越温点照样覆盖（覆盖与温度无关）；被取代 / 被隔离的车载点（adopted=False）
  不构成覆盖；交接点不测温；
* 窗内有效点才算锚点，窗外点（即使紧邻边界）不参与；
* 保管方片段按盲区内部交接时刻切分，交接点归接收方，不产生零长度片段；
* max_gap 必须为正时长。
"""
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from app.chain import (
    SOURCE_OWN,
    SOURCE_VEHICLE,
    TimelinePoint,
    measurement_point,
)
from app.coverage import coverage_gaps, custody_slices

UTC = timezone.utc
HOUR = timedelta(hours=1)


def _t(h: float, day: int = 1) -> datetime:
    return datetime(2026, 1, day, tzinfo=UTC) + timedelta(hours=h)


def _own_pt(h: float, temp="2", eid=None) -> TimelinePoint:
    return measurement_point(
        _t(h), temp, source=SOURCE_OWN, source_id=eid or f"own-{h}"
    )


def _veh(h: float, temp="2", rid=None, adopted=True, trip="T1") -> TimelinePoint:
    return TimelinePoint(
        occurred_at=_t(h),
        is_measurement=True,
        temperature=Decimal(temp),
        source=SOURCE_VEHICLE,
        source_id=rid or f"vr-{h}",
        trip_id=trip,
        adopted=adopted,
    )


def _handover(h: float, giver="A", receiver="B") -> TimelinePoint:
    return TimelinePoint(
        occurred_at=_t(h),
        is_measurement=False,
        giver=giver,
        receiver=receiver,
        source=SOURCE_OWN,
        source_id=f"ho-{h}",
    )


def _gaps(points, start=0.0, end=24.0, max_gap=HOUR, holder="A"):
    return coverage_gaps(points, holder, _t(start), _t(end), max_gap)


# ------------------------------------------------------------ 基本间隔


def test_whole_window_gap_without_any_reading():
    gaps = _gaps([], start=2, end=5, max_gap=HOUR)
    assert len(gaps) == 1
    g = gaps[0]
    assert (g.start, g.end) == (_t(2), _t(5))
    assert g.duration_seconds == 3 * 3600
    assert g.start_reading is None and g.end_reading is None
    assert [(s.start, s.end, s.party) for s in g.custody_segments] == [
        (_t(2), _t(5), "A")
    ]


def test_boundaries_and_interior_anchors_split_intervals():
    # 有效点 02:00、04:00；窗 00:00-06:00，阈值 1h
    gaps = _gaps([_own_pt(2), _own_pt(4)], start=0, end=6, max_gap=HOUR)
    assert [(g.start, g.end) for g in gaps] == [
        (_t(0), _t(2)),
        (_t(2), _t(4)),
        (_t(4), _t(6)),
    ]
    # 两端读数来源
    assert gaps[0].start_reading is None
    assert gaps[0].end_reading.source == SOURCE_OWN
    assert gaps[0].end_reading.event_id == "own-2"
    assert gaps[1].start_reading.event_id == "own-2"
    assert gaps[1].end_reading.event_id == "own-4"
    assert gaps[2].start_reading.event_id == "own-4"
    assert gaps[2].end_reading is None


def test_strictly_longer_than_threshold():
    # 02:00 一个点；窗 00:00-04:00
    # 阈值 2h：两段恰好 2h，都不是盲区
    assert _gaps([_own_pt(2)], start=0, end=4, max_gap=2 * HOUR) == []
    # 阈值 2h - 1s：两段都严格更长
    gaps = _gaps([_own_pt(2)], start=0, end=4, max_gap=2 * HOUR - timedelta(seconds=1))
    assert len(gaps) == 2
    assert gaps[0].duration_seconds == 7200.0


def test_dense_readings_no_gap():
    points = [_own_pt(h) for h in range(0, 7)]  # 整点全覆盖
    assert _gaps(points, start=0, end=6, max_gap=HOUR) == []


def test_reading_at_window_end_excluded_but_start_included():
    # 左闭右开：恰在查询起点的点算锚点；恰在终点的点不算
    gaps = _gaps([_own_pt(0), _own_pt(6)], start=0, end=6, max_gap=HOUR)
    assert len(gaps) == 1
    assert (gaps[0].start, gaps[0].end) == (_t(0), _t(6))
    assert gaps[0].start_reading is not None
    assert gaps[0].end_reading is None  # 06:00 在窗外（开区间）


def test_outside_window_readings_are_not_anchors():
    # 点在窗外：00:00 之前一点、06:00 之后一点，都不缩短窗内间隔
    points = [_own_pt(-0.5), _own_pt(6.5)]
    gaps = _gaps(points, start=0, end=6, max_gap=HOUR)
    assert len(gaps) == 1
    assert (gaps[0].start, gaps[0].end) == (_t(0), _t(6))
    assert gaps[0].start_reading is None and gaps[0].end_reading is None


# ------------------------------------------------------------ 有效点口径


def test_over_limit_reading_still_covers():
    # 9℃（超过 4℃ 上限语义无关）仍是有效覆盖点
    gaps = _gaps([_own_pt(2, temp="9"), _own_pt(4, temp="9")], start=0, end=6, max_gap=HOUR)
    assert [(g.start, g.end) for g in gaps] == [
        (_t(0), _t(2)),
        (_t(2), _t(4)),
        (_t(4), _t(6)),
    ]


def test_supplanted_and_quarantined_vehicle_points_do_not_cover():
    points = [
        _veh(2, adopted=False, rid="vr-x"),   # 被同刻自有取代 / 被隔离
        _veh(3, adopted=True, rid="vr-y"),
        _veh(5, adopted=False, rid="vr-z"),
    ]
    gaps = _gaps(points, start=0, end=6, max_gap=HOUR)
    # 唯一有效点 03:00 -> 00-03 与 03-06 两段盲区
    assert [(g.start, g.end) for g in gaps] == [(_t(0), _t(3)), (_t(3), _t(6))]
    assert gaps[0].end_reading.reading_id == "vr-y"
    assert gaps[1].start_reading.reading_id == "vr-y"
    assert gaps[0].end_reading.trip_id == "T1"


def test_handover_does_not_cover_but_is_passed_for_custody():
    points = [_handover(3)]
    gaps = _gaps(points, start=0, end=6, max_gap=HOUR)
    assert len(gaps) == 1
    assert (gaps[0].start, gaps[0].end) == (_t(0), _t(6))


def test_vehicle_anchor_endpoints_report_vehicle_source():
    gaps = _gaps([_veh(3, rid="vr-3")], start=0, end=6, max_gap=HOUR)
    assert gaps[0].end_reading.source == SOURCE_VEHICLE
    assert gaps[0].end_reading.reading_id == "vr-3"
    assert gaps[0].end_reading.event_id is None
    assert gaps[1].start_reading.source == SOURCE_VEHICLE


# ------------------------------------------------------------ 保管方片段


def test_custody_segments_split_by_internal_handover():
    points = [_own_pt(2), _handover(4, "A", "B"), _own_pt(6)]
    gaps = _gaps(points, start=0, end=8, max_gap=HOUR)
    middle = gaps[1]  # 02:00-06:00 跨交接
    assert [(s.start, s.end, s.party) for s in middle.custody_segments] == [
        (_t(2), _t(4), "A"),
        (_t(4), _t(6), "B"),   # 交接时刻 04:00 起归接收方
    ]
    # 06:00-08:00 单段归 B
    assert [(s.start, s.end, s.party) for s in gaps[2].custody_segments] == [
        (_t(6), _t(8), "B")
    ]


def test_handover_exactly_at_gap_start_goes_to_receiver():
    # 盲区 04:00-06:00，交接恰在 04:00：起点已归接收方 B
    points = [_handover(4, "A", "B"), _own_pt(6)]
    gaps = _gaps(points, start=4, end=6, max_gap=HOUR)
    assert len(gaps) == 1
    assert [(s.start, s.end, s.party) for s in gaps[0].custody_segments] == [
        (_t(4), _t(6), "B")
    ]


def test_handover_exactly_at_gap_end_no_zero_length_segment():
    # 盲区 02:00-04:00，交接恰在 04:00：交出方片段零长，不产生
    points = [_own_pt(2), _handover(4, "A", "B"), _own_pt(6)]
    gaps = _gaps(points, start=2, end=4, max_gap=HOUR)
    assert len(gaps) == 1
    assert [(s.start, s.end, s.party) for s in gaps[0].custody_segments] == [
        (_t(2), _t(4), "A")
    ]


def test_multiple_handovers_inside_gap():
    points = [
        _own_pt(0),
        _handover(3, "A", "B"),
        _handover(5, "B", "C"),
        _own_pt(8),
    ]
    gaps = _gaps(points, start=0, end=8, max_gap=2 * HOUR)
    # 唯一盲区 00:00-08:00（阈值 2h）
    assert len(gaps) == 1
    assert [(s.start, s.end, s.party) for s in gaps[0].custody_segments] == [
        (_t(0), _t(3), "A"),
        (_t(3), _t(5), "B"),
        (_t(5), _t(8), "C"),
    ]


def test_custody_uses_holder_chain_from_before_window():
    # 窗外交接 A->B 在 -1h，窗内盲区的保管方应已是 B
    points = [_handover(-1, "A", "B")]
    slices = custody_slices(points, "A", _t(0), _t(3))
    assert [(s.start, s.end, s.party) for s in slices] == [(_t(0), _t(3), "B")]


def test_non_positive_max_gap_rejected():
    with pytest.raises(ValueError):
        coverage_gaps([], "A", _t(0), _t(1), timedelta(0))
    with pytest.raises(ValueError):
        coverage_gaps([], "A", _t(0), _t(1), timedelta(seconds=-1))


# ------------------------------------------------------------ API 集成


def _query(client, cargo="C1", start="2026-01-01T00:00:00+00:00",
           end="2026-01-02T00:00:00+00:00", max_gap="3600"):
    params = {"start_at": start, "end_at": end}
    if max_gap is not None:
        params["max_gap_seconds"] = max_gap
    return client.get(f"/cargo/{cargo}/coverage-gaps", params=params)


def _setup(client):
    r = client.post("/cargo", json={
        "cargo_id": "C1",
        "started_at": "2025-12-31T00:00:00+00:00",
        "temp_upper_limit": "4",
        "initial_holder": "A",
    })
    assert r.status_code == 201, r.text
    assert client.post("/trips", json={"trip_id": "T1", "vehicle_no": "v"}).status_code == 201


def _manifest(client, loaded, unloaded, cargo="C1", trip="T1"):
    r = client.post(f"/trips/{trip}/manifest", json={"items": [{
        "cargo_id": cargo,
        "loaded_at": loaded,
        "unloaded_at": unloaded,
    }]})
    assert r.status_code == 201, r.text


def _vr(client, rid, at, temp="2", trip="T1"):
    r = client.post("/vehicle-readings", json={
        "reading_id": rid, "trip_id": trip, "read_at": at,
        "temperature": str(temp),
    })
    assert r.status_code in (200, 201), r.text


def _own(client, eid, at, temp="2", cargo="C1"):
    r = client.post("/events", json={
        "event_id": eid, "cargo_id": cargo, "occurred_at": at,
        "temperature": str(temp),
    })
    assert r.status_code == 201, r.text


def _ho(client, eid, at, giver, receiver, cargo="C1"):
    r = client.post("/events", json={
        "event_id": eid, "cargo_id": cargo, "occurred_at": at,
        "giver": giver, "receiver": receiver,
    })
    assert r.status_code == 201, r.text


def test_api_full_custody_split_and_sources(client):
    _setup(client)
    _manifest(client, "2026-01-01T00:00:00+00:00", "2026-01-01T12:00:00+00:00")
    _vr(client, "r2", "2026-01-01T02:00:00+00:00")
    _vr(client, "r10", "2026-01-01T10:00:00+00:00")
    _ho(client, "h6", "2026-01-01T06:00:00+00:00", "A", "B")

    data = _query(
        client,
        start="2026-01-01T00:00:00+00:00",
        end="2026-01-01T12:00:00+00:00",
        max_gap="3600",
    ).json()
    assert data["max_gap_seconds"] == 3600.0
    gaps = data["gaps"]
    assert [g["start"][11:16] + "-" + g["end"][11:16] for g in gaps] == [
        "00:00-02:00", "02:00-10:00", "10:00-12:00",
    ]
    middle = gaps[1]
    assert middle["start_reading"]["reading_id"] == "r2"
    assert middle["end_reading"]["reading_id"] == "r10"
    assert [(s["start"][11:16], s["end"][11:16], s["responsible_party"])
            for s in middle["custody_segments"]] == [
        ("02:00", "06:00", "A"),
        ("06:00", "10:00", "B"),
    ]


def test_api_over_limit_reading_covers(client):
    _setup(client)
    _own(client, "e1", "2026-01-01T02:00:00+00:00", temp="9")  # 远超上限 4
    data = _query(
        client, start="2026-01-01T01:00:00+00:00",
        end="2026-01-01T03:00:00+00:00", max_gap="1800",
    ).json()
    assert [g["start"][11:16] + "-" + g["end"][11:16] for g in data["gaps"]] == [
        "01:00-02:00", "02:00-03:00",
    ]


def test_api_vehicle_applicability_half_open_loading(client):
    _setup(client)
    _manifest(client, "2026-01-01T04:00:00+00:00", "2026-01-01T08:00:00+00:00")
    _vr(client, "r-lo", "2026-01-01T04:00:00+00:00")   # 装车时刻：在车
    _vr(client, "r-un", "2026-01-01T08:00:00+00:00")   # 卸货时刻：已下车
    data = _query(
        client, start="2026-01-01T00:00:00+00:00",
        end="2026-01-01T12:00:00+00:00", max_gap="3600",
    ).json()
    assert [g["start"][11:16] + "-" + g["end"][11:16] for g in data["gaps"]] == [
        "00:00-04:00", "04:00-12:00",
    ]
    assert data["gaps"][0]["end_reading"]["reading_id"] == "r-lo"


def test_api_same_time_own_supersedes_vehicle(client):
    _setup(client)
    _manifest(client, "2026-01-01T00:00:00+00:00", "2026-01-01T12:00:00+00:00")
    _vr(client, "r2", "2026-01-01T02:00:00+00:00", temp="9")
    _own(client, "e2", "2026-01-01T02:00:00+00:00", temp="1")  # 同刻自有优先
    data = _query(
        client, start="2026-01-01T00:00:00+00:00",
        end="2026-01-01T04:00:00+00:00", max_gap="3600",
    ).json()
    assert len(data["gaps"]) == 2
    assert data["gaps"][0]["end_reading"]["source"] == "own"
    assert data["gaps"][0]["end_reading"]["event_id"] == "e2"
    assert data["gaps"][1]["start_reading"]["source"] == "own"


def test_api_probe_fault_quarantine_and_revocation(client):
    _setup(client)
    _manifest(client, "2026-01-01T00:00:00+00:00", "2026-01-01T12:00:00+00:00")
    _vr(client, "r4", "2026-01-01T04:00:00+00:00")
    _vr(client, "r6", "2026-01-01T06:00:00+00:00")
    _vr(client, "r8", "2026-01-01T08:00:00+00:00")
    # 故障覆盖 [05:30, 06:30) -> r6 被隔离，只剩 r4 / r8 两个有效点
    r = client.post("/trips/T1/probe-faults", json={
        "fault_id": "f1",
        "fault_start": "2026-01-01T05:30:00+00:00",
        "fault_end": "2026-01-01T06:30:00+00:00",
        "reason": "探头失准",
    })
    assert r.status_code == 201, r.text
    data = _query(
        client, start="2026-01-01T00:00:00+00:00",
        end="2026-01-01T12:00:00+00:00", max_gap="3600",
    ).json()
    assert [g["start"][11:16] + "-" + g["end"][11:16] for g in data["gaps"]] == [
        "00:00-04:00", "04:00-08:00", "08:00-12:00",
    ]
    # 撤销故障 -> r6 恢复，04:00-08:00 被切成两段 2h
    r = client.post("/trips/T1/probe-faults/revoke", json={
        "revocation_id": "rv1", "fault_id": "f1", "expected_version": 0,
    })
    assert r.status_code in (200, 201), r.text
    data = _query(
        client, start="2026-01-01T00:00:00+00:00",
        end="2026-01-01T12:00:00+00:00", max_gap="3600",
    ).json()
    assert [g["start"][11:16] + "-" + g["end"][11:16] for g in data["gaps"]] == [
        "00:00-04:00", "04:00-06:00", "06:00-08:00", "08:00-12:00",
    ]


def test_api_late_reading_recomputes(client):
    _setup(client)
    _manifest(client, "2026-01-01T00:00:00+00:00", "2026-01-01T12:00:00+00:00")
    data = _query(
        client, start="2026-01-01T00:00:00+00:00",
        end="2026-01-01T12:00:00+00:00", max_gap="3600",
    ).json()
    assert len(data["gaps"]) == 1  # 整窗无读数
    _vr(client, "r-late", "2026-01-01T03:00:00+00:00")  # 晚到补录
    data = _query(
        client, start="2026-01-01T00:00:00+00:00",
        end="2026-01-01T12:00:00+00:00", max_gap="3600",
    ).json()
    assert [g["start"][11:16] + "-" + g["end"][11:16] for g in data["gaps"]] == [
        "00:00-03:00", "03:00-12:00",
    ]


def test_api_loading_correction_recomputes(client):
    _setup(client)
    _manifest(client, "2026-01-01T00:00:00+00:00", "2026-01-01T05:00:00+00:00")
    _vr(client, "r7", "2026-01-01T07:00:00+00:00")  # 原窗口外，不适用
    data = _query(
        client, start="2026-01-01T00:00:00+00:00",
        end="2026-01-01T12:00:00+00:00", max_gap="3600",
    ).json()
    assert [g["start"][11:16] + "-" + g["end"][11:16] for g in data["gaps"]] == [
        "00:00-12:00",
    ]
    r = client.post("/trips/T1/loading-corrections", json={
        "correction_id": "lc1",
        "cargo_id": "C1",
        "origin_loaded_at": "2026-01-01T00:00:00+00:00",
        "expected_version": 0,
        "new_loaded_at": "2026-01-01T06:00:00+00:00",
        "new_unloaded_at": "2026-01-01T09:00:00+00:00",
        "reason": "窗口录错",
    })
    assert r.status_code == 201, r.text
    data = _query(
        client, start="2026-01-01T00:00:00+00:00",
        end="2026-01-01T12:00:00+00:00", max_gap="3600",
    ).json()
    assert [g["start"][11:16] + "-" + g["end"][11:16] for g in data["gaps"]] == [
        "00:00-07:00", "07:00-12:00",
    ]
    assert data["gaps"][0]["end_reading"]["reading_id"] == "r7"


def test_api_temperature_correction_does_not_change_coverage(client):
    _setup(client)
    _manifest(client, "2026-01-01T00:00:00+00:00", "2026-01-01T12:00:00+00:00")
    _vr(client, "r2", "2026-01-01T02:00:00+00:00", temp="9")
    before = _query(
        client, start="2026-01-01T00:00:00+00:00",
        end="2026-01-01T12:00:00+00:00", max_gap="3600",
    ).json()
    r = client.post("/corrections", json={
        "correction_id": "c1",
        "target_type": "vehicle_reading",
        "target_id": "r2",
        "expected_version": 0,
        "new_temperature": "-20",
    })
    assert r.status_code == 201, r.text
    after = _query(
        client, start="2026-01-01T00:00:00+00:00",
        end="2026-01-01T12:00:00+00:00", max_gap="3600",
    ).json()
    # 覆盖几何完全不变（端点仍是 r2，只是盲区响应不含温度字段）
    assert [
        (g["start"], g["end"],
         (g["start_reading"] or {}).get("reading_id"),
         (g["end_reading"] or {}).get("reading_id"))
        for g in before["gaps"]
    ] == [
        (g["start"], g["end"],
         (g["start_reading"] or {}).get("reading_id"),
         (g["end_reading"] or {}).get("reading_id"))
        for g in after["gaps"]
    ]


def test_api_no_gap_when_dense(client):
    _setup(client)
    for h in range(0, 6):
        _own(client, f"e{h}", f"2026-01-01T{h:02d}:00:00+00:00")
    data = _query(
        client, start="2026-01-01T00:00:00+00:00",
        end="2026-01-01T05:00:00+00:00", max_gap="3600",
    ).json()
    assert data["gaps"] == []


def test_api_errors(client):
    _setup(client)
    base = {
        "start_at": "2026-01-01T00:00:00+00:00",
        "end_at": "2026-01-01T03:00:00+00:00",
    }
    # 货物不存在
    r = client.get("/cargo/NOPE/coverage-gaps",
                   params={**base, "max_gap_seconds": "60"})
    assert r.status_code == 404 and r.json()["error"]["code"] == "cargo_not_found"
    # 终点不晚于起点
    r = client.get("/cargo/C1/coverage-gaps", params={
        "start_at": "2026-01-01T03:00:00+00:00",
        "end_at": "2026-01-01T00:00:00+00:00",
        "max_gap_seconds": "60",
    })
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_query_window"
    # 起止相等同样拒绝
    r = client.get("/cargo/C1/coverage-gaps", params={
        "start_at": "2026-01-01T00:00:00+00:00",
        "end_at": "2026-01-01T00:00:00+00:00",
        "max_gap_seconds": "60",
    })
    assert r.status_code == 422
    # 缺时区
    r = client.get("/cargo/C1/coverage-gaps", params={
        "start_at": "2026-01-01T00:00:00",
        "end_at": "2026-01-01T03:00:00+00:00",
        "max_gap_seconds": "60",
    })
    assert r.status_code == 422 and r.json()["error"]["code"] == "timezone_required"
    # 阈值非正 / 非法 / 缺失
    for val, code in [
        ("0", "invalid_max_gap"),
        ("-1", "invalid_max_gap"),
        ("abc", "invalid_max_gap"),
    ]:
        r = client.get("/cargo/C1/coverage-gaps",
                       params={**base, "max_gap_seconds": val})
        assert r.status_code == 422 and r.json()["error"]["code"] == code, (val, r.text)
    r = client.get("/cargo/C1/coverage-gaps", params=base)
    assert r.status_code == 422 and r.json()["error"]["code"] == "missing_parameter"
    # 缺起止
    r = client.get("/cargo/C1/coverage-gaps", params={"max_gap_seconds": "60"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "missing_parameter"


def test_api_fractional_max_gap_seconds(client):
    _setup(client)
    # 窗内不放任何读数：整窗 10 秒
    r = _query(
        client, start="2026-01-01T00:00:00+00:00",
        end="2026-01-01T00:00:10+00:00", max_gap="10",
    )
    assert r.status_code == 200 and r.json()["gaps"] == []  # 恰好相等
    r = _query(
        client, start="2026-01-01T00:00:00+00:00",
        end="2026-01-01T00:00:10+00:00", max_gap="9.999",
    )
    assert len(r.json()["gaps"]) == 1
    assert r.json()["gaps"][0]["duration_seconds"] == 10.0
