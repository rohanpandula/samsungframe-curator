"""Tests for the Immich source connector + feedback sink (M005/S05).

Deterministic and air-gapped: the synthetic transport is fully in-memory. Covers
transport probe/browse/filter/pagination, SHA-256 download verification,
checkpointed + idempotent + isolated sync against the catalog, availability
tombstones (never deleted), and the disabled-by-default feedback sink that only
writes confirmed favorites and never deletes.
"""

from __future__ import annotations

from curator.catalog import Catalog
from curator.connectors import (
    ImmichConnector,
    ImmichFeedbackSink,
    ImmichQuery,
    SyntheticImmichTransport,
)
from curator.errors import ConnectorError
from curator.hashing import sha256_hex

ASSETS = [
    {"asset_id": "a-1", "data": b"alpha-bytes", "favorite": True, "star": 5.0,
     "tags": ("vacation", "bali"), "date": "2024-06-01"},
    {"asset_id": "b-2", "data": b"bravo-bytes!", "favorite": False, "star": 3.0,
     "tags": ("night",), "date": "2024-02-15"},
    {"asset_id": "c-3", "data": b"charlie-bytes", "favorite": True, "star": 4.0,
     "tags": ("vacation",), "date": "2024-09-20"},
]


def _transport(page_size: int = 100) -> SyntheticImmichTransport:
    return SyntheticImmichTransport(list(ASSETS), page_size=page_size)


class _NoFeedback(SyntheticImmichTransport):
    """Synthetic transport whose server does not advertise feedback."""

    def probe(self) -> dict:
        d = super().probe()
        d["capabilities"]["feedback"] = False
        return d


# -- transport: probe / browse / filter / pagination ---------------------------

def test_probe_reports_version_and_capabilities():
    probe = _transport().probe()
    assert "1.0.0-synthetic" == probe["version"]
    caps = probe["capabilities"]
    assert set(caps["filters"]) == {"favorite", "star", "tags", "date"}
    assert caps["smart_search"] is True
    assert caps["feedback"] is True
    assert caps["feedback_scope"] == "user:write"


def test_browse_returns_all_assets_sorted_no_filter():
    transport = _transport()
    assets, next_cursor = transport.browse(ImmichQuery())
    assert [a.asset_id for a in assets] == ["a-1", "b-2", "c-3"]
    assert next_cursor is None  # single page fit


def test_browse_filter_favorite():
    transport = _transport()
    ids = [a.asset_id for a in transport.browse(ImmichQuery(favorite=True))[0]]
    assert ids == ["a-1", "c-3"]


def test_browse_filter_star_min():
    transport = _transport()
    ids = [a.asset_id for a in transport.browse(ImmichQuery(star_min=4.0))[0]]
    assert ids == ["a-1", "c-3"]


def test_browse_filter_tags_required_subset():
    transport = _transport()
    ids = [a.asset_id for a in transport.browse(ImmichQuery(tags=("vacation",)))[0]]
    assert ids == ["a-1", "c-3"]


def test_browse_filter_date_range():
    transport = _transport()
    query = ImmichQuery(date_from="2024-05-01", date_to="2024-12-31")
    ids = [a.asset_id for a in transport.browse(query)[0]]
    assert ids == ["a-1", "c-3"]


def test_browse_pagination_cursor_returns_non_overlapping_remaining():
    transport = _transport(page_size=2)
    page1, next_cursor = transport.browse(ImmichQuery())
    assert [a.asset_id for a in page1] == ["a-1", "b-2"]
    assert next_cursor == "b-2"
    page2, next_cursor2 = transport.browse(ImmichQuery(), cursor=next_cursor)
    assert [a.asset_id for a in page2] == ["c-3"]
    assert next_cursor2 is None


def test_browse_cursor_filters_non_overlapping():
    transport = _transport(page_size=2)
    _, next_cursor = transport.browse(ImmichQuery())
    remaining = transport.browse(ImmichQuery(), cursor=next_cursor)[0]
    assert [a.asset_id for a in remaining] == ["c-3"]


def test_smart_search_matches_tags():
    transport = _transport()
    ids = [a.asset_id for a in transport.search(ImmichQuery(text="vacation"))]
    assert ids == ["a-1", "c-3"]


def test_smart_search_disabled_raises():
    transport = _transport()
    transport.smart_search = False
    try:
        transport.search(ImmichQuery(text="vacation"))
        assert False, "expected ConnectorError"
    except ConnectorError:
        pass


# -- download + SHA-256 verification --------------------------------------------

def test_read_original_matches_recorded_checksum():
    connector = ImmichConnector("svc:immich", _transport())
    for wanted in (b"alpha-bytes", b"bravo-bytes!", b"charlie-bytes"):
        asset_id = {b"alpha-bytes": "a-1", b"bravo-bytes!": "b-2", b"charlie-bytes": "c-3"}[wanted]
        data = connector.read_original(asset_id)
        assert data == wanted
        assert sha256_hex(data) == connector.transport.get_checksum(asset_id)


def test_read_original_tombstoned_raises():
    transport = _transport()
    connector = ImmichConnector("svc:immich", transport)
    transport.mark_unavailable("b-2")
    try:
        connector.read_original("b-2")
        assert False, "expected ConnectorError"
    except ConnectorError:
        pass


# -- connector: capability + enumeration ----------------------------------------

def test_connector_health_and_capabilities():
    connector = ImmichConnector("svc:immich", _transport())
    assert connector.health().healthy is True
    assert connector.capabilities.original_stream is True
    assert connector.capabilities.cursor_pagination is True
    assert connector.capabilities.revision_support is True
    assert connector.filters == ("favorite", "star", "tags", "date")
    assert connector.smart_search is True
    assert connector.feedback_supported is True


def test_connector_enumerate_deterministic_and_scoped():
    connector = ImmichConnector("svc:immich", _transport())
    metas = {m.asset_id: m for m in connector.enumerate()}
    assert set(metas) == {"a-1", "b-2", "c-3"}
    assert all(m.connector_id == "svc:immich" for m in metas.values())
    assert all(m.available for m in metas.values())
    assert metas["a-1"].extra["favorite"] is True


def test_connector_enumerate_cursor_resumes_non_overlapping():
    connector = ImmichConnector("svc:immich", _transport())
    all_ids = [m.asset_id for m in connector.enumerate()]
    resumed = [m.asset_id for m in connector.enumerate(all_ids[0])]
    assert resumed == all_ids[1:]


# -- sync: checkpointed + idempotent + isolated ---------------------------------

def test_sync_persists_state_and_cursor(data_root):
    catalog = Catalog(data_root=data_root)
    connector = ImmichConnector("svc:immich", _transport(), catalog=catalog)
    report = connector.sync()
    assert report.enumerated == 3
    assert report.written == 3
    assert report.cursor == "c-3"
    assert catalog.db.execute("SELECT COUNT(*) FROM immich_asset_state").fetchone()[0] == 3
    assert catalog.db.execute(
        "SELECT cursor FROM immich_sync_state WHERE connector_id='svc:immich'"
    ).fetchone()[0] == "c-3"
    catalog.db.close()


def test_sync_second_run_is_idempotent_no_duplicate_work(data_root):
    catalog = Catalog(data_root=data_root)
    connector = ImmichConnector("svc:immich", _transport(), catalog=catalog)
    connector.sync()
    report2 = connector.sync()
    assert report2.enumerated == 0
    assert report2.written == 0
    assert report2.tombstoned == 0
    assert catalog.db.execute("SELECT COUNT(*) FROM immich_asset_state").fetchone()[0] == 3
    catalog.db.close()


def test_sync_resumes_at_cursor_only_picks_up_new_asset(data_root):
    catalog = Catalog(data_root=data_root)
    transport = _transport()
    connector = ImmichConnector("svc:immich", transport, catalog=catalog)
    connector.sync()
    transport.add("d-4", b"delta-bytes", favorite=False, tags=("city",), date="2025-01-05")
    report2 = connector.sync()
    assert report2.enumerated == 1
    assert report2.written == 1
    # state rows: 3 original + 1 new = 4, none duplicated
    assert catalog.db.execute("SELECT COUNT(*) FROM immich_asset_state").fetchone()[0] == 4
    catalog.db.close()


def test_sync_isolated_per_connector_instance(data_root):
    catalog = Catalog(data_root=data_root)
    transport = _transport()
    conn_a = ImmichConnector("svc:immich-a", transport, catalog=catalog)
    conn_b = ImmichConnector("svc:immich-b", transport, catalog=catalog)
    conn_a.sync()
    conn_b.sync()
    a_count = catalog.db.execute(
        "SELECT COUNT(*) FROM immich_asset_state WHERE connector_id='svc:immich-a'"
    ).fetchone()[0]
    b_count = catalog.db.execute(
        "SELECT COUNT(*) FROM immich_asset_state WHERE connector_id='svc:immich-b'"
    ).fetchone()[0]
    assert a_count == 3
    assert b_count == 3
    assert catalog.db.execute("SELECT COUNT(*) FROM immich_sync_state").fetchone()[0] == 2
    # re-syncing one instance leaves the other untouched
    conn_a.sync()
    assert catalog.db.execute(
        "SELECT COUNT(*) FROM immich_asset_state WHERE connector_id='svc:immich-b'"
    ).fetchone()[0] == 3
    catalog.db.close()


# -- availability tombstone ------------------------------------------------------

def test_tombstone_not_enumerated_and_reference_preserved(data_root):
    catalog = Catalog(data_root=data_root)
    transport = _transport()
    connector = ImmichConnector("svc:immich", transport, catalog=catalog)
    connector.sync()
    transport.mark_unavailable("a-1")
    assert "a-1" not in {m.asset_id for m in connector.enumerate()}
    report = connector.sync()
    assert report.tombstoned == 1
    # reference preserved with available=0, row never deleted
    row = catalog.db.execute(
        "SELECT available FROM immich_asset_state"
        " WHERE connector_id='svc:immich' AND asset_id='a-1'"
    ).fetchone()[0]
    assert row == 0
    assert catalog.db.execute("SELECT COUNT(*) FROM immich_asset_state").fetchone()[0] == 3
    obs = list(connector.revisions("a-1"))
    assert obs[-1].available is False  # tombstone in history, not deleted
    catalog.db.close()


def test_tombstone_read_original_raises(data_root):
    transport = _transport()
    connector = ImmichConnector("svc:immich", transport)
    transport.mark_unavailable("b-2")
    try:
        connector.read_original("b-2")
        assert False, "expected ConnectorError"
    except ConnectorError:
        pass


# -- feedback sink ---------------------------------------------------------------

def test_feedback_sink_disabled_by_default_is_noop():
    transport = _transport()
    sink = ImmichFeedbackSink(transport)  # enabled=False by default
    caps = sink.capabilities()
    assert caps.enabled is False
    assert caps.supported is True
    result = sink.write_favorite("b-2")  # b-2 is not a favorite by default
    assert result.ok is False
    assert result.status == "excluded"
    assert transport.is_favorite("b-2") is False  # nothing written


def test_feedback_sink_enabled_writes_confirmed_favorite():
    transport = _transport()
    sink = ImmichFeedbackSink(transport, enabled=True)
    caps = sink.capabilities()
    assert caps.enabled is True
    assert caps.supported is True
    result = sink.write_favorite("b-2")
    assert result.ok is True
    assert result.status == "written"
    assert transport.is_favorite("b-2") is True


def test_feedback_sink_writes_album_membership():
    transport = _transport()
    sink = ImmichFeedbackSink(transport, enabled=True)
    result = sink.write_favorite("c-3", album_id="fav-album")
    assert result.ok is True
    assert transport.album_members("fav-album") == {"c-3"}


def test_feedback_sink_unsupported_when_server_lacks_capability():
    transport = _NoFeedback()
    sink = ImmichFeedbackSink(transport, enabled=True)
    caps = sink.capabilities()
    assert caps.enabled is True
    assert caps.supported is False
    result = sink.write_favorite("a-1")
    assert result.ok is False
    assert result.status == "unsupported"
    assert transport.is_favorite("a-1") is False


def test_feedback_sink_never_deletes_or_admin():
    transport = _transport()
    sink = ImmichFeedbackSink(transport, enabled=True)
    assert not hasattr(sink, "delete")
    assert sink.capabilities().permission_scope == "user:write"
    sink.write_favorite("a-1")
    sink.write_favorite("b-2", album_id="a")
    # assets still available and enumerated; nothing removed
    assert transport.get_availability("a-1") is True
    assert transport.get_availability("b-2") is True
    assert {m.asset_id for m in ImmichConnector("svc:immich", transport).enumerate()} == {
        "a-1",
        "b-2",
        "c-3",
    }
