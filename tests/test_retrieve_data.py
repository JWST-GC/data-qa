"""Offline unit tests for data_qa.retrieve_data (MAST timeout config + size
precheck helpers; no network)."""
import pytest

from data_qa import retrieve_data as rd


# ------------------------------------------------------------------- timeout config
def test_configure_mast_sets_conf_timeout_and_pagesize():
    """astroquery >= 0.4 has no Observations.TIMEOUT; the knobs live on
    astroquery.mast.conf and must be bounded (the 22h-hang incident)."""
    from astroquery.mast import conf
    old_timeout, old_pagesize = conf.timeout, conf.pagesize
    try:
        got = rd.configure_mast()
        assert got is conf
        assert conf.timeout == rd.MAST_TIMEOUT_S == 120
        assert conf.pagesize == rd.MAST_PAGESIZE
        rd.configure_mast(timeout_s=7, pagesize=9)
        assert conf.timeout == 7 and conf.pagesize == 9
    finally:
        conf.timeout, conf.pagesize = old_timeout, old_pagesize


def test_mast_query_errors_cover_requests_and_astroquery():
    import requests
    from astroquery.exceptions import RemoteServiceError
    from astroquery.exceptions import TimeoutError as AqTimeout
    errs = rd.mast_query_errors()
    assert issubclass(requests.exceptions.ConnectionError, errs)
    assert issubclass(requests.exceptions.Timeout, errs)
    assert issubclass(RemoteServiceError, errs)
    assert issubclass(AqTimeout, errs)


# -------------------------------------------------------------------- size precheck
class _FakeProducts:
    def __init__(self, sizes, colnames=("productFilename", "size")):
        self._sizes = list(sizes)
        self.colnames = list(colnames)

    def __getitem__(self, key):
        assert key == "size"
        return self._sizes


def test_product_list_size_sums_bytes(monkeypatch):
    monkeypatch.setattr(rd, "filtered_products",
                        lambda *a, **kw: _FakeProducts([100, 200, 44]))
    assert rd.product_list_size_bytes(2221, "001") == 344


def test_product_list_size_missing_observation(monkeypatch):
    monkeypatch.setattr(rd, "filtered_products", lambda *a, **kw: None)
    assert rd.product_list_size_bytes(2221, "001") is None


def test_product_list_size_missing_size_column(monkeypatch):
    monkeypatch.setattr(rd, "filtered_products",
                        lambda *a, **kw: _FakeProducts([], colnames=["productFilename"]))
    assert rd.product_list_size_bytes(2221, "001") is None


def test_product_list_size_unparseable_entries(monkeypatch):
    monkeypatch.setattr(rd, "filtered_products",
                        lambda *a, **kw: _FakeProducts([100, "masked", 3]))
    assert rd.product_list_size_bytes(2221, "001") is None
