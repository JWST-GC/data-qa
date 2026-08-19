"""Registry field mapping, incl. per-instrument overrides for coordinated-parallel programs."""
from data_qa import observations as obsmod
from data_qa import mast_monitor as MM


def _fake_rows():
    """One NIRCam and one MIRI released product for jw02221 o002.  In program 2221 the two
    instruments share the obs number but point at different fields: NIRCam images Cloud C, the
    MIRI parallel images the Brick (MAST target_name BRICK-IKP2016-G0.253+0.015)."""
    return [
        {"obs_id": "jw02221-o002_t001_nircam_clear-f200w",
         "instrument_name": "NIRCAM/IMAGE", "calib_level": 3, "filters": "F200W", "t_max": 59800.0},
        {"obs_id": "jw02221-o002_t001_miri_f2550w",
         "instrument_name": "MIRI/IMAGE", "calib_level": 3, "filters": "F2550W", "t_max": 59800.0},
    ]


def test_miri_o002_overrides_to_brick_nircam_stays_cloudc(monkeypatch):
    monkeypatch.setattr(MM, "query_program", lambda program: _fake_rows())
    obs = {(o.instrument): o for o in obsmod._observations_for_program(2221)}
    assert obs["NIRCam"].target == "Cloud C"          # pipeline obsnum->field (unchanged)
    assert obs["NIRCam"].release_field == "cloudc"
    assert obs["MIRI"].target == "Brick"              # per-instrument override applied
    assert obs["MIRI"].release_field == "brick"
    # issue titles the two produce -- MIRI must NOT read "Cloud C" (issue #31 mislabel)
    assert obs["MIRI"].issue_title == "Brick — jw02221-o002 (MIRI)"
    assert obs["NIRCam"].issue_title == "Cloud C — jw02221-o002 (NIRCam)"


def test_field_override_is_instrument_specific():
    # the override key is (program, obsnum, instrument): only 2221 o002 MIRI is remapped
    assert obsmod.FIELD_OVERRIDE.get((2221, "002", "MIRI")) == "brick"
    assert obsmod.FIELD_OVERRIDE.get((2221, "002", "NIRCam")) is None
    assert obsmod.FIELD_OVERRIDE.get((2221, "001", "MIRI")) is None
