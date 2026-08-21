"""Pointing centres are carried in the state file (issue #72 item 1).

The per-tile reference catalogs of keflavich/jwst-gc-pipeline#415 need a
centre per observation.  The monitor already polls one -- MAST returns
``s_ra``/``s_dec`` on every row -- and used to drop it, so a builder had to
re-query MAST for a coordinate the state file could have held since the
observation was planned.

Every row of one observation reports the same centre.  Measured 2026-08-20:
all 139 planned 10678 observations (12 rows each) and all 3 delivered 2221
observations, maximum within-observation spread 0.00".  So "the observation's
centre" is well defined and the first row carrying finite coordinates answers
for it.
"""
import io
import json
from contextlib import redirect_stdout, redirect_stderr

import pytest

from data_qa import mast_monitor as mm


def _state(tmp_path, obs):
    path = tmp_path / 'mast_state.json'
    path.write_text(json.dumps({'programs': {'10678': {'obs': obs}}}))
    return str(path)


#: the shape the live state file uses (verified against
#: /orange/adamginsburg/jwst/ops/mast_state.json on 2026-08-20)
def _rec(tile, ra, dec, calib=-1, released=False, instrument='NIRCAM/IMAGE'):
    return {'calib_level': calib, 'filters': 'F212N;F480M',
            'instrument_name': instrument, 'released': released,
            't_max': None, 't_obs_release': None, 'target_name': tile,
            's_ra': ra, 's_dec': dec}


def test_the_columns_the_query_asks_for_include_the_centre():
    assert 's_ra' in mm.MONITOR_COLUMNS and 's_dec' in mm.MONITOR_COLUMNS
    # and they are read as numbers; parsed as strings they reach the state
    # file as '266.16495791666666' and every consumer has to re-cast
    assert 's_ra' in mm._FLOAT_COLUMNS and 's_dec' in mm._FLOAT_COLUMNS


def test_summarize_keeps_the_centre():
    rows = [{'obs_id': 'jw10678001001_xx101_00001_nircam', 'calib_level': -1,
             't_obs_release': None, 't_max': None,
             'instrument_name': 'NIRCAM/IMAGE', 'filters': 'F212N;F480M',
             'target_name': 'GC_1', 's_ra': 266.4, 's_dec': -29.0}]
    rec = mm.summarize(rows, poll_mjd=60000.0)['jw10678001001_xx101_00001_nircam']
    assert rec['s_ra'] == 266.4 and rec['s_dec'] == -29.0


def test_one_row_per_observation_not_per_mast_row(tmp_path):
    """A 10678 observation has 12 MAST rows; the builder wants one tile."""
    obs = {}
    for i in range(1, 13):
        obs[f'jw10678001001_xx1{i:02d}_{i:05d}_nircam'] = _rec('GC_1', 266.4, -29.0)
    for i in range(1, 13):
        obs[f'jw10678002001_xx1{i:02d}_{i:05d}_nircam'] = _rec('GC_2', 266.5, -29.1)
    got = mm.pointings(_state(tmp_path, obs), '10678')
    assert [r['obsnum'] for r in got] == ['001', '002']
    assert [r['tile'] for r in got] == ['GC_1', 'GC_2']
    assert got[0]['ra'] == 266.4 and got[1]['dec'] == -29.1


def test_an_observation_with_no_coordinates_is_omitted(tmp_path):
    """Reporting (0, 0) would send a builder to query a field in Cetus."""
    obs = {'jw10678001001_xx101_00001_nircam': _rec('GC_1', None, None),
           'jw10678002001_xx101_00001_nircam': _rec('GC_2', 266.5, -29.1)}
    got = mm.pointings(_state(tmp_path, obs), '10678')
    assert [r['obsnum'] for r in got] == ['002']


def test_released_only_selects_what_is_on_the_archive(tmp_path):
    obs = {'jw10678001001_xx101_00001_nircam': _rec('GC_1', 266.4, -29.0),
           'jw10678002001_xx101_00001_nircam': _rec('GC_2', 266.5, -29.1,
                                                    calib=3, released=True)}
    everything = mm.pointings(_state(tmp_path, obs), '10678')
    released = mm.pointings(_state(tmp_path, obs), '10678', released_only=True)
    assert [r['obsnum'] for r in everything] == ['001', '002']
    assert [r['obsnum'] for r in released] == ['002']


def test_an_unknown_program_is_empty_rather_than_a_crash(tmp_path):
    assert mm.pointings(_state(tmp_path, {}), '99999') == []


def test_the_cli_prints_a_table_a_build_loop_can_read(tmp_path):
    obs = {'jw10678001001_xx101_00001_nircam': _rec('GC_1', 266.164958, -29.404778)}
    out = io.StringIO()
    with redirect_stdout(out):
        rc = mm.print_pointings(_state(tmp_path, obs), '10678')
    assert rc == 0
    lines = out.getvalue().strip().splitlines()
    assert lines[0].split('\t') == ['obsnum', 'tile', 'ra_deg', 'dec_deg',
                                    'calib', 'released']
    body = lines[1].split('\t')
    assert body[0] == '001' and body[1] == 'GC_1'
    # full precision: rounding the centre to 2 dp moves it ~10" on the sky
    assert body[2] == '266.164958' and body[3] == '-29.404778'


def test_the_cli_refuses_rather_than_printing_an_empty_table(tmp_path):
    """A driver reading zero lines must not read that as 'nothing to build'."""
    err = io.StringIO()
    with redirect_stderr(err), redirect_stdout(io.StringIO()):
        rc = mm.print_pointings(_state(tmp_path, {}), '10678')
    assert rc == 1
    assert 'no observations of program 10678' in err.getvalue()


def test_the_cli_verb_is_wired(tmp_path, monkeypatch):
    """--pointings must reach print_pointings and exit without polling MAST."""
    obs = {'jw10678001001_xx101_00001_nircam': _rec('GC_1', 266.4, -29.0)}
    path = _state(tmp_path, obs)

    def _boom(*a, **k):                     # nothing may hit the network
        raise AssertionError('--pointings queried MAST')

    monkeypatch.setattr(mm, 'query_program', _boom)
    out = io.StringIO()
    with redirect_stdout(out):
        rc = mm.main(['--pointings', '10678', '--state', path])
    assert rc == 0
    assert 'GC_1' in out.getvalue()
