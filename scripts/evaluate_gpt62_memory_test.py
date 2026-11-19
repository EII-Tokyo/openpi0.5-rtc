import importlib.util,json
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('runner',Path(__file__).with_name('evaluate_gpt62_memory.py'));runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


def test_budget_reserves_inflight_and_unknown_failures(tmp_path):
    x=runner.Ledger(tmp_path/'ledger.json',limit=.1)
    assert x.reserve('a',.06)
    assert not x.reserve('b',.05)
    x.finish('a',status='unresolved_charge_reserved')
    assert not x.reserve('b',.05)
    x.finish('a',.02,status='completed')
    assert x.reserve('b',.05)
    assert not x.reserve('b',.01)
    assert x.total()==runner.Decimal('.07')
    y=runner.Ledger(x.path,limit=.1)
    assert y.total()==x.total()
    with pytest.raises(RuntimeError):x.finish('b',.06)


def test_cost_includes_reasoning_output_and_cache_write_tokens():
    u={'input_tokens':1000,'output_tokens':500,'input_tokens_details':{'cached_tokens':200,'cache_write_tokens':300},'output_tokens_details':{'reasoning_tokens':450}}
    assert runner.usage_cost(u,'gpt-6.2-astra')==pytest.approx((500*8+200*.4+300*10+500*40)/1e6)
