import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from compare_candidate import decide  # noqa: E402


def test_thay_khi_tot_hon_dang_ke():
    assert decide(0.2077, 0.1900)[0] == "thay"


def test_giu_khi_chi_nhinh_hon_khong_dang_ke():
    assert decide(0.2077, 0.2050)[0] == "giu"


def test_giu_khi_te_hon():
    assert decide(0.2077, 0.2300)[0] == "giu"


def test_margin_tuy_chinh():
    assert decide(0.2000, 0.1950, margin=0.01)[0] == "giu"
    assert decide(0.2000, 0.1950, margin=0.004)[0] == "thay"
