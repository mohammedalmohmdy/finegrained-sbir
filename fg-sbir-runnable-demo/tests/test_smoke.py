import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

def test_imports():
    import src.models.backbones as backbones
    import src.losses.nce as nce
    import src.losses.triplet as trip
    import src.losses.prototype as proto
    import src.data.dataset as data
    import src.utils.metrics as metrics
    assert True
