"""Structural test for kc23_s3_inventory.py. KC-S3 has no outcome letters
(the plan states none), so this checks the inventory table is well-formed
rather than driving a decision grid."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kc23_s3_inventory import build_inventory, FAMILIES, NORMS


def test_inventory_covers_every_cell():
    inv = build_inventory()
    assert len(inv) == len(FAMILIES) * len(NORMS)
    assert set(inv.columns) == {"family", "normalization", "present", "source"}
    assert set(inv["family"]) == set(FAMILIES)
    assert set(inv["normalization"]) == set(NORMS)


def test_present_is_boolean():
    inv = build_inventory()
    assert inv["present"].dtype == bool


if __name__ == "__main__":
    fns = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
