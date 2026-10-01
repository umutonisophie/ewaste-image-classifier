"""Repair the e-waste checkpoint that shipped with a stray file in its archive.

`ewaste_mobilenet_master_model.pth` is a PyTorch zip archive, but it also
contained a loose `split_dataset.py` at the archive root. torch validates the
archive layout and refuses to load it, so the model was unusable even though
the weights inside were intact and correct.

This script rebuilds the archive containing only the entries torch expects,
which produces a checkpoint that loads cleanly and fits a 29-class
MobileNetV3-Large exactly.

The original is left untouched so the failure stays reproducible and testable.

Usage:  python repair_checkpoint.py
"""

import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
BROKEN = HERE / "ewaste_mobilenet_master_model.pth"
REPAIRED = HERE / "ewaste_mobilenet_master_model.repaired.pth"


def main() -> int:
    if not BROKEN.exists():
        print(f"error: {BROKEN.name} not found", )
        return 1

    if REPAIRED.exists():
        print(f"{REPAIRED.name} already exists; nothing to do")
        return 0

    with zipfile.ZipFile(BROKEN) as archive:
        names = archive.namelist()
        # Every legitimate torch entry sits under the archive's own root name.
        if not names:
            print("error: archive is empty", )
            return 1

        root = names[0].split("/")[0]
        keep = [n for n in names if n.startswith(f"{root}/")]
        dropped = [n for n in names if not n.startswith(f"{root}/")]

        if not dropped:
            print(f"{BROKEN.name} looks well-formed; nothing to repair")
            return 0

        print(f"archive entries : {len(names)}")
        print(f"expected       : {len(keep)} (under {root}/)")
        print(f"dropped        : {dropped}")

        with zipfile.ZipFile(REPAIRED, "w", zipfile.ZIP_STORED) as out:
            for name in keep:
                out.writestr(name, archive.read(name))

    print(f"\nwrote {REPAIRED.name} ({REPAIRED.stat().st_size:,} bytes)")

    # Verify before telling anyone it worked.
    import torch
    state = torch.load(REPAIRED, map_location="cpu", weights_only=False)
    print(f"verified: loads cleanly, {len(state)} tensors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
