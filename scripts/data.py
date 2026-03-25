from pathlib import Path

FIG_ROOT = Path(__file__).parent.parent / "figures"
IMG_ROOT = FIG_ROOT / "img"

DATASETS = {
    "drive_map": 17541_78906_091283691,
    # "drive_map": 17545_53380_516283691, # larger range
    "ramsey_2d": 17540_65787_573283691,
    "larmor_peak_q1_ro_1_2": 17537_77456_344283691,
    "larmor_peak_q2_ro_1_2": 17537_77674_583283691,
    "larmor_peak_q2_ro_2_3": 17537_76966_722283691,
    "larmor_peak_q3_ro_2_3": 17537_77526_420283691,
    "rabi_q1_ro_1_2": 17537_77475_572283691,
    "rabi_q2_ro_1_2": 17537_77693_777283691,
    "rabi_q2_ro_2_3": 17537_77013_431283691,
    "rabi_q3_ro_2_3": 17537_77551_561283691,
    "blind_rb": 17546_66229_637283691,
}

IMAGES = {
    "device": IMG_ROOT / "343-sem.png",
}