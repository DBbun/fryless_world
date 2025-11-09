# fryless_world_onefile.py
# One-file generator with explicit size controls + streaming writes.

import numpy as np, pandas as pd, json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

# =========================
# CONFIG (edit here)
# =========================
PROFILE = "xl"   # tiny | small | medium | large | xl

PROFILES = {
    "tiny":   dict(SEED=42, YEAR=2029, N_LOCATIONS=25,  N_MONTHS=12,
                   INNOV_RATE_BASE=0.5, INNOV_RATE_SCALE=2.2, MAX_INNOV_PER_TS=6,
                   SHOCK_INTENSITY=1.0, OUTDIR="./out_fryless", NULL_RATE=0.02,
                   CHUNK_ROWS=100_000, MAX_INNOV_ROWS=300_000),
    "small":  dict(SEED=42, YEAR=2029, N_LOCATIONS=120, N_MONTHS=12,
                   INNOV_RATE_BASE=0.5, INNOV_RATE_SCALE=2.2, MAX_INNOV_PER_TS=10,
                   SHOCK_INTENSITY=1.0, OUTDIR="./out_fryless", NULL_RATE=0.02,
                   CHUNK_ROWS=250_000, MAX_INNOV_ROWS=1_500_000),
    "medium": dict(SEED=42, YEAR=2029, N_LOCATIONS=500, N_MONTHS=12,
                   INNOV_RATE_BASE=0.45, INNOV_RATE_SCALE=2.0, MAX_INNOV_PER_TS=12,
                   SHOCK_INTENSITY=1.0, OUTDIR="./out_fryless", NULL_RATE=0.02,
                   CHUNK_ROWS=500_000, MAX_INNOV_ROWS=6_000_000),
    "large":  dict(SEED=42, YEAR=2029, N_LOCATIONS=2_000, N_MONTHS=12,
                   INNOV_RATE_BASE=0.40, INNOV_RATE_SCALE=1.8, MAX_INNOV_PER_TS=14,
                   SHOCK_INTENSITY=1.0, OUTDIR="./out_fryless", NULL_RATE=0.02,
                   CHUNK_ROWS=1_000_000, MAX_INNOV_ROWS=20_000_000),
    "xl":     dict(SEED=42, YEAR=2029, N_LOCATIONS=10_000, N_MONTHS=12,
                   INNOV_RATE_BASE=0.35, INNOV_RATE_SCALE=1.6, MAX_INNOV_PER_TS=16,
                   SHOCK_INTENSITY=1.0, OUTDIR="./out_fryless", NULL_RATE=0.02,
                   CHUNK_ROWS=1_500_000, MAX_INNOV_ROWS=60_000_000),
}

C = PROFILES[PROFILE]

# Optional: override any single value here if you want custom
# C["N_LOCATIONS"] = 3000

# =========================
# STATIC VOCABS & HELPERS
# =========================
continents  = ["North America","Europe","Asia","Latin America","Africa","Oceania"]
urbanicity  = ["Urban","Suburban","Rural"]
actor_types = ["FastFoodChain","HighEndRestaurant","Household"]
ingredients = ["Zucchini","Avocado","Parsnip","Plantain","Taro","Cassava","SweetPotato","Carrot","Beet","Celeriac","Yucca","Kohlrabi"]
methods     = ["Baked","Roasted","AirFried","PanSeared","Griddled","ConfitLowOil"]
seasonings  = ["Smoked Paprika","Garlic-Lemon","Sea Salt","Za'atar","Chili-Lime","Sumac","Herb de Provence"]

...

The full generator source code is available under a paid commercial license
from DBbun LLC. To purchase access, email: contact@dbbun.com
