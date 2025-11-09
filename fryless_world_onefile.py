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

@dataclass
class Location:
    location_id: str
    name: str
    continent: str
    urbanicity: str
    climate: str

def rng():
    np.random.seed(C["SEED"])

def sample_location(i:int) -> Location:
    cont = np.random.choice(continents, p=[0.23,0.22,0.27,0.12,0.10,0.06])
    urb  = np.random.choice(urbanicity, p=[0.55,0.30,0.15])
    climates = {
        "North America":["Temperate","Continental","Arid"],
        "Europe":["Temperate","Oceanic","Continental"],
        "Asia":["Monsoon","Temperate","Arid","Tropical"],
        "Latin America":["Tropical","Temperate","Arid"],
        "Africa":["Tropical","Arid","Temperate"],
        "Oceania":["Oceanic","Tropical","Arid"]
    }
    clim = np.random.choice(climates[cont])
    return Location(f"L{i:06d}", f"{urb} Region {i}", cont, urb, clim)

def months():
    return pd.period_range(f"{C['YEAR']}-01", freq="M", periods=C["N_MONTHS"]).to_timestamp()

def draw_baselines(loc: Location):
    ff_penetration = {"Urban":0.62,"Suburban":0.48,"Rural":0.30}[loc.urbanicity] + np.random.normal(0,0.04)
    return dict(
        potato_acreage_kha=np.clip(np.random.lognormal(2.8,0.45), 5, 80),
        ff_penetration=ff_penetration,
        highend_share={"Urban":0.18,"Suburban":0.10,"Rural":0.04}[loc.urbanicity] + np.random.normal(0,0.02),
        sustainable_share=0.12 + np.random.normal(0,0.03),
        biodiversity_idx=np.clip(np.random.normal(0.52,0.06),0.30,0.75),
        public_interest=np.clip(np.random.normal(0.28,0.08),0.05,0.6),
        nutrition_index=np.clip(np.random.normal(0.46,0.05),0.30,0.65),
    )

def shock_deltas(t:int, SHOCK_MONTH:int=1):
    delta = t - SHOCK_MONTH
    if delta < 0:
        return dict(potato=0, ff=0, sust=0, interest=0, biodiv=0, nutr=0)

    s = float(C["SHOCK_INTENSITY"])
    def approach(shock_delta, new_equil_delta, rate):
        return shock_delta * np.exp(-(0.10+0.12)*delta) + new_equil_delta * (1 - np.exp(-(rate)*delta))
    return dict(
        potato=approach(-0.35*s, -0.25*s, 0.22),
        ff=approach(-0.17*s, -0.05*s, 0.32),
        sust=approach( 0.20*s,  0.22*s, 0.30),
        interest=approach(0.45*s, 0.20*s, 0.37),
        biodiv=0.12*s*(1-np.exp(-0.18*delta)),
        nutr=0.10*s*(1-np.exp(-0.16*delta)),
    )

def logistic(x): return 1/(1+np.exp(-x))

# =========================
# MAIN GENERATION
# =========================
def main():
    rng()
    OUT = Path(C["OUTDIR"]); OUT.mkdir(parents=True, exist_ok=True)
    ts_path    = OUT/"fryless_timeseries.csv"
    innov_path = OUT/"fryless_innovations.csv"
    events_path= OUT/"fryless_events.csv"
    dict_path  = OUT/"fryless_datadict.json"

    # 1) Locations
    locs: List[Location] = [sample_location(i) for i in range(C["N_LOCATIONS"])]

    # 2) Time series (fits in memory even when large; rows=N_LOCATIONS*N_MONTHS)
    ts_rows = []
    mlist = list(months())
    for l in locs:
        base = draw_baselines(l)
        for idx, dt in enumerate(mlist, start=1):
            d = shock_deltas(idx)
            interest = np.clip(base["public_interest"] + d["interest"] + np.random.normal(0,0.02), 0, 1)
            # Expected innovations per TS row
            lam = max(0.2, C["INNOV_RATE_BASE"] + C["INNOV_RATE_SCALE"]*interest)
            innov = min(np.random.poisson(lam), C["MAX_INNOV_PER_TS"])

            potato = np.clip(base["potato_acreage_kha"]*(1+d["potato"]+np.random.normal(0,0.02)), 1.0, None)
            biodiversity = np.clip(base["biodiversity_idx"]+d["biodiv"]+np.random.normal(0,0.01), 0, 1)
            sust = np.clip(base["sustainable_share"]+d["sust"]+np.random.normal(0,0.02), 0, 0.95)
            nutr = np.clip(base["nutrition_index"]+d["nutr"]+np.random.normal(0,0.01), 0, 1)
            ff = np.clip((1 + base["ff_penetration"] + d["ff"]) * (1+np.random.normal(0,0.015)), 0.2, 2.0)
            highend = np.clip((1 + base["highend_share"]) * (1+np.random.normal(0,0.02)), 0.2, 2.2)
            home = np.clip((1.0 + 0.10*sust) * (1+np.random.normal(0,0.02)), 0.3, 2.5)

            ts_rows.append(dict(
                location_id=l.location_id, date=dt, continent=l.continent, urbanicity=l.urbanicity, climate=l.climate,
                potato_acreage_kha=round(float(potato),3),
                biodiversity_idx=round(float(biodiversity),3),
                sustainable_menu_share=round(float(sust),3),
                public_interest_nutrition=round(float(interest),3),
                nutrition_index=round(float(nutr),3),
                ff_sales_index=round(float(ff),3),
                highend_sales_index=round(float(highend),3),
                household_cooking_index=round(float(home),3),
                monthly_innovations=int(innov),
            ))

    ts_df = pd.DataFrame(ts_rows)

    # sprinkle nulls
    for col in ["sustainable_menu_share","public_interest_nutrition","ff_sales_index","potato_acreage_kha"]:
        mask = np.random.rand(len(ts_df)) < C["NULL_RATE"]
        ts_df.loc[mask, col] = np.nan

    ts_df.to_csv(ts_path, index=False)

    # 3) Innovations — stream to disk in chunks (no big RAM), with a hard cap
    wrote_header = False
    total_innov = 0
    for start in range(0, len(ts_df), 50_000):
        chunk = ts_df.iloc[start:start+50_000]
        rows = []
        for _, r in chunk.iterrows():
            if total_innov >= C["MAX_INNOV_ROWS"]: break
            n = int(r["monthly_innovations"])
            # you can scale here again if you want extra big: n *= 2
            for _ in range(n):
                ing = np.random.choice(ingredients)
                met = np.random.choice(methods, p=[0.20,0.22,0.26,0.16,0.10,0.06])
                seas= np.random.choice(seasonings)
                method_bonus = {"AirFried":0.6,"Baked":0.45,"Roasted":0.35,"PanSeared":0.15,"Griddled":0.05,"ConfitLowOil":0.10}[met]
                x = (0.4 * (r["public_interest_nutrition"] if pd.notnull(r["public_interest_nutrition"]) else 0.3)
                     + 0.5 * (r["sustainable_menu_share"] if pd.notnull(r["sustainable_menu_share"]) else 0.15)
                     + method_bonus + np.random.normal(0,0.25))
                adoption = logistic(x)
                health = np.clip(np.random.normal(0.55 + 0.25*(met=="AirFried") + 0.10*(met=="Baked"), 0.10), 0.2, 0.95)
                env    = np.clip(np.random.normal(0.80, 0.08), 0.3, 0.98)
                rows.append(dict(
                    date=r["date"], location_id=r["location_id"],
                    ingredient=ing, method=met, seasoning=seas,
                    adoption_score=round(float(adoption),3),
                    health_score=round(float(health),3),
                    env_score=round(float(env),3)
                ))
        if rows:
            df = pd.DataFrame(rows)
            total_innov += len(df)
            df.to_csv(innov_path, mode="a", header=not wrote_header, index=False)
            wrote_header = True
        if total_innov >= C["MAX_INNOV_ROWS"]:
            break

    # 4) Events (small)
    events = []
    for l in locs:
        events.append(dict(location_id=l.location_id, date=pd.Timestamp(f"{C['YEAR']}-01-01"),
                           actor_type="System", event="GlobalMemoryWipe_FrenchFries", intensity=1.0))
        for m in [1,4,8]:
            for a in actor_types:
                base_int = {"FastFoodChain":0.9,"HighEndRestaurant":0.7,"Household":0.5}[a]
                events.append(dict(
                    location_id=l.location_id, date=pd.Timestamp(f"{C['YEAR']}-{m:02d}-01"),
                    actor_type=a, event=("ShockResponse" if m==1 else "MenuInnovation"),
                    intensity=float(np.clip(np.random.normal(base_int,0.15),0.1,1.3))
                ))
    pd.DataFrame(events).to_csv(events_path, index=False)

    # 5) Data dictionary + size summary
    summary = {
        "profile": PROFILE,
        "config": C,
        "expected_rows": {
            "timeseries": len(ts_df),
            "events": len(events),
            "innovations_written": total_innov
        }
    }
    dd = {
      "fryless_timeseries.csv": {
        "description": "Monthly per-location indicators for the fryless world.",
        "rows": len(ts_df)
      },
      "fryless_innovations.csv": {
        "description": "Ingredient × method × seasoning experiments with adoption/impact scores.",
        "rows": total_innov
      },
      "fryless_events.csv": {
        "description": "Shock and response events by actor type.",
        "rows": len(events)
      },
      "_summary.json": summary
    }
    with open(dict_path, "w", encoding="utf-8") as f:
        json.dump(dd, f, indent=2, default=str)

    print(json.dumps(summary, indent=2, default=str))
    print(f"Saved to: {Path(C['OUTDIR']).resolve()}")

if __name__ == "__main__":
    main()
