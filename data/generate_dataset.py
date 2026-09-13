"""
Realistic synthetic retail dataset generator (v2).

Replaces the old uniformly-random dataset with one that behaves like a real
Slovenian retail chain, while keeping the exact same 76-column schema and the
same product / brand / category vocabulary that the agent
(ParameterExtractor.VALID_PRODUCT_NAMES etc.) already knows.

What "realistic" means here, concretely:

1.  Six real stores with different sizes:
        Ljubljana BTC City   (largest)   Ljubljana Citypark
        Maribor Europark                 Maribor Qlandia
        Celje Citycenter                 Koper Supernova (coastal - summer boost)
2.  Time patterns: stores are CLOSED on Sundays and Slovenian public holidays,
    Fridays/Saturdays are busiest, December is peak season, Black Friday is a
    huge spike, mild year-over-year growth and price inflation.
3.  Category seasonality: drinks peak in summer, toys and electronics peak in
    November/December, clothing peaks at season changes and during sales.
4.  Every product has ONE fixed id, brand, category and a realistic EUR price
    (a chocolate bar costs ~2 EUR, a MacBook ~1250 EUR), with small noise and
    yearly inflation on top.
5.  Promotions are a real calendar (January sale, Easter, Black Friday week,
    December holidays, ...). A promotion covers a subset of products of one
    category; DURING its window those products genuinely sell more (uplift
    driven by the effectiveness label) and carry the promotion's discount.
    Rows outside any window carry the nearest promotion of that product as a
    tag (the PromotionAnalyzer uses the tag only as a product<->promo link)
    but get discount 0.
6.  Deliberate anomalies for the AnomalyDetector: Black Friday spikes,
    pre-Christmas rush, and a two-week closure of Maribor Qlandia in
    September 2024 (renovation).
7.  Customer-level columns are COHERENT: every aggregate (total_sales per
    customer, avg_transaction_value, last_purchase_date, churned, preferred
    store...) is computed from that customer's actual transaction rows.

Row grain: one row = one product line of an in-store transaction (same as the
old dataset). ~1.05 million rows, 60,000 customers, 2023-01-01 .. 2025-12-31.

Run:  python data/generate_dataset.py     (writes data/retail_data_v2.csv)
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

N_ROWS = 1_050_000
N_CUSTOMERS = 60_000
START, END = "2023-01-01", "2025-12-31"
OUT_PATH = "data/retail_data_v2.csv"

DAYS = pd.date_range(START, END, freq="D")
N_DAYS = len(DAYS)
DATA_END = pd.Timestamp(END)

# ---------------------------------------------------------------------------
# 1. Stores
# ---------------------------------------------------------------------------

STORES = ["Ljubljana BTC City", "Ljubljana Citypark", "Maribor Europark",
          "Maribor Qlandia", "Celje Citycenter", "Koper Supernova"]
STORE_CITY = ["Ljubljana", "Ljubljana", "Maribor", "Maribor", "Celje", "Koper"]
# Relative store size / footfall (BTC City is the biggest shopping area in SI)
STORE_WEIGHT = np.array([0.27, 0.21, 0.17, 0.10, 0.13, 0.12])

CITY_STATE = {"Ljubljana": "Osrednjeslovenska", "Maribor": "Podravska",
              "Celje": "Štajerska", "Koper": "Primorska"}
CITY_ZIP = {"Ljubljana": 1000, "Maribor": 2000, "Celje": 3000, "Koper": 6000}
CITIES = ["Ljubljana", "Maribor", "Celje", "Koper"]
# Where customers live (roughly proportional to the cities' populations)
CITY_POP_W = np.array([0.40, 0.24, 0.18, 0.18])

# Road distances between the four cities in km (for distance_to_store),
# order: Ljubljana, Maribor, Celje, Koper
CITY_KM_MATRIX = np.array([
    [0, 128, 74, 105],
    [128, 0, 54, 232],
    [74, 54, 0, 178],
    [105, 232, 178, 0],
], dtype=float)

# ---------------------------------------------------------------------------
# 2. Product catalog (name, brand, category, id from the old dataset,
#    base 2023 price EUR, popularity weight, rating, size, weight kg,
#    color, material, shelf life days)
# ---------------------------------------------------------------------------

CATALOG = [
    # --- Electronics ---
    ("iPad 10. generacije", "Apple", "Electronics", 1480, 429.00, 1.2, 4.6, "Small", 0.48, "White", "Metal", 1825),
    ("iPhone 15", "Apple", "Electronics", 5367, 979.00, 1.6, 4.8, "Small", 0.17, "Black", "Metal", 1825),
    ("MacBook Air M2", "Apple", "Electronics", 9134, 1249.00, 0.9, 4.8, "Medium", 1.24, "White", "Metal", 1825),
    ("AirPods Pro", "Apple", "Electronics", 2216, 279.00, 1.5, 4.7, "Small", 0.06, "White", "Plastic", 1825),
    ("Galaxy S24", "Samsung", "Electronics", 642, 899.00, 1.3, 4.7, "Small", 0.17, "Black", "Metal", 1825),
    ("Galaxy Tab S9", "Samsung", "Electronics", 4404, 899.00, 0.8, 4.5, "Small", 0.50, "Black", "Metal", 1825),
    ("Galaxy Buds slušalke", "Samsung", "Electronics", 2218, 149.00, 1.1, 4.4, "Small", 0.05, "White", "Plastic", 1825),
    ("Samsung QLED televizor", "Samsung", "Electronics", 460, 849.00, 0.8, 4.5, "Large", 12.0, "Black", "Metal", 1825),
    ("LG OLED televizor", "LG", "Electronics", 3311, 1199.00, 0.7, 4.7, "Large", 14.0, "Black", "Metal", 1825),
    ("LG hladilnik", "LG", "Electronics", 6986, 749.00, 0.5, 4.3, "Large", 65.0, "White", "Metal", 1825),
    ("LG pralni stroj", "LG", "Electronics", 8618, 549.00, 0.6, 4.4, "Large", 60.0, "White", "Metal", 1825),
    ("LG zvočna vrstica", "LG", "Electronics", 8280, 249.00, 0.7, 4.2, "Medium", 2.8, "Black", "Plastic", 1825),
    # --- Clothing ---
    ("Oversized blazer", "Zara", "Clothing", 6025, 59.99, 0.8, 4.2, "Medium", 0.60, "Black", "Cotton", 3650),
    ("Osnovna majica", "Zara", "Clothing", 7739, 9.99, 1.8, 4.1, "Medium", 0.20, "White", "Cotton", 3650),
    ("Slim fit jeans hlače", "Zara", "Clothing", 1376, 49.99, 1.2, 4.2, "Medium", 0.65, "Blue", "Cotton", 3650),
    ("Pleten pulover", "Zara", "Clothing", 9560, 45.99, 0.9, 4.3, "Medium", 0.50, "Red", "Cotton", 3650),
    ("Hoodie pulover", "HM", "Clothing", 1883, 29.99, 1.4, 4.3, "Medium", 0.55, "Blue", "Cotton", 3650),
    ("Cargo hlače", "HM", "Clothing", 568, 39.99, 1.0, 4.0, "Medium", 0.60, "Green", "Cotton", 3650),
    ("Osnovni top", "HM", "Clothing", 1149, 7.99, 1.3, 3.9, "Small", 0.15, "White", "Cotton", 3650),
    ("Jeans jakna", "HM", "Clothing", 9847, 59.99, 0.7, 4.2, "Medium", 0.80, "Blue", "Cotton", 3650),
    ("AIRism spodnje perilo", "Uniqlo", "Clothing", 7704, 14.99, 0.9, 4.5, "Small", 0.10, "White", "Cotton", 3650),
    ("Ravne jeans hlače", "Uniqlo", "Clothing", 288, 49.99, 0.9, 4.4, "Medium", 0.65, "Blue", "Cotton", 3650),
    ("Ultra Light Down jakna", "Uniqlo", "Clothing", 1877, 89.99, 0.8, 4.6, "Medium", 0.40, "Black", "Cotton", 3650),
    ("Heattech majica", "Uniqlo", "Clothing", 1082, 19.99, 1.0, 4.5, "Medium", 0.20, "Black", "Cotton", 3650),
    # --- Food ---
    ("Milka lešnik čokolada", "Milka", "Food", 1597, 2.19, 2.2, 4.6, "Small", 0.10, "Blue", "Plastic", 365),
    ("Milka Alpska mlečna čokolada", "Milka", "Food", 1992, 1.99, 2.5, 4.7, "Small", 0.10, "Blue", "Plastic", 365),
    ("Milka Oreo čokolada", "Milka", "Food", 931, 2.29, 1.8, 4.5, "Small", 0.10, "Blue", "Plastic", 365),
    ("Milka piškoti s čokolado", "Milka", "Food", 3340, 2.79, 1.5, 4.4, "Small", 0.15, "Blue", "Plastic", 270),
    ("Penne Rigate", "Barilla", "Food", 8642, 1.79, 2.0, 4.5, "Medium", 0.50, "Blue", "Cardboard", 730),
    ("Fusilli testenine", "Barilla", "Food", 7414, 1.79, 1.8, 4.5, "Medium", 0.50, "Blue", "Cardboard", 730),
    ("Špageti št. 5", "Barilla", "Food", 7151, 1.69, 2.2, 4.6, "Medium", 0.50, "Blue", "Cardboard", 730),
    ("Pesto Genovese omaka", "Barilla", "Food", 4854, 3.29, 1.3, 4.4, "Small", 0.19, "Green", "Glass", 540),
    ("Paradižnikova juha", "Podravka", "Food", 7193, 1.99, 1.5, 4.2, "Medium", 0.40, "Red", "Cardboard", 730),
    ("Vegeta začimba", "Podravka", "Food", 7339, 2.49, 2.0, 4.7, "Small", 0.25, "Blue", "Plastic", 730),
    ("Fižolova enolončnica", "Podravka", "Food", 3009, 2.89, 1.2, 4.1, "Medium", 0.42, "Red", "Metal", 1095),
    ("Ajvar", "Podravka", "Food", 92, 3.49, 1.6, 4.6, "Small", 0.35, "Red", "Glass", 730),
    ("Lay's klasični čips", "PepsiCO", "Food", 2012, 2.19, 2.0, 4.3, "Small", 0.14, "Red", "Plastic", 180),
    # --- Drinks ---
    ("Pepsi", "PepsiCO", "Drinks", 3445, 1.49, 1.8, 4.3, "Medium", 1.55, "Blue", "Plastic", 365),
    ("7UP", "PepsiCO", "Drinks", 1804, 1.49, 1.2, 4.2, "Medium", 1.55, "Green", "Plastic", 365),
    ("Gatorade pomaranča", "PepsiCO", "Drinks", 8917, 1.99, 1.0, 4.2, "Small", 0.75, "Red", "Plastic", 365),
    ("Radenska Classic mineralna voda", "Radenska", "Drinks", 8507, 0.89, 2.5, 4.6, "Medium", 1.55, "Green", "Plastic", 540),
    ("Radenska Naturelle", "Radenska", "Drinks", 8037, 0.85, 2.0, 4.5, "Medium", 1.55, "Blue", "Plastic", 540),
    ("Radenska limeta gazirana voda", "Radenska", "Drinks", 9870, 0.99, 1.4, 4.3, "Medium", 1.55, "Green", "Plastic", 540),
    ("Radenska 1,5 L plastenka", "Radenska", "Drinks", 735, 1.09, 2.3, 4.5, "Large", 1.55, "Blue", "Plastic", 540),
    ("Jabolčni nektar", "Fructal", "Drinks", 3281, 1.69, 1.4, 4.4, "Medium", 1.05, "Green", "Glass", 270),
    ("Breskov sok", "Fructal", "Drinks", 5553, 1.79, 1.3, 4.4, "Medium", 1.05, "Red", "Glass", 270),
    ("Pomarančni sok", "Fructal", "Drinks", 856, 1.89, 1.5, 4.5, "Medium", 1.05, "Red", "Glass", 270),
    ("Sadni napitek", "Fructal", "Drinks", 7035, 1.39, 1.1, 4.0, "Medium", 1.55, "Red", "Plastic", 365),
    # --- Toys ---
    ("LEGO Classic kocke", "LEGO", "Toys", 7781, 29.99, 1.6, 4.8, "Medium", 0.70, "Red", "Plastic", 3650),
    ("LEGO Star Wars X-Wing", "LEGO", "Toys", 5266, 54.99, 1.1, 4.8, "Medium", 0.60, "White", "Plastic", 3650),
    ("LEGO City policijska postaja", "LEGO", "Toys", 9463, 89.99, 0.9, 4.8, "Large", 1.30, "Blue", "Plastic", 3650),
    ("LEGO Technic avto", "LEGO", "Toys", 8637, 49.99, 1.0, 4.7, "Medium", 0.80, "Green", "Plastic", 3650),
    ("Nerf Elite blaster", "Hasbro", "Toys", 5142, 34.99, 1.2, 4.4, "Medium", 0.70, "Blue", "Plastic", 3650),
    ("Play-Doh set za modeliranje", "Hasbro", "Toys", 8715, 16.99, 1.3, 4.3, "Medium", 0.90, "Red", "Plastic", 1095),
    ("Monopoly klasična izdaja", "Hasbro", "Toys", 3121, 29.99, 1.2, 4.6, "Medium", 1.00, "White", "Cardboard", 3650),
    ("Operacija družabna igra", "Hasbro", "Toys", 2843, 24.99, 0.8, 4.2, "Medium", 0.80, "White", "Cardboard", 3650),
    ("Barbie sanjska hiša", "Mattel", "Toys", 8447, 219.99, 0.5, 4.6, "Large", 5.00, "Red", "Plastic", 3650),
    ("Uno kartna igra", "Mattel", "Toys", 414, 9.99, 1.8, 4.7, "Small", 0.15, "Red", "Cardboard", 3650),
    ("Hot Wheels dirkalna steza", "Mattel", "Toys", 5134, 44.99, 0.9, 4.4, "Large", 1.40, "Red", "Plastic", 3650),
    ("Fisher-Price učna igrača", "Mattel", "Toys", 5934, 24.99, 1.0, 4.5, "Medium", 0.80, "Red", "Plastic", 1095),
]

CATEGORIES = ["Food", "Drinks", "Clothing", "Electronics", "Toys"]

P_NAME = np.array([p[0] for p in CATALOG])
P_BRAND = np.array([p[1] for p in CATALOG])
P_CAT = np.array([p[2] for p in CATALOG])
P_ID = np.array([p[3] for p in CATALOG])
P_PRICE = np.array([p[4] for p in CATALOG])
P_POP = np.array([p[5] for p in CATALOG])
P_RATING = np.array([p[6] for p in CATALOG])
P_SIZE = np.array([p[7] for p in CATALOG])
P_WEIGHT = np.array([p[8] for p in CATALOG])
P_COLOR = np.array([p[9] for p in CATALOG])
P_MATERIAL = np.array([p[10] for p in CATALOG])
P_SHELF = np.array([p[11] for p in CATALOG])
N_PRODUCTS = len(CATALOG)

# Products are returned at realistic per-category rates
CAT_RETURN_RATE = {"Clothing": 0.09, "Electronics": 0.05, "Toys": 0.04,
                   "Food": 0.008, "Drinks": 0.005}
P_RETURN = np.array([round(CAT_RETURN_RATE[c] * rng.uniform(0.7, 1.3), 3) for c in P_CAT])
P_REVIEWS = (P_POP * rng.uniform(300, 3000, N_PRODUCTS)).astype(int)

# ---------------------------------------------------------------------------
# 3. Calendar: daily traffic weights
# ---------------------------------------------------------------------------

# Slovenian public holidays when shops are closed (law since 2021: also all
# Sundays). Easter Monday moves each year.
FIXED_HOLIDAYS = ["01-01", "01-02", "02-08", "04-27", "05-01", "05-02",
                  "06-25", "08-15", "10-31", "11-01", "12-25", "12-26"]
EASTER_MONDAY = ["2023-04-10", "2024-04-01", "2025-04-21"]
BLACK_FRIDAY = ["2023-11-24", "2024-11-29", "2025-11-28"]

closed = np.zeros(N_DAYS, dtype=bool)
closed |= DAYS.dayofweek == 6  # Sundays
for d in DAYS:
    if d.strftime("%m-%d") in FIXED_HOLIDAYS:
        closed[DAYS.get_loc(d)] = True
for d in EASTER_MONDAY:
    closed[DAYS.get_loc(d)] = True

# Mon..Sat traffic profile (Friday/Saturday busiest)
DOW_W = np.array([0.95, 0.88, 0.92, 1.00, 1.18, 1.40, 0.0])
# Month profile: December peak, quiet Jan/Feb, small summer dip
MONTH_W = np.array([0.92, 0.88, 0.96, 1.00, 1.02, 1.00, 0.96, 0.94, 1.00, 1.03, 1.12, 1.35])

day_w = DOW_W[DAYS.dayofweek] * MONTH_W[DAYS.month - 1]
# Mild year-over-year growth of the whole chain (~4.5 %/year)
day_w *= 1.0 + 0.045 * (DAYS.year - 2023)
# Anomaly: Black Friday spike + pre-Christmas rush (Dec 20-23)
for d in BLACK_FRIDAY:
    day_w[DAYS.get_loc(d)] *= 4.5
rush = (DAYS.month == 12) & DAYS.day.isin([20, 21, 22, 23])
day_w[rush.values if hasattr(rush, "values") else rush] *= 1.8
# Ordinary day-to-day randomness (weather, paydays, ...)
day_w *= rng.uniform(0.88, 1.12, N_DAYS)
day_w[closed] = 0.0

# Per-store daily weights: base size x day traffic (+ store-specific effects)
store_day_w = day_w[:, None] * STORE_WEIGHT[None, :]
summer = np.isin(DAYS.month, [6, 7, 8])
store_day_w[summer, STORES.index("Koper Supernova")] *= 1.35  # coast tourism
# Anomaly: Maribor Qlandia closed for renovation 2024-09-02 .. 2024-09-15
ql = STORES.index("Maribor Qlandia")
reno = (DAYS >= "2024-09-02") & (DAYS <= "2024-09-15")
store_day_w[reno, ql] = 0.0
store_day_w *= rng.uniform(0.93, 1.07, store_day_w.shape)

# ---------------------------------------------------------------------------
# 4. Promotion calendar
# ---------------------------------------------------------------------------

EASTER = {2023: "2023-04-09", 2024: "2024-03-31", 2025: "2025-04-20"}
BF_WEEK = {2023: ("2023-11-20", "2023-11-26"),
           2024: ("2024-11-25", "2024-12-01"),
           2025: ("2025-11-24", "2025-11-30")}
# Effectiveness label drives the REAL sales uplift during the window, so the
# label in the data is true by construction.
UPLIFT = {"High": 1.65, "Medium": 1.30, "Low": 1.05}
DISCOUNT = {"20% Off": 0.20, "Flash Sale": 0.35, "Buy One Get One Free": 0.50}

promo_rows = []
for y in (2023, 2024, 2025):
    e = pd.Timestamp(EASTER[y])
    promo_rows += [
        ("Clothing", "20% Off", f"{y}-01-03", f"{y}-01-31", "High"),         # January sale
        ("Electronics", "Flash Sale", f"{y}-02-10", f"{y}-02-16", "Medium"),
        ("Food", "20% Off", str((e - pd.Timedelta(days=14)).date()), str((e - pd.Timedelta(days=1)).date()), "Medium"),  # Easter
        ("Drinks", "Buy One Get One Free", f"{y}-06-15", f"{y}-06-30", "High"),  # summer kickoff
        ("Clothing", "20% Off", f"{y}-07-10", f"{y}-08-10", "Medium"),       # summer sale
        ("Toys", "20% Off", f"{y}-08-20", f"{y}-09-10", "Medium"),           # back to school
        ("Electronics", "Flash Sale", *BF_WEEK[y], "High"),                  # Black Friday week
        ("Toys", "Flash Sale", *BF_WEEK[y], "High"),
        ("Toys", "20% Off", f"{y}-12-01", f"{y}-12-23", "High"),             # December
        ("Food", "20% Off", f"{y}-12-10", f"{y}-12-24", "Medium"),
        # a few promotions that simply did not work (Low = no real uplift)
        ("Drinks", "20% Off", f"{y}-03-05", f"{y}-03-12", "Low"),
        ("Food", "Flash Sale", f"{y}-05-08", f"{y}-05-10", "Low"),
        ("Clothing", "Buy One Get One Free", f"{y}-10-05", f"{y}-10-12", "Low"),
        ("Electronics", "20% Off", f"{y}-04-14", f"{y}-04-20", "Low"),
    ]

CHANNELS = ["In-store", "Online", "Social Media"]
AUDIENCES = ["New Customers", "Returning Customers"]

promos = pd.DataFrame(promo_rows, columns=["category", "type", "start", "end", "label"])
promos["promotion_id"] = np.arange(101, 101 + len(promos))
promos["start_i"] = [DAYS.get_loc(s) for s in promos["start"]]
promos["end_i"] = [DAYS.get_loc(s) for s in promos["end"]]
promos["channel"] = rng.choice(CHANNELS, len(promos))
promos["audience"] = rng.choice(AUDIENCES, len(promos))

# Each promotion covers a random 60-85 % subset of its category's products
promo_products = []
for _, pr in promos.iterrows():
    idx = np.where(P_CAT == pr["category"])[0]
    take = max(2, int(len(idx) * rng.uniform(0.60, 0.85)))
    promo_products.append(set(rng.choice(idx, take, replace=False)))
# Guarantee every product appears in at least one promotion (the analyzer
# links products to promotions through the row tag)
covered = set().union(*promo_products)
for j in range(N_PRODUCTS):
    if j not in covered:
        for k, pr in promos.iterrows():
            if pr["category"] == P_CAT[j]:
                promo_products[k].add(j)
                break

# Day x category demand multiplier (seasonality + promotion uplift)
CAT_BASE = {"Food": 0.35, "Drinks": 0.22, "Clothing": 0.21, "Electronics": 0.11, "Toys": 0.11}
CAT_MONTH = {
    "Food":        [1.0, 1.0, 1.0, 1.05, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.05, 1.25],
    "Drinks":      [0.9, 0.9, 0.95, 1.0, 1.1, 1.3, 1.4, 1.35, 1.05, 0.95, 0.9, 1.0],
    "Clothing":    [1.2, 0.95, 1.1, 1.1, 1.0, 0.9, 1.05, 1.0, 1.15, 1.1, 0.95, 1.05],
    "Electronics": [0.95, 0.9, 0.9, 0.9, 0.95, 0.95, 0.9, 0.95, 1.0, 1.0, 1.6, 1.4],
    "Toys":        [0.8, 0.8, 0.85, 0.9, 0.9, 0.9, 0.95, 0.95, 1.0, 1.0, 1.4, 2.5],
}

cat_day_w = np.zeros((N_DAYS, len(CATEGORIES)))
for ci, cat in enumerate(CATEGORIES):
    cat_day_w[:, ci] = CAT_BASE[cat] * np.array(CAT_MONTH[cat])[DAYS.month - 1]
# category-level part of the promotion uplift
for k, pr in promos.iterrows():
    ci = CATEGORIES.index(pr["category"])
    sl = slice(pr["start_i"], pr["end_i"] + 1)
    cat_day_w[sl, ci] = np.maximum(cat_day_w[sl, ci], cat_day_w[sl, ci] * UPLIFT[pr["label"]])

# Day x product weight within each category (popularity + promo product boost)
prod_day_boost = np.ones((N_DAYS, N_PRODUCTS))
for k, pr in promos.iterrows():
    sl = slice(pr["start_i"], pr["end_i"] + 1)
    for j in promo_products[k]:
        prod_day_boost[sl, j] *= UPLIFT[pr["label"]]

# ---------------------------------------------------------------------------
# 5. Sample transactions: (day, store) -> category -> product
# ---------------------------------------------------------------------------

print("Sampling days/stores...")
flat = (store_day_w / store_day_w.sum()).ravel()
pick = rng.choice(len(flat), N_ROWS, p=flat)
day_idx = pick // len(STORES)
store_idx = pick % len(STORES)

print("Sampling categories...")
cw = cat_day_w / cat_day_w.sum(axis=1, keepdims=True)
cat_cum = np.cumsum(cw, axis=1)
r = rng.random(N_ROWS)
cat_idx = (r[:, None] > cat_cum[day_idx]).sum(axis=1)

print("Sampling products...")
prod_idx = np.empty(N_ROWS, dtype=np.int64)
for ci, cat in enumerate(CATEGORIES):
    rows = np.where(cat_idx == ci)[0]
    jdx = np.where(P_CAT == cat)[0]                    # products of this category
    w = P_POP[jdx][None, :] * prod_day_boost[:, jdx]   # [n_days, n_prod]
    cum = np.cumsum(w / w.sum(axis=1, keepdims=True), axis=1)
    rr = rng.random(len(rows))
    local = (rr[:, None] > cum[day_idx[rows]]).sum(axis=1)
    prod_idx[rows] = jdx[local]

# ---------------------------------------------------------------------------
# 6. Promotion tag per row + discounts + quantities + prices
# ---------------------------------------------------------------------------

print("Tagging promotions...")
# For every row: the promotion active for its product on its day, otherwise
# the nearest promotion covering that product (tag only, no discount).
promo_of_row = np.zeros(N_ROWS, dtype=np.int64)
promo_active = np.zeros(N_ROWS, dtype=bool)
starts = promos["start_i"].to_numpy()
ends = promos["end_i"].to_numpy()
mids = (starts + ends) / 2
for j in range(N_PRODUCTS):
    rows = np.where(prod_idx == j)[0]
    ks = np.array([k for k in range(len(promos)) if j in promo_products[k]])
    d = day_idx[rows]
    active = (d[:, None] >= starts[ks]) & (d[:, None] <= ends[ks])   # [rows, promos]
    has_active = active.any(axis=1)
    first_active = active.argmax(axis=1)
    nearest = np.abs(d[:, None] - mids[ks]).argmin(axis=1)
    chosen = np.where(has_active, first_active, nearest)
    promo_of_row[rows] = ks[chosen]
    promo_active[rows] = has_active

promo_type_row = promos["type"].to_numpy()[promo_of_row]
discount = np.where(promo_active, np.vectorize(DISCOUNT.get)(promo_type_row), 0.0)

print("Quantities and prices...")
# Typical basket-line quantities per category
quantity = np.ones(N_ROWS, dtype=np.int64)
geo_p = {"Food": 0.45, "Drinks": 0.35}
for ci, cat in enumerate(CATEGORIES):
    rows = cat_idx == ci
    n = rows.sum()
    if cat in geo_p:
        quantity[rows] = np.clip(rng.geometric(geo_p[cat], n), 1, 10)
    elif cat == "Clothing":
        quantity[rows] = 1 + (rng.random(n) < 0.25)
    elif cat == "Toys":
        quantity[rows] = 1 + (rng.random(n) < 0.15)
    else:  # Electronics
        quantity[rows] = 1 + (rng.random(n) < 0.05)
# "Buy One Get One Free" only makes sense with at least 2 items
bogo = promo_active & (promo_type_row == "Buy One Get One Free")
quantity[bogo] = np.maximum(quantity[bogo], 2)

# Price = catalog price x yearly inflation x small day-to-day noise
inflation = np.array([1.0, 1.035, 1.06])[DAYS.year[day_idx] - 2023]
unit_price = np.round(P_PRICE[prod_idx] * inflation * rng.normal(1.0, 0.015, N_ROWS), 2)
line_total = np.round(quantity * unit_price * (1 - discount), 2)

# ---------------------------------------------------------------------------
# 7. Customers
# ---------------------------------------------------------------------------

print("Building customers...")
cust_city_idx = rng.choice(4, N_CUSTOMERS, p=CITY_POP_W)
# Heavy-tailed activity: a few very frequent shoppers, many occasional ones
cust_activity = rng.lognormal(0.0, 1.0, N_CUSTOMERS)

age = np.concatenate([
    rng.integers(18, 31, int(N_CUSTOMERS * 0.25)),
    rng.integers(31, 51, int(N_CUSTOMERS * 0.40)),
    rng.integers(51, 71, int(N_CUSTOMERS * 0.27)),
    rng.integers(71, 86, N_CUSTOMERS - int(N_CUSTOMERS * 0.25) - int(N_CUSTOMERS * 0.40) - int(N_CUSTOMERS * 0.27)),
])
rng.shuffle(age)

gender = rng.choice(["Female", "Male", "Other"], N_CUSTOMERS, p=[0.505, 0.475, 0.02])

occupation = np.empty(N_CUSTOMERS, dtype=object)
young, mid, old = age < 27, (age >= 27) & (age < 64), age >= 64
occupation[young] = rng.choice(["Employed", "Self-Employed", "Unemployed"], young.sum(), p=[0.55, 0.10, 0.35])
occupation[mid] = rng.choice(["Employed", "Self-Employed", "Unemployed"], mid.sum(), p=[0.72, 0.18, 0.10])
occupation[old] = rng.choice(["Retired", "Employed", "Self-Employed"], old.sum(), p=[0.85, 0.10, 0.05])

INCOME_P = {"Employed": [0.20, 0.55, 0.25], "Self-Employed": [0.20, 0.45, 0.35],
            "Unemployed": [0.70, 0.28, 0.02], "Retired": [0.45, 0.45, 0.10]}
income = np.empty(N_CUSTOMERS, dtype=object)
for occ, p in INCOME_P.items():
    m = occupation == occ
    income[m] = rng.choice(["Low", "Medium", "High"], m.sum(), p=p)

education = np.where(age < 22,
                     rng.choice(["High School", "Bachelor's"], N_CUSTOMERS, p=[0.8, 0.2]),
                     rng.choice(["High School", "Bachelor's", "Master's", "PhD"], N_CUSTOMERS, p=[0.34, 0.36, 0.22, 0.08]))

marital = np.empty(N_CUSTOMERS, dtype=object)
a1, a2, a3 = age < 30, (age >= 30) & (age < 50), age >= 50
marital[a1] = rng.choice(["Single", "Married", "Divorced"], a1.sum(), p=[0.72, 0.25, 0.03])
marital[a2] = rng.choice(["Single", "Married", "Divorced"], a2.sum(), p=[0.25, 0.60, 0.15])
marital[a3] = rng.choice(["Single", "Married", "Divorced"], a3.sum(), p=[0.12, 0.62, 0.26])

children = np.zeros(N_CUSTOMERS, dtype=np.int64)
fam = (marital != "Single") & (age >= 25)
children[fam] = rng.choice([0, 1, 2, 3, 4], fam.sum(), p=[0.22, 0.28, 0.35, 0.12, 0.03])

loyalty = rng.choice(["Yes", "No"], N_CUSTOMERS, p=[0.38, 0.62])
membership_years = np.where(loyalty == "Yes", rng.integers(1, 9, N_CUSTOMERS), 0)
email_sub = np.where(loyalty == "Yes",
                     rng.choice(["Yes", "No"], N_CUSTOMERS, p=[0.70, 0.30]),
                     rng.choice(["Yes", "No"], N_CUSTOMERS, p=[0.25, 0.75]))

def by_age_level(p_young, p_mid, p_old):
    out = np.empty(N_CUSTOMERS, dtype=object)
    m1, m2, m3 = age < 35, (age >= 35) & (age < 60), age >= 60
    out[m1] = rng.choice(["High", "Medium", "Low"], m1.sum(), p=p_young)
    out[m2] = rng.choice(["High", "Medium", "Low"], m2.sum(), p=p_mid)
    out[m3] = rng.choice(["High", "Medium", "Low"], m3.sum(), p=p_old)
    return out

app_usage = by_age_level([0.50, 0.35, 0.15], [0.25, 0.45, 0.30], [0.08, 0.27, 0.65])
social = by_age_level([0.55, 0.33, 0.12], [0.25, 0.45, 0.30], [0.10, 0.25, 0.65])
website_visits = np.clip(rng.lognormal(2.5, 1.0, N_CUSTOMERS), 0, 200).astype(int)
support_calls = rng.poisson(0.8, N_CUSTOMERS)

# ---------------------------------------------------------------------------
# 8. Assign customers to transactions (mostly locals of the store's city)
# ---------------------------------------------------------------------------

print("Assigning customers...")
store_city_idx = np.array([CITIES.index(c) for c in STORE_CITY])[store_idx]
# 85 % of shoppers are from the store's own city
row_city = store_city_idx.copy()
away = rng.random(N_ROWS) >= 0.85
other = rng.choice(4, N_ROWS, p=CITY_POP_W)
row_city[away] = other[away]

customer_of_row = np.empty(N_ROWS, dtype=np.int64)
for ci in range(4):
    pool = np.where(cust_city_idx == ci)[0]
    p = cust_activity[pool] / cust_activity[pool].sum()
    m = row_city == ci
    customer_of_row[m] = rng.choice(pool, m.sum(), p=p)

# ---------------------------------------------------------------------------
# 9. Timestamps
# ---------------------------------------------------------------------------

print("Timestamps...")
HOURS = np.arange(8, 21)  # shops open 08:00-21:00
HOUR_W_WEEKDAY = np.array([3, 5, 7, 8, 8, 8, 7, 7, 9, 11, 12, 10, 5], dtype=float)
HOUR_W_SATURDAY = np.array([4, 7, 10, 11, 10, 8, 7, 6, 8, 9, 9, 7, 4], dtype=float)
is_sat = DAYS.dayofweek[day_idx] == 5
hour = np.where(
    is_sat,
    rng.choice(HOURS, N_ROWS, p=HOUR_W_SATURDAY / HOUR_W_SATURDAY.sum()),
    rng.choice(HOURS, N_ROWS, p=HOUR_W_WEEKDAY / HOUR_W_WEEKDAY.sum()),
)
tx_time = (DAYS.values[day_idx]
           + hour * np.timedelta64(3600, "s")
           + rng.integers(0, 3600, N_ROWS) * np.timedelta64(1, "s"))
tx_time = pd.DatetimeIndex(tx_time)

# ---------------------------------------------------------------------------
# 10. Returns (needed for per-customer return aggregates)
# ---------------------------------------------------------------------------

returned = rng.random(N_ROWS) < P_RETURN[prod_idx]

# ---------------------------------------------------------------------------
# 11. Per-customer aggregates computed from the ACTUAL transactions
# ---------------------------------------------------------------------------

print("Computing customer aggregates...")
tx = pd.DataFrame({
    "customer_id": customer_of_row,
    "day": tx_time.normalize(),
    "line_total": line_total,
    "quantity": quantity,
    "discount": discount,
    "discount_value": np.round(quantity * unit_price * discount, 2),
    "category": pd.Categorical.from_codes(cat_idx, CATEGORIES),
    "store": pd.Categorical.from_codes(store_idx, STORES),
    "returned_items": np.where(returned, quantity, 0),
    "returned_value": np.where(returned, line_total, 0.0),
})

g = tx.groupby("customer_id")
agg = g.agg(
    n_tx=("line_total", "size"),
    total_spent=("line_total", "sum"),
    total_items=("quantity", "sum"),
    avg_items=("quantity", "mean"),
    avg_tx_value=("line_total", "mean"),
    avg_discount=("discount", "mean"),
    total_discounts=("discount_value", "sum"),
    last_purchase=("day", "max"),
    n_categories=("category", "nunique"),
    returned_items=("returned_items", "sum"),
    returned_value=("returned_value", "sum"),
)
# "Purchase" = one shopping visit (one customer-day, possibly several lines)
visits = tx.groupby(["customer_id", "day"], observed=True)["line_total"].sum()
vg = visits.groupby("customer_id")
agg["avg_visit_value"] = vg.mean()
agg["max_visit_value"] = vg.max()
agg["min_visit_value"] = vg.min()
agg["preferred_store"] = g["store"].agg(lambda s: s.value_counts().idxmax())

agg["days_since_last"] = (DATA_END - agg["last_purchase"]).dt.days
agg["churned"] = np.where(agg["days_since_last"] > 180, "Yes", "No")
# transactions per week over the 3 years -> frequency label
per_week = agg["n_tx"] / (N_DAYS / 7)
agg["purchase_frequency"] = np.select(
    [per_week >= 4, per_week >= 1, agg["n_tx"] >= 28], ["Daily", "Weekly", "Monthly"], default="Yearly")
agg["avg_spent_per_category"] = agg["total_spent"] / agg["n_categories"]
# The chain's webshop is not part of this dataset; online purchase counts are
# a plausible extra channel proportional to in-store activity and app usage
app_high = pd.Series(app_usage).reindex(agg.index).eq("High").to_numpy()
agg["online_purchases"] = (agg["n_tx"] * rng.uniform(0.0, 0.30, len(agg)) * np.where(app_high, 1.6, 1.0)).astype(int)

# ---------------------------------------------------------------------------
# 12. Assemble the final frame (exact same 76 columns as the old dataset)
# ---------------------------------------------------------------------------

print("Assembling final dataframe...")
cid = customer_of_row
A = agg  # shorthand; indexed by customer_id

store_city_arr = np.array(STORE_CITY)[store_idx]
cust_city_arr = np.array(CITIES)[cust_city_idx[cid]]

# distance home city <-> store city (in-city trips are short)
same_city = cust_city_arr == store_city_arr
base_km = CITY_KM_MATRIX[cust_city_idx[cid], np.array([CITIES.index(c) for c in STORE_CITY])[store_idx]]
distance = np.where(
    same_city,
    np.clip(rng.lognormal(np.log(4.0), 0.6, N_ROWS), 0.5, 15),
    np.clip(base_km + rng.normal(0, 5, N_ROWS), 20, 260),
).round(2)

season = np.array(["Winter", "Winter", "Spring", "Spring", "Spring", "Summer",
                   "Summer", "Summer", "Fall", "Fall", "Fall", "Winter"])[tx_time.month - 1]
holiday_season = np.where((tx_time.month == 12) | ((tx_time.month == 11) & (tx_time.day >= 15)), "Yes", "No")

# Payment habits differ by age (younger -> more mobile payments)
young_row = age[cid] < 40
payment = np.where(
    young_row,
    rng.choice(["Credit Card", "Debit Card", "Cash", "Mobile Payment"], N_ROWS, p=[0.38, 0.22, 0.16, 0.24]),
    rng.choice(["Credit Card", "Debit Card", "Cash", "Mobile Payment"], N_ROWS, p=[0.40, 0.24, 0.30, 0.06]),
)

# Product batch dates: made before the sale, expires shelf-life later
made_ago = (rng.uniform(0.05, 0.5, N_ROWS) * P_SHELF[prod_idx]).astype("timedelta64[D]")
manufacture = tx_time.normalize().values - made_ago
expiry = manufacture + P_SHELF[prod_idx].astype("timedelta64[D]")

promo_sel = promos.iloc[promo_of_row]

df = pd.DataFrame({
    "customer_id": cid + 1,
    "age": age[cid],
    "gender": pd.Categorical(gender[cid]),
    "income_bracket": pd.Categorical(income[cid]),
    "loyalty_program": pd.Categorical(loyalty[cid]),
    "membership_years": membership_years[cid],
    "churned": pd.Categorical(A["churned"].to_numpy()[np.searchsorted(A.index, cid)]),
    "marital_status": pd.Categorical(marital[cid]),
    "number_of_children": children[cid],
    "education_level": pd.Categorical(education[cid]),
    "occupation": pd.Categorical(occupation[cid]),
    "transaction_id": 0,  # filled after sorting by date
    "transaction_date": tx_time,
    "product_id": P_ID[prod_idx],
    "product_category": pd.Categorical.from_codes(cat_idx, CATEGORIES),
    "quantity": quantity,
    "unit_price": unit_price,
    "discount_applied": discount,
    "payment_method": pd.Categorical(payment),
    "store_location": pd.Categorical.from_codes(store_idx, STORES),
    "transaction_hour": hour,
    "day_of_week": pd.Categorical(tx_time.day_name()),
    "week_of_year": tx_time.isocalendar().week.to_numpy(),
    "month_of_year": tx_time.month,
    "avg_purchase_value": np.round(A["avg_visit_value"].to_numpy()[np.searchsorted(A.index, cid)], 2),
    "purchase_frequency": pd.Categorical(A["purchase_frequency"].to_numpy()[np.searchsorted(A.index, cid)]),
    "last_purchase_date": A["last_purchase"].to_numpy()[np.searchsorted(A.index, cid)],
    "avg_discount_used": np.round(A["avg_discount"].to_numpy()[np.searchsorted(A.index, cid)], 2),
    "preferred_store": pd.Categorical(A["preferred_store"].to_numpy()[np.searchsorted(A.index, cid)]),
    "online_purchases": A["online_purchases"].to_numpy()[np.searchsorted(A.index, cid)],
    "in_store_purchases": A["n_tx"].to_numpy()[np.searchsorted(A.index, cid)],
    "avg_items_per_transaction": np.round(A["avg_items"].to_numpy()[np.searchsorted(A.index, cid)], 2),
    "avg_transaction_value": np.round(A["avg_tx_value"].to_numpy()[np.searchsorted(A.index, cid)], 2),
    "total_returned_items": A["returned_items"].to_numpy()[np.searchsorted(A.index, cid)],
    "total_returned_value": np.round(A["returned_value"].to_numpy()[np.searchsorted(A.index, cid)], 2),
    "total_sales": line_total,
    "total_transactions": A["n_tx"].to_numpy()[np.searchsorted(A.index, cid)],
    "total_items_purchased": A["total_items"].to_numpy()[np.searchsorted(A.index, cid)],
    "total_discounts_received": np.round(A["total_discounts"].to_numpy()[np.searchsorted(A.index, cid)], 2),
    "avg_spent_per_category": np.round(A["avg_spent_per_category"].to_numpy()[np.searchsorted(A.index, cid)], 2),
    "max_single_purchase_value": np.round(A["max_visit_value"].to_numpy()[np.searchsorted(A.index, cid)], 2),
    "min_single_purchase_value": np.round(A["min_visit_value"].to_numpy()[np.searchsorted(A.index, cid)], 2),
    "product_name": pd.Categorical.from_codes(prod_idx, list(P_NAME)),
    "product_brand": pd.Categorical(P_BRAND[prod_idx]),
    "product_rating": P_RATING[prod_idx],
    "product_review_count": P_REVIEWS[prod_idx],
    "product_stock": rng.integers(10, 600, N_ROWS),
    "product_return_rate": P_RETURN[prod_idx],
    "product_size": pd.Categorical(P_SIZE[prod_idx]),
    "product_weight": P_WEIGHT[prod_idx],
    "product_color": pd.Categorical(P_COLOR[prod_idx]),
    "product_material": pd.Categorical(P_MATERIAL[prod_idx]),
    "product_manufacture_date": manufacture,
    "product_expiry_date": expiry,
    "product_shelf_life": P_SHELF[prod_idx],
    "promotion_id": promo_sel["promotion_id"].to_numpy(),
    "promotion_type": pd.Categorical(promo_sel["type"].to_numpy()),
    "promotion_start_date": pd.to_datetime(promo_sel["start"].to_numpy()),
    "promotion_end_date": pd.to_datetime(promo_sel["end"].to_numpy()),
    "promotion_effectiveness": pd.Categorical(promo_sel["label"].to_numpy()),
    "promotion_channel": pd.Categorical(promo_sel["channel"].to_numpy()),
    "promotion_target_audience": pd.Categorical(promo_sel["audience"].to_numpy()),
    "customer_zip_code": np.vectorize(CITY_ZIP.get)(cust_city_arr),
    "customer_city": pd.Categorical(cust_city_arr),
    "customer_state": pd.Categorical(np.vectorize(CITY_STATE.get)(cust_city_arr)),
    "store_zip_code": np.vectorize(CITY_ZIP.get)(store_city_arr),
    "store_city": pd.Categorical(store_city_arr),
    "store_state": pd.Categorical(np.vectorize(CITY_STATE.get)(store_city_arr)),
    "distance_to_store": distance,
    "holiday_season": pd.Categorical(holiday_season),
    "season": pd.Categorical(season),
    "weekend": pd.Categorical(np.where(tx_time.dayofweek >= 5, "Yes", "No")),
    "customer_support_calls": support_calls[cid],
    "email_subscriptions": pd.Categorical(email_sub[cid]),
    "app_usage": pd.Categorical(app_usage[cid]),
    "website_visits": website_visits[cid],
    "social_media_engagement": pd.Categorical(social[cid]),
    "days_since_last_purchase": A["days_since_last"].to_numpy()[np.searchsorted(A.index, cid)],
})

df = df.sort_values("transaction_date", kind="stable").reset_index(drop=True)
df["transaction_id"] = np.arange(1, len(df) + 1)

print(f"Writing {len(df):,} rows to {OUT_PATH} ...")
df.to_csv(OUT_PATH, index=False, encoding="utf-8")
print("Done.")