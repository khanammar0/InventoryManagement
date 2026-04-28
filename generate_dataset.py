import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# ── Seed for reproducibility ──────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

# ── Master data ───────────────────────────────────────────────────────────────
products = {
    "Furniture": [
        "Bookcases", "Chairs", "Furnishings", "Tables"
    ],
    "Office Supplies": [
        "Appliances", "Art", "Binders", "Envelopes",
        "Fasteners", "Labels", "Paper", "Storage", "Supplies"
    ],
    "Technology": [
        "Accessories", "Copiers", "Machines", "Phones"
    ]
}

product_names = {
    "Bookcases":    ["Bush Bookcase", "Sauder Bookcase", "Atlantic Bookcase"],
    "Chairs":       ["Hon Chair", "Global Chair", "Ergonomic Chair"],
    "Furnishings":  ["Eldon Shelf", "Tensor Lamp", "Howard Clock"],
    "Tables":       ["Bevis Table", "Chromcraft Table", "Bretford Table"],
    "Appliances":   ["Holmes Air Purifier", "Belkin Surge Protector", "Hoover Fan"],
    "Art":          ["Avery Marker Set", "Faber Pencil Set", "Dixon Crayons"],
    "Binders":      ["Avery Binder", "Cardinal Binder", "Ibico Binder"],
    "Envelopes":    ["Kraft Envelope Pack", "Poly Mailer Set", "Business Envelopes"],
    "Fasteners":    ["Swingline Stapler", "Acme Staple Pack", "Bostitch Stapler"],
    "Labels":       ["Avery Labels", "3M Labels", "Maco Labels"],
    "Paper":        ["Xerox Paper Ream", "Hammermill Paper", "HP Printer Paper"],
    "Storage":      ["Fellowes Storage Box", "Storex Box", "Iris Box"],
    "Supplies":     ["Kleenex Tissues", "Scotch Tape Set", "Sharpie Set"],
    "Accessories":  ["Logitech Mouse", "Belkin USB Hub", "Kingston USB Drive"],
    "Copiers":      ["Canon Copier", "Hewlett-Packard Copier", "Samsung Copier"],
    "Machines":     ["Sharp Calculator", "Lexmark Printer", "Zebra Printer"],
    "Phones":       ["Cisco Phone", "Motorola Phone", "Samsung Galaxy Tab"]
}

regions      = ["East", "West", "Central", "South"]
ship_modes   = ["First Class", "Second Class", "Standard Class", "Same Day"]
segments     = ["Consumer", "Corporate", "Home Office"]
states_by_region = {
    "East":    ["New York", "Pennsylvania", "Ohio", "Virginia"],
    "West":    ["California", "Washington", "Oregon", "Nevada"],
    "Central": ["Texas", "Illinois", "Michigan", "Minnesota"],
    "South":   ["Florida", "Georgia", "North Carolina", "Tennessee"]
}

price_range = {
    "Furniture":       (200, 1500),
    "Office Supplies": (5,   300),
    "Technology":      (50,  1800)
}

# ── Row generator ─────────────────────────────────────────────────────────────
rows = []
order_counter = 10000
customer_ids  = [f"CG-{random.randint(10000,99999)}" for _ in range(300)]
customer_names = [
    "Aaron Hawk", "Ben Carter", "Claire Young", "David Smith",
    "Eva Green", "Frank Lee", "Grace Kim", "Henry Brown",
    "Iris Chan", "Jake Wilson", "Kara Patel", "Liam Johnson",
    "Mia Davis", "Noah Martinez", "Olivia White", "Paul Thomas"
]

start_date = datetime(2021, 1, 1)
end_date   = datetime(2023, 12, 31)

for _ in range(9994):
    # order basics
    order_date  = start_date + timedelta(days=random.randint(0, (end_date - start_date).days))
    ship_days   = random.randint(2, 7)
    ship_date   = order_date + timedelta(days=ship_days)

    region      = random.choice(regions)
    state       = random.choice(states_by_region[region])
    segment     = random.choice(segments)
    ship_mode   = random.choice(ship_modes)
    customer_id = random.choice(customer_ids)
    customer    = random.choice(customer_names)

    # product
    category    = random.choice(list(products.keys()))
    sub_cat     = random.choice(products[category])
    prod_name   = random.choice(product_names[sub_cat])
    prod_id     = f"PRD-{abs(hash(prod_name)) % 90000 + 10000}"

    # financials
    lo, hi      = price_range[category]
    unit_price  = round(random.uniform(lo, hi), 2)
    quantity    = random.randint(1, 14)
    discount    = random.choice([0, 0, 0, 0.1, 0.2, 0.3, 0.5])
    sales       = round(unit_price * quantity * (1 - discount), 2)
    profit      = round(sales * random.uniform(0.05, 0.35), 2)

    # inventory columns (our AI additions)
    stock_level   = random.randint(5, 500)
    reorder_point = random.randint(20, 80)
    lead_time     = random.randint(2, 14)   # days

    order_id = f"CA-{order_date.year}-{order_counter}"
    order_counter += 1

    rows.append({
        "Order ID":      order_id,
        "Order Date":    order_date.strftime("%Y-%m-%d"),
        "Ship Date":     ship_date.strftime("%Y-%m-%d"),
        "Ship Mode":     ship_mode,
        "Customer ID":   customer_id,
        "Customer Name": customer,
        "Segment":       segment,
        "Region":        region,
        "State":         state,
        "Category":      category,
        "Sub-Category":  sub_cat,
        "Product ID":    prod_id,
        "Product Name":  prod_name,
        "Sales":         sales,
        "Quantity":      quantity,
        "Discount":      discount,
        "Profit":        profit,
        "Stock Level":   stock_level,
        "Reorder Point": reorder_point,
        "Lead Time Days":lead_time
    })

df = pd.DataFrame(rows)
df.to_csv("superstore_inventory.csv", index=False)
print(f"✅ Dataset created: {len(df)} rows, {len(df.columns)} columns")
print(df.head(3).to_string())
