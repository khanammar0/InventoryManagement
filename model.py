# =============================================================================
# model.py  –  AI Inventory Management Core
# B.Tech 2nd Year Project
# Models used:
#   1. Linear Regression  → Predict future demand (next 30 days)
#   2. Rule-based AI      → Low stock alerts
#   3. EOQ Formula        → Reorder quantity suggestion
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOAD & CLEAN DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_data(filepath="superstore_inventory.csv"):
    """Load the dataset and do basic cleaning."""
    df = pd.read_csv(filepath, parse_dates=["Order Date", "Ship Date"])
    df.dropna(inplace=True)
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    df["Month"]      = df["Order Date"].dt.month
    df["Year"]       = df["Order Date"].dt.year
    df["DayOfWeek"]  = df["Order Date"].dt.dayofweek
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2.  DEMAND FORECASTING  (Linear Regression)
# ─────────────────────────────────────────────────────────────────────────────

def train_demand_model(df):
    """
    Trains a Linear Regression model to predict Quantity (demand)
    based on Month, DayOfWeek, Discount, and Category.
    Returns: trained model, label encoder, MAE, R2 score
    """

    # Encode 'Category' as numbers (ML needs numbers, not text)
    le = LabelEncoder()
    df["Category_enc"] = le.fit_transform(df["Category"])

    # Features (inputs) and Target (output)
    features = ["Month", "DayOfWeek", "Discount", "Category_enc", "Year"]
    target   = "Quantity"

    X = df[features]
    y = df[target]

    # Simple 80/20 train-test split (no shuffle — respects time order)
    split    = int(len(df) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mae    = round(mean_absolute_error(y_test, y_pred), 2)
    r2     = round(r2_score(y_test, y_pred), 4)

    return model, le, mae, r2, features


def predict_next_30_days(model, le, category, features):
    """
    Predict demand for a selected category for the next 30 days.
    Returns a DataFrame with Date and Predicted Quantity.
    """
    future_dates = pd.date_range(start=pd.Timestamp.today(), periods=30)
    cat_enc      = le.transform([category])[0]

    records = []
    for date in future_dates:
        row = {
            "Month":        date.month,
            "DayOfWeek":    date.dayofweek,
            "Discount":     0.0,           # assume no discount
            "Category_enc": cat_enc,
            "Year":         date.year
        }
        records.append(row)

    future_df           = pd.DataFrame(records)[features]
    preds               = model.predict(future_df)
    preds               = np.clip(np.round(preds).astype(int), 1, None)

    result              = pd.DataFrame()
    result["Date"]      = future_dates.strftime("%Y-%m-%d")
    result["Predicted Demand"] = preds
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 3.  LOW STOCK ALERT  (Rule-based AI)
# ─────────────────────────────────────────────────────────────────────────────

def get_low_stock_alerts(df):
    """
    Rule: If Stock Level <= Reorder Point → flag as LOW STOCK.
    Returns a filtered DataFrame of products that need attention.
    """
    # Get latest stock snapshot per product
    latest = (
        df.sort_values("Order Date")
          .groupby("Product Name")
          .last()
          .reset_index()
    )

    low_stock = latest[latest["Stock Level"] <= latest["Reorder Point"]].copy()
    low_stock = low_stock[[
        "Product Name", "Category", "Sub-Category",
        "Stock Level", "Reorder Point", "Lead Time Days"
    ]].sort_values("Stock Level")

    low_stock["Status"] = low_stock.apply(
        lambda r: "🔴 Critical" if r["Stock Level"] <= r["Reorder Point"] * 0.5
                  else "🟡 Low",
        axis=1
    )
    return low_stock.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  REORDER SUGGESTION  (EOQ Formula)
# ─────────────────────────────────────────────────────────────────────────────

def get_reorder_suggestions(df, ordering_cost=50, holding_cost_pct=0.2):
    """
    Economic Order Quantity (EOQ) formula:
        EOQ = sqrt( (2 * Annual Demand * Ordering Cost) / Holding Cost )

    Reorder Point:
        ROP = Average Daily Demand * Lead Time + Safety Stock

    Returns a DataFrame with reorder recommendations.
    """
    # Annual demand per product (total quantity sold)
    demand_df = (
        df.groupby(["Product Name", "Category"])
          .agg(
              Annual_Demand  = ("Quantity",      "sum"),
              Avg_Price      = ("Sales",         "mean"),
              Avg_Lead_Time  = ("Lead Time Days","mean"),
              Avg_Stock      = ("Stock Level",   "mean"),
              Reorder_Point  = ("Reorder Point", "mean")
          )
          .reset_index()
    )

    demand_df["Avg_Lead_Time"]  = demand_df["Avg_Lead_Time"].round(1)
    demand_df["Avg_Daily_Demand"] = (demand_df["Annual_Demand"] / 365).round(2)

    # Holding cost per unit per year
    demand_df["Holding_Cost"] = (demand_df["Avg_Price"] * holding_cost_pct).round(2)
    demand_df["Holding_Cost"] = demand_df["Holding_Cost"].clip(lower=1)   # min ₹1

    # EOQ
    demand_df["EOQ"] = np.sqrt(
        (2 * demand_df["Annual_Demand"] * ordering_cost)
        / demand_df["Holding_Cost"]
    ).round(0).astype(int)

    # Safety stock = 2 * avg lead time * avg daily demand
    demand_df["Safety_Stock"] = (
        2 * demand_df["Avg_Lead_Time"] * demand_df["Avg_Daily_Demand"]
    ).round(0).astype(int)

    # Reorder Point
    demand_df["Reorder_Point_Calc"] = (
        demand_df["Avg_Daily_Demand"] * demand_df["Avg_Lead_Time"]
        + demand_df["Safety_Stock"]
    ).round(0).astype(int)

    # Should reorder now?
    demand_df["Reorder Now?"] = demand_df.apply(
        lambda r: "✅ Yes" if r["Avg_Stock"] <= r["Reorder_Point_Calc"] else "⏳ No",
        axis=1
    )

    result = demand_df[[
        "Product Name", "Category", "Annual_Demand",
        "EOQ", "Safety_Stock", "Reorder_Point_Calc",
        "Avg_Stock", "Avg_Lead_Time", "Reorder Now?"
    ]].rename(columns={
        "Annual_Demand":      "Annual Demand",
        "EOQ":                "Order Qty (EOQ)",
        "Safety_Stock":       "Safety Stock",
        "Reorder_Point_Calc": "Reorder Point",
        "Avg_Stock":          "Current Avg Stock",
        "Avg_Lead_Time":      "Lead Time (days)"
    })

    return result.sort_values("Reorder Now?", ascending=True).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  INVENTORY SUMMARY STATS
# ─────────────────────────────────────────────────────────────────────────────

def get_summary(df):
    """Returns key KPI numbers for the dashboard home page."""
    total_products  = df["Product Name"].nunique()
    total_orders    = df["Order ID"].nunique()
    total_revenue   = round(df["Sales"].sum(), 2)
    total_profit    = round(df["Profit"].sum(), 2)
    avg_stock       = round(df["Stock Level"].mean(), 1)
    low_stock_count = len(get_low_stock_alerts(df))

    return {
        "Total Products":    total_products,
        "Total Orders":      total_orders,
        "Total Revenue (₹)": f"₹{total_revenue:,.2f}",
        "Total Profit (₹)":  f"₹{total_profit:,.2f}",
        "Avg Stock Level":   avg_stock,
        "Low Stock Items":   low_stock_count
    }


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST  (run: python model.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading data...")
    df = load_data()

    print("\n── Summary ──────────────────────────────")
    for k, v in get_summary(df).items():
        print(f"  {k}: {v}")

    print("\n── Training Demand Model ────────────────")
    model, le, mae, r2, features = train_demand_model(df)
    print(f"  MAE : {mae}   (avg prediction error in units)")
    print(f"  R²  : {r2}    (1.0 = perfect, >0.5 = good for inventory)")

    print("\n── Next 30-Day Forecast (Technology) ───")
    forecast = predict_next_30_days(model, le, "Technology", features)
    print(forecast.head(7).to_string(index=False))

    print("\n── Low Stock Alerts ─────────────────────")
    alerts = get_low_stock_alerts(df)
    print(f"  {len(alerts)} products need attention")
    print(alerts.head(5).to_string(index=False))

    print("\n── Reorder Suggestions (top 5) ──────────")
    reorder = get_reorder_suggestions(df)
    print(reorder.head(5).to_string(index=False))

    print("\n✅ model.py working correctly!")
