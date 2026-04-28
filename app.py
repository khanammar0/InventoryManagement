# =============================================================================
# app.py  –  Streamlit Dashboard for AI Inventory Management
# Run with:  streamlit run app.py
# =============================================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from model import (
    load_data,
    train_demand_model,
    predict_next_30_days,
    get_low_stock_alerts,
    get_reorder_suggestions,
    get_summary
)
from groq_assistant import ask_groq

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Inventory Manager",
    page_icon="📦",
    layout="wide"
)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA & TRAIN MODEL  (cached so it runs once)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def get_data():
    return load_data("superstore_inventory.csv")

@st.cache_resource
def get_model(df):
    return train_demand_model(df)

df                        = get_data()
model, le, mae, r2, feats = get_model(df)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3144/3144456.png", width=80)
st.sidebar.title("📦 Inventory AI")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home Dashboard",
     "📈 Demand Forecast",
     "🚨 Low Stock Alerts",
     "📦 Reorder Suggestions",
     "🗂️ Full Inventory",
     "🤖 AI Assistant"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Model Info**")
st.sidebar.metric("MAE",  f"{mae} units")
st.sidebar.metric("R² Score", str(r2))
st.sidebar.markdown("*Linear Regression Model*")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — HOME DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠 Home Dashboard":
    st.title("🏠 AI Inventory Management Dashboard")
    st.markdown("#### Superstore Inventory — AI-Powered Overview")
    st.markdown("---")

    # KPI cards
    summary = get_summary(df)
    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)

    col1.metric("📦 Total Products",    summary["Total Products"])
    col2.metric("🛒 Total Orders",      summary["Total Orders"])
    col3.metric("💰 Total Revenue",     summary["Total Revenue (₹)"])
    col4.metric("📈 Total Profit",      summary["Total Profit (₹)"])
    col5.metric("📊 Avg Stock Level",   summary["Avg Stock Level"])
    col6.metric("🚨 Low Stock Items",   summary["Low Stock Items"],
                delta="Need Reorder", delta_color="inverse")

    st.markdown("---")

    # Charts side by side
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("📊 Sales by Category")
        cat_sales = df.groupby("Category")["Sales"].sum().reset_index()
        fig = px.pie(cat_sales, names="Category", values="Sales",
                     color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("📅 Monthly Order Trend")
        df["Month-Year"] = df["Order Date"].dt.to_period("M").astype(str)
        monthly = df.groupby("Month-Year")["Quantity"].sum().reset_index()
        monthly = monthly.sort_values("Month-Year")
        fig2 = px.line(monthly, x="Month-Year", y="Quantity",
                       markers=True, color_discrete_sequence=["#FF6B6B"])
        fig2.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # Top 10 products by quantity sold
    st.subheader("🏆 Top 10 Products by Quantity Sold")
    top10 = (df.groupby("Product Name")["Quantity"]
               .sum()
               .sort_values(ascending=False)
               .head(10)
               .reset_index())
    fig3 = px.bar(top10, x="Quantity", y="Product Name",
                  orientation="h",
                  color="Quantity",
                  color_continuous_scale="Blues")
    fig3.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig3, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — DEMAND FORECAST
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📈 Demand Forecast":
    st.title("📈 AI Demand Forecasting")
    st.markdown("Uses **Linear Regression** to predict product demand for the next 30 days.")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        category = st.selectbox("Select Category", df["Category"].unique())
    with col2:
        st.info(f"Model MAE: **{mae} units** | R²: **{r2}**")

    if st.button("🔮 Generate Forecast"):
        with st.spinner("Predicting demand..."):
            forecast_df = predict_next_30_days(model, le, category, feats)

        st.success(f"✅ 30-day forecast for **{category}** generated!")

        # Line chart
        fig = px.line(
            forecast_df, x="Date", y="Predicted Demand",
            title=f"Predicted Daily Demand — {category} (Next 30 Days)",
            markers=True,
            color_discrete_sequence=["#4ECDC4"]
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

        # Summary stats
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("📦 Total Predicted",  int(forecast_df["Predicted Demand"].sum()))
        col_b.metric("📊 Daily Average",    int(forecast_df["Predicted Demand"].mean()))
        col_c.metric("🔺 Peak Day Demand",  int(forecast_df["Predicted Demand"].max()))

        st.markdown("#### 📋 Day-by-Day Forecast Table")
        st.dataframe(forecast_df, use_container_width=True)

    # Historical demand chart
    st.markdown("---")
    st.subheader("📜 Historical Demand (Past Data)")
    hist = (df[df["Category"] == category]
              .groupby("Order Date")["Quantity"]
              .sum()
              .reset_index()
              .rename(columns={"Order Date": "Date", "Quantity": "Actual Demand"}))
    fig_hist = px.area(hist, x="Date", y="Actual Demand",
                       color_discrete_sequence=["#A8E6CF"])
    st.plotly_chart(fig_hist, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — LOW STOCK ALERTS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🚨 Low Stock Alerts":
    st.title("🚨 Low Stock Alerts")
    st.markdown("Products where **Stock Level ≤ Reorder Point**. Immediate action needed.")
    st.markdown("---")

    alerts = get_low_stock_alerts(df)

    if len(alerts) == 0:
        st.success("✅ All products are well stocked!")
    else:
        critical = alerts[alerts["Status"] == "🔴 Critical"]
        low      = alerts[alerts["Status"] == "🟡 Low"]

        col1, col2 = st.columns(2)
        col1.metric("🔴 Critical Items", len(critical))
        col2.metric("🟡 Low Items",      len(low))

        st.markdown("---")
        st.subheader("🔴 Critical Stock Items")
        if len(critical):
            st.dataframe(critical, use_container_width=True)
        else:
            st.info("No critical items.")

        st.subheader("🟡 Low Stock Items")
        if len(low):
            st.dataframe(low, use_container_width=True)
        else:
            st.info("No low stock items.")

        # Bar chart
        st.markdown("---")
        st.subheader("📊 Stock Level vs Reorder Point")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Stock Level",
            x=alerts["Product Name"],
            y=alerts["Stock Level"],
            marker_color="#4ECDC4"
        ))
        fig.add_trace(go.Bar(
            name="Reorder Point",
            x=alerts["Product Name"],
            y=alerts["Reorder Point"],
            marker_color="#FF6B6B"
        ))
        fig.update_layout(barmode="group", xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 — REORDER SUGGESTIONS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📦 Reorder Suggestions":
    st.title("📦 AI Reorder Suggestions")
    st.markdown("""
    Uses the **EOQ (Economic Order Quantity)** formula:

    > **EOQ = √ ( 2 × Annual Demand × Ordering Cost ) / Holding Cost**

    This tells you the *optimal* quantity to order to minimize total cost.
    """)
    st.markdown("---")

    reorder_df = get_reorder_suggestions(df)

    # Filter: show only items that need reorder now
    show_all = st.checkbox("Show all products (not just reorder-needed)", value=False)

    if not show_all:
        display_df = reorder_df[reorder_df["Reorder Now?"] == "✅ Yes"]
    else:
        display_df = reorder_df

    st.metric("📦 Products Needing Reorder",
              len(reorder_df[reorder_df["Reorder Now?"] == "✅ Yes"]))

    st.dataframe(display_df, use_container_width=True)

    st.markdown("---")
    st.subheader("📊 EOQ vs Current Stock")
    top = reorder_df.head(15)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="EOQ (Order Qty)",
        x=top["Product Name"],
        y=top["Order Qty (EOQ)"],
        marker_color="#6C5CE7"
    ))
    fig.add_trace(go.Bar(
        name="Current Avg Stock",
        x=top["Product Name"],
        y=top["Current Avg Stock"],
        marker_color="#FDCB6E"
    ))
    fig.update_layout(barmode="group", xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5 — FULL INVENTORY TABLE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🗂️ Full Inventory":
    st.title("🗂️ Full Inventory Table")
    st.markdown("Browse, filter, and search the complete inventory dataset.")
    st.markdown("---")

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        cat_filter = st.multiselect("Category", df["Category"].unique(),
                                    default=list(df["Category"].unique()))
    with col2:
        reg_filter = st.multiselect("Region", df["Region"].unique(),
                                    default=list(df["Region"].unique()))
    with col3:
        seg_filter = st.multiselect("Segment", df["Segment"].unique(),
                                    default=list(df["Segment"].unique()))

    filtered = df[
        df["Category"].isin(cat_filter) &
        df["Region"].isin(reg_filter) &
        df["Segment"].isin(seg_filter)
    ]

    st.markdown(f"**Showing {len(filtered):,} of {len(df):,} records**")

    display_cols = [
        "Order ID", "Order Date", "Product Name", "Category",
        "Sub-Category", "Region", "Quantity", "Sales",
        "Profit", "Stock Level", "Reorder Point"
    ]
    st.dataframe(filtered[display_cols], use_container_width=True)

    # Download button
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Filtered Data as CSV",
        data=csv,
        file_name="filtered_inventory.csv",
        mime="text/csv"
    )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 6 — AI ASSISTANT  (Groq LLaMA 3)
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🤖 AI Assistant":
    st.title("🤖 AI Inventory Assistant")
    st.markdown("Powered by **Groq API + LLaMA 3**. Ask anything about your inventory!")
    st.markdown("---")

    # ── Example questions ────────────────────────────────────────────────────
    st.markdown("#### 💡 Example Questions You Can Ask")
    col1, col2 = st.columns(2)
    with col1:
        st.info("Which product has the lowest stock right now?")
        st.info("Which category has the highest sales?")
        st.info("Which products need reorder immediately?")
    with col2:
        st.info("What is the total revenue of the store?")
        st.info("Which region performs best in sales?")
        st.info("Summarize the overall inventory status")

    st.markdown("---")

    # ── Initialize chat history in session state ──────────────────────────
    # session_state keeps chat history alive while app is running
    if "chat_history" not in st.session_state:
        st.session_state.chat_history  = []   # stores API message format
    if "display_history" not in st.session_state:
        st.session_state.display_history = [] # stores display format

    # ── Display previous chat messages ───────────────────────────────────
    for msg in st.session_state.display_history:
        if msg["role"] == "user":
            st.chat_message("user").markdown(msg["content"])
        else:
            st.chat_message("assistant").markdown(msg["content"])

    # ── Chat input box ────────────────────────────────────────────────────
    user_input = st.chat_input("Ask about your inventory...")

    if user_input:
        # Show user message immediately
        st.chat_message("user").markdown(user_input)

        # Call Groq API
        with st.spinner("🤖 AI is thinking..."):
            try:
                response = ask_groq(
                    user_question = user_input,
                    df            = df,
                    chat_history  = st.session_state.chat_history
                )

                # Show AI response
                st.chat_message("assistant").markdown(response)

                # Save to history for multi-turn conversation
                st.session_state.chat_history.append(
                    {"role": "user",      "content": user_input}
                )
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": response}
                )
                st.session_state.display_history.append(
                    {"role": "user",      "content": user_input}
                )
                st.session_state.display_history.append(
                    {"role": "assistant", "content": response}
                )

                # Keep only last 10 messages to avoid token overflow
                if len(st.session_state.chat_history) > 20:
                    st.session_state.chat_history  = st.session_state.chat_history[-20:]
                if len(st.session_state.display_history) > 20:
                    st.session_state.display_history = st.session_state.display_history[-20:]

            except ValueError as e:
                st.error(str(e))
                st.markdown("""
                **How to fix:**
                1. Open `.env` file in your project folder
                2. Replace `your_groq_api_key_here` with your actual key
                3. Get free key at → **console.groq.com**
                4. Restart the app
                """)
            except Exception as e:
                st.error(f"Something went wrong: {str(e)}")

    # ── Clear chat button ─────────────────────────────────────────────────
    st.markdown("---")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history    = []
        st.session_state.display_history = []
        st.success("Chat cleared!")
        st.rerun()
