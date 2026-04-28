# =============================================================================
# groq_assistant.py  –  Groq LLaMA 3 AI Assistant for Inventory Management
# This file handles all Groq API communication
# =============================================================================

import os
from groq import Groq
from dotenv import load_dotenv
import pandas as pd

# Load API key from .env file
load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# INITIALIZE GROQ CLIENT
# ─────────────────────────────────────────────────────────────────────────────

def get_groq_client():
    """Initialize and return Groq client using API key from .env"""
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key or api_key == "your_groq_api_key_here":
        raise ValueError("❌ Groq API key not set. Please add it to your .env file.")

    return Groq(api_key=api_key)


# ─────────────────────────────────────────────────────────────────────────────
# BUILD INVENTORY CONTEXT FROM DATASET
# (This is sent to Groq so it knows about your data)
# ─────────────────────────────────────────────────────────────────────────────

def build_inventory_context(df):
    """
    Creates a text summary of the current inventory state.
    This is passed to Groq API as context so it can answer
    questions about YOUR specific data.
    """

    # ── Basic stats ──────────────────────────────────────────────────────────
    total_products  = df["Product Name"].nunique()
    total_orders    = df["Order ID"].nunique()
    total_revenue   = round(df["Sales"].sum(), 2)
    total_profit    = round(df["Profit"].sum(), 2)
    avg_stock       = round(df["Stock Level"].mean(), 1)

    # ── Low stock items ──────────────────────────────────────────────────────
    latest = (
        df.sort_values("Order Date")
          .groupby("Product Name")
          .last()
          .reset_index()
    )
    low_stock = latest[latest["Stock Level"] <= latest["Reorder Point"]]
    low_stock_list = low_stock[["Product Name", "Stock Level", "Reorder Point"]]\
                        .head(10).to_string(index=False)

    # ── Top 5 selling products ───────────────────────────────────────────────
    top5 = (df.groupby("Product Name")["Quantity"]
              .sum()
              .sort_values(ascending=False)
              .head(5)
              .reset_index()
              .to_string(index=False))

    # ── Category wise sales ──────────────────────────────────────────────────
    cat_sales = (df.groupby("Category")["Sales"]
                   .sum()
                   .round(2)
                   .reset_index()
                   .to_string(index=False))

    # ── Region wise sales ────────────────────────────────────────────────────
    reg_sales = (df.groupby("Region")["Sales"]
                   .sum()
                   .round(2)
                   .reset_index()
                   .to_string(index=False))

    # ── Reorder needed items ─────────────────────────────────────────────────
    reorder_needed = latest[latest["Stock Level"] <= latest["Reorder Point"]]
    reorder_list   = reorder_needed["Product Name"].tolist()

    # ── Build final context string ───────────────────────────────────────────
    context = f"""
You are an AI Inventory Management Assistant for a Superstore.
You have access to the following LIVE inventory data. Use it to answer questions accurately.

=== INVENTORY SUMMARY ===
- Total Unique Products : {total_products}
- Total Orders          : {total_orders}
- Total Revenue         : ₹{total_revenue:,}
- Total Profit          : ₹{total_profit:,}
- Average Stock Level   : {avg_stock} units

=== LOW STOCK ALERTS (Stock <= Reorder Point) ===
{low_stock_list}

=== TOP 5 PRODUCTS BY QUANTITY SOLD ===
{top5}

=== SALES BY CATEGORY ===
{cat_sales}

=== SALES BY REGION ===
{reg_sales}

=== PRODUCTS THAT NEED REORDER NOW ===
{reorder_list}

Instructions:
- Answer only inventory-related questions
- Use the data above to give specific, accurate answers
- Keep answers clear and concise
- If asked about a product not in the data, say it's not found
- Always suggest action when stock is low
"""
    return context


# ─────────────────────────────────────────────────────────────────────────────
# ASK GROQ  –  Main function called from app.py
# ─────────────────────────────────────────────────────────────────────────────

def ask_groq(user_question, df, chat_history=[]):
    """
    Sends user question + inventory context to Groq LLaMA 3.
    Returns AI response as string.

    Parameters:
        user_question  : string  → what the user typed
        df             : DataFrame → the inventory dataset
        chat_history   : list → previous messages for multi-turn chat

    Returns:
        response_text  : string → Groq's answer
    """

    client  = get_groq_client()
    context = build_inventory_context(df)

    # System message — tells Groq who it is
    system_message = {
        "role": "system",
        "content": context
    }

    # Build message list — system + history + new question
    messages = [system_message] + chat_history + [
        {"role": "user", "content": user_question}
    ]

    # Call Groq API
    response = client.chat.completions.create(
        model    = "llama-3.1-8b-instant",   # Replacement for deprecated llama3-8b-8192
        messages = messages,
        temperature = 0.7,             # 0 = factual, 1 = creative
        max_tokens  = 1024
    )

    return response.choices[0].message.content


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST  (run: python groq_assistant.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from model import load_data

    print("Loading data...")
    df = load_data()

    print("Asking Groq...\n")
    answer = ask_groq("Which product has the lowest stock right now?", df)
    print("🤖 Groq says:")
    print(answer)
