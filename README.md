# 📦 AI Inventory Management System

A modern **AI-powered inventory management dashboard** built with Streamlit, featuring demand forecasting, low-stock alerts, reorder optimization, and an intelligent chatbot assistant powered by Groq's LLaMA 3.1.

**B.Tech 2nd Year Project**

---

## ✨ Features

### 🏠 Dashboard Home
- **Real-time KPI metrics** — Total products, orders, revenue, profit, and stock levels
- **Sales analytics** — Category breakdown and monthly order trends
- **Top products chart** — Visualize best-selling items

### 📈 Demand Forecasting
- **Linear Regression Model** — Predicts product demand for the next 30 days
- **Category-wise predictions** — Select any category and get daily forecast
- **Model performance metrics** — MAE and R² score displayed for transparency
- **Historical demand comparison** — See past data alongside predictions

### 🚨 Low Stock Alerts
- **Rule-based detection** — Automatic flagging when stock ≤ reorder point
- **Critical vs. Low status** — Visual severity indicators
- **Comparative charts** — Stock level vs. reorder point visualization

### 📦 Reorder Suggestions
- **EOQ Formula** — Economic Order Quantity calculation to minimize costs
- **Safety stock calculation** — Accounts for lead time and demand variability
- **Smart reorder points** — Recommends when to reorder each product

### 🗂️ Full Inventory
- **Interactive data table** — Browse complete inventory dataset
- **Multi-filter support** — Filter by category, region, and segment
- **CSV export** — Download filtered data for external analysis

### 🤖 AI Assistant
- **Groq LLaMA 3.1 powered** — Natural language questions about inventory
- **Live context awareness** — Analyzes current inventory state in real-time
- **Multi-turn conversation** — Maintains chat history within session
- **Inventory insights** — Answers questions about low stock, sales trends, reorders, etc.

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Frontend** | Streamlit |
| **AI/ML** | scikit-learn (Linear Regression) |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Plotly |
| **LLM** | Groq API (LLaMA 3.1 8B) |
| **Environment** | Python 3.8+ |

---

## 📋 Requirements

- Python 3.8 or higher
- Groq API key (free tier available at https://console.groq.com)
- All dependencies listed in `requirements.txt`

---

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yuvraj2singh/InventoryManagement.git
cd InventoryManagement
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate      # On macOS/Linux
# or
venv\Scripts\activate          # On Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up API Key
Create a `.env` file in the project root and add your Groq API key:
```env
GROQ_API_KEY=your_actual_groq_api_key_here
```

**How to get a Groq API key:**
1. Visit https://console.groq.com/keys
2. Sign up or log in
3. Create a new API key
4. Copy and paste it into your `.env` file

---

## ▶️ Running the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 📁 Project Structure

```
├── app.py                      # Main Streamlit dashboard
├── model.py                    # ML models & forecasting logic
├── groq_assistant.py           # Groq API integration & AI assistant
├── generate_dataset.py         # Dataset generation script (optional)
├── superstore_inventory.csv    # Inventory dataset
├── requirements.txt            # Python dependencies
├── .env                        # API keys (gitignored, create locally)
└── README.md                   # This file
```

### Key Files

**[app.py](app.py)**
- Streamlit frontend with 6 pages
- Data loading and caching
- Dashboard visualization and interactivity

**[model.py](model.py)**
- `load_data()` — Load and clean CSV data
- `train_demand_model()` — Train Linear Regression on historical demand
- `predict_next_30_days()` — Generate demand forecasts
- `get_low_stock_alerts()` — Identify products below reorder point
- `get_reorder_suggestions()` — Calculate EOQ and reorder points
- `get_summary()` — Compute KPI metrics

**[groq_assistant.py](groq_assistant.py)**
- `get_groq_client()` — Initialize Groq API client
- `build_inventory_context()` — Create context string from current data
- `ask_groq()` — Send questions to LLaMA 3.1 and get answers

---

## 🧠 AI Models Used

### 1. **Linear Regression** (Demand Forecasting)
- **Features:** Month, Day of Week, Discount, Category, Year
- **Target:** Quantity (demand)
- **Training:** 80/20 split (time-ordered, no shuffle)
- **Evaluation:** MAE and R² score

### 2. **Rule-Based AI** (Low Stock Alerts)
- **Rule:** If `Stock Level ≤ Reorder Point` → flag as LOW STOCK
- **Severity:** Critical if stock ≤ 50% of reorder point

### 3. **EOQ Formula** (Reorder Optimization)
```
EOQ = √ ( 2 × Annual Demand × Ordering Cost ) / Holding Cost
ROP = Avg Daily Demand × Lead Time + Safety Stock
```

---


## 📊 Sample Data

The project includes `superstore_inventory.csv` with sample data containing:
- Order information (ID, Date, Ship Date)
- Product details (Name, Category, Sub-Category)
- Sales metrics (Quantity, Sales, Profit, Discount)
- Inventory metrics (Stock Level, Reorder Point, Lead Time Days)
- Geographic data (Region, Segment)

Generate additional test data:
```bash
python generate_dataset.py
```

---

## 🎯 Usage Examples

### Q: "Which product has the lowest stock right now?"
**AI Assistant Answer:** Lists the product with lowest stock and suggests immediate reorder.

### Q: "What's the total revenue by region?"
**AI Assistant Answer:** Provides region-wise revenue breakdown from current data.

### Q: "Which category has the highest sales?"
**AI Assistant Answer:** Identifies top-performing category based on live inventory data.

---

## 🐛 Troubleshooting

### Error: "Groq API key not set"
- Ensure `.env` file exists in the project root
- Verify `GROQ_API_KEY=your_key_here` is in the `.env` file
- Restart the Streamlit app

### Error: "Model decommissioned"
- The app uses `llama-3.1-8b-instant` (current recommended model)
- If you see an old model error, update [groq_assistant.py](groq_assistant.py#L156)

### Slow data loading?
- The app caches data and model (`@st.cache_data` and `@st.cache_resource`)
- First load may take a few seconds; subsequent loads are instant

---

## 📝 License

This project is open-source. Use freely for educational and commercial purposes.

---


## 📞 Support

For questions or issues:
- Check the [Streamlit docs](https://docs.streamlit.io)
- Review [Groq API documentation](https://console.groq.com/docs)
- Open an issue on GitHub

