
# 📦 Multi-Agent Automated Ingredient Tracking with LangChain and n8n

## 🧠 Project Summary

**SmartReconcile** is a modular, AI-powered data reconciliation system designed to automate the process of tracking ingredients across purchase, production, and sales pipelines. It leverages multi-agent orchestration (LangChain, AutoGen, AutoGPT) and automation platforms (n8n, Airflow) to generate clean, consolidated reports with natural language summaries.

With advanced AI logic, customizable tooling, and seamless integration into low-code platforms, SmartReconcile empowers businesses to minimize manual reconciliation, detect anomalies in real-time, and improve operational visibility with PDF outputs.

---

## 📊 Dataset Overview

| Dataset               | Description                                                    | Key Fields                                               |
|-----------------------|----------------------------------------------------------------|----------------------------------------------------------|
| Raw Material Purchase | Ingredients bought from vendors                               | Ingredient ID, Purchase Date, Quantity (Kg)              |
| Raw-to-Prep Recipe    | Maps raw ingredients to prep recipes                          | Ingredient ID → Prep Item ID, Yield %, Quantity Used     |
| Production Data       | Prep item manufacturing logs                                  | Prep ID, Production Date, Qty Produced, Spoilage         |
| Prep-to-Dish Recipe   | Maps prep items to dish recipes                               | Prep ID → Dish ID, Quantity per Dish                     |
| Sales Data            | Final dish transactions                                       | Dish ID, Sale Date, Dishes Sold                          |

---

## 🧠 Multi-Agent Intelligence: LangChain + AutoGen + AutoGPT

### 🔗 LangChain (CSV & Pandas Agents)

LangChain agents use tools like Python, Pandas, and CSV loading to analyze tabular data with natural language prompts. They allow statements like:

> “Which ingredient's production exceeded purchases?”

The `create_pandas_dataframe_agent()` from `langchain_experimental` enables querying merged DataFrames directly using LLMs.

[Reference](https://medium.com/@thallyscostalat/talk-to-your-data-using-langchain-csv-agents-and-amazon-bedrock-07ee3d35e9f7)

### 🤖 AutoGen

AutoGen enables multi-agent workflows like:
- **Data Agent**: CSV ingestion
- **Analysis Agent**: Find anomalies
- **Report Agent**: Write insights

Its **Studio** GUI supports visual workflow design—making it a powerful no-code addition.

[GitHub - microsoft/autogen](https://github.com/microsoft/autogen)

### 🔁 AutoGPT-style Agents

Designed to autonomously plan and complete tasks:
- Load → Analyze → Report
- Uses goal-directed loops with dynamic function calling

[GitHub - Significant Gravitas AutoGPT](https://github.com/Significant-Gravitas/AutoGPT)

---

## ⚙️ Tooling Integration

| Tool     | Role                                                                 |
|----------|----------------------------------------------------------------------|
| n8n      | Automates scheduling, data loading, PDF dispatch                     |
| LangChain| Python-based CSV/Pandas agents for AI reconciliation                 |
| Airflow  | DAG-based orchestration for scale                                    |
| GPT-4    | LLM for summarization and narrative generation                       |
| Google Docs Template | PDF reporting structure with auto-filled summaries      |

---

## 📂 Key Files

- `reconcile.py`: Core reconciliation script
- `requirements.txt`: Project dependencies
- `Dummy Report 1.pdf` and `Dummy Report 2.pdf`: Sample outputs
- `Terttulia Purchase - Production - Sales Report.pdf`: Annotated final report

---

## 🧾 Installation

```bash
git clone <this-repo>
cd smart-reconcile
pip install -r requirements.txt
```

`.env` file (optional):

```dotenv
OPENAI_API_KEY=your-key
USE_LLM_AGENT=true
PURCHASE_FILE=...
PRODUCTION_FILE=...
SALES_FILE=...
```

---

## 🔍 reconcile.py – Code Flow

1. **File Cache Check**
2. **CSV Cleaning & Normalization**
   - Converts to kilograms
   - Handles missing or malformed units
3. **Summary Table Construction**
   - Merge purchase, production, sales
   - Fills missing SKUs with 0
4. **Reconciliation Logic**
   - Identifies mismatches
   - Groups by ingredient
5. **PDF Report Generation**
   - Multi-page summary
   - Overview + detailed per-SKU tables

---

## 📄 Sample Output

### ✅ Executive Summary (from Dummy Report 2)

- Total Purchased: 728.35 Kg  
- Total Produced: 739.20 Kg  
- Total Sold: 717.51 Kg  
- Highest Purchased: **Chicken Breast Boneless (168.3 Kg)**  
- Highest Sold: **Chicken Breast Boneless (166.1 Kg)**

### ✅ Reconciliation Table (Dummy Report 1)

| Ingredient SKU         | Purchased | Produced | Sold   |
|------------------------|-----------|----------|--------|
| Chicken Breast Boneless| 168.3     | 170.5    | 166.1  |
| Paneer                 | 52.78     | 54.2     | 51.36  |
| Rajma                  | 45.12     | 44.0     | 46.24  |

_(Full tables in the PDFs)_

---

## 📦 Real Report Sample

Check: [`Terttulia Purchase - Production - Sales Report.pdf`](sandbox:/mnt/data/Terttulia%20Purchase%20-%20%20Production%20-%20Sales%20Report.pdf)

Covers:
- Yield calculations per dish
- Spoilage tracking
- Dish-wise breakdown (e.g. Butter Chicken Meal Box: 4 sold, 200g used each)

---

## 🛠 Requirements

```text
pandas
python-dotenv
langchain_experimental
langchain_openai
openai
matplotlib
fpdf
```

---

## 🔧 How to Run

```bash
python reconcile.py
```

Generates:
- Reconciliation logic in pandas or via LangChain Agent
- Timestamped PDF in local folder

---

## 📈 Extend This

- Add cost columns to reconcile finance vs inventory
- Track yield % vs spoilage dynamically
- Integrate with BigQuery or Firebase for live data pulls

---

## 📘 Conclusion

SmartReconcile is a powerful starting point for supply chain auditing, inventory management, or automated restaurant reporting. Combining Python with agentic AI provides:
- Automation of tedious joins
- Reliable, readable reporting
- Natural language narratives at scale

---

**Happy Reconciling! 🚀**
