Here is the updated **`README.md`** with the dataset links included directly in the main file.

-----

````markdown
# 📊 Data Mining Project (2025) - Modular Approach

Welcome team! 👋
This is the official repository for our Data Mining Project.
We are adopting a **Modular Architecture** to ensure our code is organized, reusable, and professional.

---

## 📂 1. Project Structure
To avoid conflicts and ensure smooth collaboration, the project is divided into two main sections:

```text
Data-Mining-Project-2025/
│
├── 📂 data/                   # 💾 Place your downloaded .csv files here
│
├── 📂 modules/                # 🛠️ Shared Tools (Generic Algorithms)
│   ├── tool_decision_tree.py
│   ├── tool_knn.py
│   ├── tool_kmeans.py
│   └── tool_apriori.py
│
├── 📂 domains/                # 🚀 Team Workspaces (Implementation)
│   ├── 📁 healthcare/         # Healthcare Team Workspace
│   ├── 📁 finance/            # Finance Team Workspace
│   ├── 📁 education/          # Education Team Workspace
│   └── 📁 automotive/         # Automotive & E-Commerce Workspace
│
├── uv.lock                    # 🔒 Critical File (Ensures Version Consistency)
└── pyproject.toml             # Dependencies Definition
````

### 📝 Workflow Rules:

1.  **`modules/` folder:** Contains the core "Tools" (Algorithms). **DO NOT** edit these files unless you are the tool maintainer.
2.  **`domains/` folder:** Each team has their own folder to clean their specific data and write their implementation scripts.

-----

## ⚙️ 2. Setup & Installation Guide

We are using **`uv`** (a modern replacement for pip) to manage this project.
**Why?** To guarantee that every team member uses the **exact same Python version** and **library versions**.

### Step 1: Install uv (If you haven't already)

Open your Terminal (PowerShell or CMD) and run:

**For Windows:**

```powershell
powershell -c "irm [https://astral.sh/uv/install.ps1](https://astral.sh/uv/install.ps1) | iex"
```

**For Mac/Linux:**

```bash
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh
```

*(Note: Close and reopen your terminal after installation to apply changes)*

### Step 2: Clone & Sync

With a single command, `uv` will download Python, create the Virtual Environment, and install all libraries.

1.  **Clone the Repository:**

    ```bash
    git clone <PASTE_REPO_URL_HERE>
    cd Data-Mining-Project-2025
    ```

2.  **Run the Magic Command (Sync):** ✨

    ```bash
    uv sync
    ```

    *This reads `uv.lock` and sets up your environment automatically.*

-----

## 💾 3. Data Download Links

Since datasets are not included in the repository, you must download them manually and place them in the `data/` folder.

**⚠️ IMPORTANT:** You must rename the downloaded files **exactly** as shown in the table below, or the code will not find them.

| Team / Domain | Algorithm | Dataset Link | **Required Filename** |
| :--- | :--- | :--- | :--- |
| **Healthcare** | KNN / Tree | [Click to Download (Kaggle)](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) | `healthcare-dataset-stroke-data.csv` |
| **Finance** | Decision Tree | [Click to Download (Kaggle)](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling) | `Churn_Modelling.csv` |
| **Education** | K-Means | [Click to Download (Kaggle)](https://www.kaggle.com/datasets/hellbuoy/car-price-prediction) | `CarPrice_Assignment.csv` |
| **E-Commerce** | Apriori | [Click to Download (UCI)](https://archive.ics.uci.edu/ml/datasets/online+retail) | `Online_Retail.csv` |

**Final Folder Check:**
Ensure your `data/` folder looks like this:

```text
data/
├── healthcare-dataset-stroke-data.csv
├── Churn_Modelling.csv
├── CarPrice_Assignment.csv
└── Online_Retail.csv
```

-----

## 💻 4. VS Code Configuration

To ensure VS Code detects the installed libraries:

1.  Open the project folder in VS Code.
2.  Open any Python file (e.g., `main.py`).
3.  Look at the bottom right corner (or press `Ctrl+Shift+P` and search for `Python: Select Interpreter`).
4.  Select the option marked as **`('.venv': venv)`** or **`Recommended`**.

-----

## 📦 5. How to Run Code?

You can run your scripts in two ways:

1.  **Via VS Code:** Open the file and click the **Run** button (ensure the correct interpreter is selected).
2.  **Via Terminal (using uv):**
    ```bash
    # Example for Healthcare team
    uv run domains/healthcare/main_health.py

    # Example for E-Commerce team
    uv run domains/automotive/main_ecommerce.py
    ```

-----

**Good luck, team\! 💪🚀**

```