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
├── 📂 modules/                # 🛠️ Shared Tools (DO NOT edit without coordination)
│   ├── tool_decision_tree.py  # Generic Algorithm Logic
│   ├── tool_knn.py            # Generic Algorithm Logic
│   ├── tool_kmeans.py         # Generic Algorithm Logic
│   └── tool_apriori.py        # Generic Algorithm Logic
│
├── 📂 domains/                # 🚀 Team Workspaces (Implementation)
│   ├── 📁 healthcare/         # Healthcare Team Workspace
│   ├── 📁 finance/            # Finance Team Workspace
│   ├── 📁 education/          # Education Team Workspace
│   └── 📁 automotive/         # Automotive Team Workspace
│
├── uv.lock                    # 🔒 Critical File (Ensures Version Consistency)
└── pyproject.toml             # Dependencies Definition
````
### 📝 Workflow Rules:
1. **`modules/` folder:** Contains the core "Tools" used by everyone. Editing files here affects the entire team.
2. **`domains/` folder:** Each team has their own folder to clean their specific data and write their implementation scripts.
---
## ⚙️ 2. Setup & Installation Guide

We are using `uv` (a modern replacement for pip) to manage this project.

Why? To guarantee that every team member uses the exact same Python version and library versions (scikit-learn, pandas, etc.). This prevents the "it works on my machine" issue.
### Step 1: Install uv (If you haven't already)
Open your Terminal (PowerShell or CMD) and run the following command:
**For Windows:**
```powershell
powershell -c "irm [https://astral.sh/uv/install.ps1](https://astral.sh/uv/install.ps1) | iex"
```
_(Note: Close and reopen your terminal after installation to apply changes)_

---
### Step 2: Clone & Sync

With a single command, `uv` will download Python, create the Virtual Environment, and install all required libraries.
1. **Clone the Repository:**
```bash
    git clone https://github.com/Khalid7466/Data-Mining-Project.git
    cd Data-Mining-Project-2025
```    
2. **Run the Magic Command (Sync):** ✨
```bash
    uv sync
```

    _This reads `uv.lock` and sets up your environment automatically._
---
## 💻 3. VS Code Configuration
To ensure VS Code detects the installed libraries:
1. Open the project folder in VS Code.
2. Open any Python file (e.g., `main.py`).
3. Look at the bottom right corner (or press `Ctrl+Shift+P` and search for `Python: Select Interpreter`).
4. Select the option marked as **`('.venv': venv)`** or **`Recommended`**.

---
## 📦 4. Library Management (Important)

### How to Run Code?
You can run your scripts in two ways:
1. Using the **Run** button in VS Code (Ensure the correct interpreter is selected).
2. Via the terminal using `uv`:
```bash
    uv run domains/finance/main_finance.py
```
### Adding New Libraries (Admins Only)
If we need to add a new library to the project, **DO NOT** use `pip install`. Instead, follow this process:
```bash
uv add library_name
```

Then, push the updated uv.lock file to GitHub.
Other team members simply need to run uv sync to get the new library.

---

**Good luck, team! 💪🚀**