# Foresee App

**Automated Machine Learning Analysis & Reporting Platform**

Foresee is an intelligent web application that automates the entire machine learning workflow: from data upload to model training to generating professional PDF reports. Built with a multi-agent architecture powered by AI, it makes advanced ML analysis accessible to everyone.

---

## 🎯 What Does It Do?

Upload a CSV file → Get professional ML insights in minutes

1. **Upload** your dataset (CSV format)
2. **AI suggests** the best target variables to predict
3. **Select** your prediction target
4. **Automatic training** of 3 ML models (Logistic Regression, Decision Tree, XGBoost)
5. **Download** a comprehensive PDF report with insights and recommendations

---

## ✨ Key Features

### 🤖 **AI-Powered Target Selection**
- Uses Google Gemini to analyze your dataset
- Recommends the top 5 most valuable prediction targets
- Provides importance scores and business rationale
- Distinguishes between target variables and features

### 📊 **Automatic Exploratory Data Analysis (EDA)**
- Comprehensive statistical analysis of all columns
- Detects data types, missing values, duplicates
- Calculates numeric metrics (mean, std, quartiles, skewness, kurtosis)
- Analyzes categorical distributions and cardinality
- Identifies datetime patterns and text characteristics
- Stores all results in Snowflake for querying

### 🚀 **Parallel Model Training**
Trains 3 machine learning models simultaneously:
- **Logistic Regression**: Fast, interpretable baseline
- **Decision Tree**: Captures non-linear patterns
- **XGBoost**: State-of-the-art gradient boosting

Each model provides:
- Performance metrics (accuracy, precision, recall, F1, ROC-AUC)
- Feature importance rankings
- Confusion matrices
- Model-specific insights

### 📄 **Professional PDF Reports**
- AI-generated natural language insights
- Data quality assessment
- Model performance comparisons
- Feature importance analysis
- Visualizations and charts
- Actionable recommendations

### ❄️ **Snowflake Integration**
- Scalable data warehouse for enterprise datasets
- Isolated workflow schemas (one per upload)
- SQL-based data processing
- Secure data storage

### ⚡ **Modern Web Interface**
- React frontend with Tailwind CSS
- Drag-and-drop file upload
- Real-time progress tracking
- Interactive model selection
- Responsive design

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Frontend                            │
│                      (React + Vite)                         │
│                                                              │
│  • File upload interface                                    │
│  • Target variable selection                                │
│  • Progress tracking                                        │
│  • Report viewing/download                                  │
└──────────────────────┬──────────────────────────────────────┘
                       │ REST API
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                    Backend (Flask)                          │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │           Multi-Agent System                        │    │
│  │                                                      │    │
│  │  • EDA Agent          → Analyze data               │    │
│  │  • Target Agent       → Suggest targets (Gemini)   │    │
│  │  • ML Agents (3x)     → Train models               │    │
│  │  • NL Insights Agent  → Generate report (Gemini)   │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                    Snowflake                                │
│                                                              │
│  • WORKFLOW_<UUID> schemas (isolated per upload)           │
│  • Raw data tables                                          │
│  • EDA results (workflow_eda_summary, column_stats)        │
│  • ML results (logistic/tree/xgboost_summary)              │
│  • Metadata storage                                         │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔄 Application Workflow

### 1. Data Upload & EDA (Parallel)
```
User uploads CSV
    │
    ├─→ Store in Snowflake → EDA Agent analyzes data
    │
    └─→ Target Variable Agent suggests targets (Gemini AI)
```

### 2. Target Selection
```
User sees AI recommendations (ranked 1-5)
    │
    └─→ Selects target variable → Saves to workflow metadata
```

### 3. Model Training (Sequential)
```
Train Logistic Regression
    │
    ├─→ Feature engineering
    ├─→ Model training
    ├─→ Performance evaluation
    └─→ Save results to Snowflake
    
Train Decision Tree
    │
    └─→ [same steps]
    
Train XGBoost
    │
    └─→ [same steps]
```

### 4. Report Generation
```
Natural Language Agent (Gemini)
    │
    ├─→ Collect EDA insights
    ├─→ Collect ML results
    ├─→ Generate narrative insights
    ├─→ Create visualizations
    └─→ Generate PDF report
```

---

## 🛠️ Technology Stack

### Frontend
- **React 19** with **Vite**
- **Tailwind CSS** for styling
- **React Router** for navigation
- **AOS** for animations

### Backend
- **Python 3.11+**
- **Flask 3.0** for REST API
- **Flask-CORS** for cross-origin requests

### AI & ML
- **Google Gemini 2.0/2.5** (via `google-generativeai`)
  - Target variable recommendations
  - Natural language insights generation
- **scikit-learn 1.5.0** (Logistic Regression, Decision Tree)
- **XGBoost 2.1.0** (Gradient Boosting)
- **SHAP 0.44.0** (Model explainability)
- **pandas 2.2.0** & **NumPy 1.26.0**

### Data Platform
- **Snowflake** (Data Warehouse)
  - `snowflake-connector-python 3.12.0`
  - `snowflake-snowpark-python 1.39.1`

### Report Generation
- **ReportLab 4.0.7** (PDF generation)
- **Matplotlib 3.8.0** (Visualizations)

---

## 📋 Project Structure

```
foresee-app/
├── frontend/                    # React application
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Home.jsx        # Landing page
│   │   │   ├── Foresee.jsx     # Main app interface
│   │   │   ├── AboutUs.jsx     # About page
│   │   │   └── Help.jsx        # Help page
│   │   ├── components/
│   │   │   ├── TopBanner.jsx   # Navigation header
│   │   │   └── Footer.jsx      # Footer
│   │   ├── App.jsx             # Main app component
│   │   └── main.jsx            # Entry point
│   ├── package.json
│   └── vite.config.js
│
├── backend/                     # Flask API + Agents
│   ├── app.py                  # Main Flask application
│   │
│   ├── agents/                 # ML & AI Agents
│   │   ├── eda_agent/          # EDA Agent (Snowflake-based)
│   │   │   ├── agent.py        # Main EDA logic
│   │   │   ├── database/       # Snowflake connection & storage
│   │   │   ├── metrics/        # Metric calculators (numeric, categorical, etc.)
│   │   │   └── utils/          # Helpers & validators
│   │   │
│   │   ├── target_variable_agent.py    # AI target recommendations (Gemini)
│   │   ├── logistic_regression_agent.py # Logistic Regression trainer
│   │   ├── decision_tree_agent.py       # Decision Tree trainer
│   │   ├── xgboost_agent.py             # XGBoost trainer
│   │   └── natural_language_agent.py    # PDF report generator (Gemini)
│   │
│   ├── services/               # Business logic services
│   │   ├── workflow_manager.py # Workflow & schema management
│   │   ├── snowflake_ingestion.py # CSV upload to Snowflake
│   │   ├── eda_service.py      # EDA orchestration
│   │   └── config.py           # Configuration
│   │
│   ├── insights/               # Generated JSON insights
│   └── pdf/                    # Generated PDF reports
│
├── requirements.txt            # Python dependencies
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.11+**
- **Node.js 18+**
- **Snowflake account** (with credentials)
- **Google Gemini API key** ([Get one here](https://aistudio.google.com/app/apikey))

### Installation

#### 1. Clone the repository
```bash
git clone https://github.com/yourusername/foresee-app.git
cd foresee-app
```

#### 2. Set up backend

```bash
# Create virtual environment
python -m venv myenv

# Activate virtual environment
# Windows:
myenv\Scripts\activate
# macOS/Linux:
source myenv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### 3. Configure environment variables

Create a `.env` file in the project root:

```env
# Snowflake Configuration
SNOWFLAKE_ACCOUNT=your_account_identifier
SNOWFLAKE_USER=your_username
SNOWFLAKE_PASSWORD=your_password
SNOWFLAKE_WAREHOUSE=your_warehouse
SNOWFLAKE_DATABASE=your_database
SNOWFLAKE_SCHEMA=PUBLIC
SNOWFLAKE_ROLE=your_role

# Google Gemini API
GEMINI_API_KEY=your_gemini_api_key
```

#### 4. Set up frontend

```bash
cd frontend
npm install
```

---

## 🎬 Running the Application

### Option 1: Quick Start (One Terminal) ⭐

**Windows:**
```bash
start.bat
```

**macOS/Linux:**
```bash
chmod +x start.sh  # First time only
./start.sh
```

This will automatically:
1. Activate the Python virtual environment
2. Start the backend server
3. Start the frontend dev server

### Option 2: Using npm (One Terminal)

First, install `concurrently`:
```bash
cd frontend
npm install
```

Then run both servers:
```bash
npm run dev:all
```

### Option 3: Manual (Two Terminals)

**Terminal 1 - Backend:**
```bash
# Activate virtual environment
myenv\Scripts\activate  # Windows
# or
source myenv/bin/activate  # macOS/Linux

# Start Flask server
cd backend
python app.py
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

### Access the Application

Open your browser and navigate to:
- **Frontend:** `http://localhost:5173`
- **Backend API:** `http://localhost:5000`
- **API Docs:** `http://localhost:5000/api/health` (health check)

---

## 📖 Usage Guide

### 1. Upload Your Dataset

- Click **"Foresee"** in the navigation menu
- Drag & drop your CSV file or click **"Choose File"**
- Click **"Upload & Analyze"**

The system will:
- Upload your data to Snowflake
- Run automatic EDA analysis
- Generate target variable recommendations (using AI)

### 2. Select Target Variable

After upload, you'll see 3 recommended targets in a podium layout:
- 🥇 **Gold** (Most important)
- 🥈 **Silver** (Second best)
- 🥉 **Bronze** (Third option)

Click **"Other Options"** to see all 5 recommendations with detailed explanations.

Each recommendation includes:
- **Importance Score** (1-100)
- **Problem Type** (regression/classification)
- **Why Important** (business value)
- **Predictability** (HIGH/MEDIUM/LOW)
- **Suggested Features** (best predictors)

### 3. Model Training

After selecting a target, the system automatically:
1. Trains 3 ML models (10-15 seconds)
2. Evaluates performance metrics
3. Generates natural language insights
4. Creates a PDF report

### 4. View/Download Report

When complete:
- Click **"View Report"** to see it in browser
- Click **"Download Report"** to save PDF

---

## 📊 What's in the Report?

### Executive Summary
- Dataset overview
- Target variable selected
- Best performing model

### Data Analysis
- Data quality metrics
- Missing value analysis
- Duplicate detection
- Column type breakdown

### Model Performance
- Comparison table (all 3 models)
- Performance metrics:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - ROC-AUC
- Confusion matrices

### Feature Importance
- Top 10 most important features
- SHAP value analysis (planned)
- Feature distributions

### Recommendations
- Model selection advice
- Data quality improvements
- Next steps for deployment

---

## 🎯 API Endpoints

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check |
| `POST` | `/api/upload` | Upload CSV & run EDA + Target Agent |
| `GET` | `/api/workflows` | List all workflows |
| `DELETE` | `/api/workflow/<id>` | Delete a workflow |
| `POST` | `/api/query` | Execute SQL query |

### Target Variable Selection

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/target-suggestions/<workflow_id>/<table_name>` | Get AI recommendations |
| `POST` | `/api/workflow/<id>/select-target` | Save target & auto-train models |

### ML Model Training (Manual)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/workflow/<id>/train-logistic-regression` | Train logistic regression |
| `POST` | `/api/workflow/<id>/train-decision-tree` | Train decision tree |
| `POST` | `/api/workflow/<id>/train-xgboost` | Train XGBoost |

### Model Results

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/workflow/<id>/logistic-regression-results` | Get LR results |
| `GET` | `/api/workflow/<id>/decision-tree-results` | Get DT results |
| `GET` | `/api/workflow/<id>/xgboost-results` | Get XGB results |

### Report Generation

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/workflow/<id>/generate-insights` | Generate insights & PDF |
| `GET` | `/api/workflow/<id>/report/view` | View PDF in browser |
| `GET` | `/api/workflow/<id>/report/download` | Download PDF |

---

## 🗄️ Database Schema (Snowflake)

Each workflow creates an isolated schema: `WORKFLOW_<UUID>`

### Tables Created Per Workflow

#### `WORKFLOW_METADATA`
Stores workflow-level metadata including selected target variable.

#### `WORKFLOW_EDA_SUMMARY`
```sql
- analysis_id (UUID)
- table_name (VARCHAR)
- total_rows (INT)
- total_columns (INT)
- duplicate_rows (INT)
- target_column (VARCHAR)
- analysis_type (VARCHAR)
- created_at (TIMESTAMP)
```

#### `COLUMN_STATS`
Detailed statistics for each column:
- Basic metrics (null_count, unique_count, completeness)
- Numeric metrics (mean, std, min, max, quartiles, skewness, kurtosis)
- Categorical metrics (mode, top values, cardinality)
- Text metrics (avg_length, max_length)
- Datetime metrics (date_range, frequency)

#### `LOGISTIC_REGRESSION_SUMMARY`
```sql
- analysis_id (UUID)
- target_variable (VARCHAR)
- test_accuracy (FLOAT)
- test_precision (FLOAT)
- test_recall (FLOAT)
- test_f1_score (FLOAT)
- test_roc_auc (FLOAT)
- confusion_matrix (ARRAY)
- top_features (ARRAY)
- recommendations (VARCHAR)
```

#### `DECISION_TREE_SUMMARY`
Same structure as Logistic Regression + tree-specific metrics:
- tree_depth (INT)
- n_leaves (INT)
- max_depth (INT)

#### `XGBOOST_SUMMARY`
Same structure + XGBoost-specific metrics:
- n_estimators (INT)
- max_depth (INT)
- learning_rate (FLOAT)

---

## ⚙️ Configuration

### Backend Configuration

Located in `backend/services/config.py`:

```python
# Snowflake connection loaded from .env
SNOWFLAKE_ACCOUNT
SNOWFLAKE_USER
SNOWFLAKE_PASSWORD
SNOWFLAKE_WAREHOUSE
SNOWFLAKE_DATABASE
SNOWFLAKE_SCHEMA

# AI Configuration
GEMINI_API_KEY

# Flask Settings
MAX_FILE_SIZE = 500 MB
ALLOWED_EXTENSIONS = ['csv']
```

### Frontend Configuration

Located in `frontend/src/pages/Foresee.jsx`:

```javascript
const API_BASE_URL = "http://localhost:5000/api";
```

Change this to your backend URL in production.

---

## 🔒 Security & Privacy

### Data Isolation
- Each upload creates an isolated Snowflake schema
- No data mixing between workflows
- Automatic cleanup on workflow deletion

### API Security
- CORS enabled for frontend-backend communication
- File size limits (500MB max)
- File type validation (CSV only)

### Data Privacy
- Data stored in your Snowflake account
- AI models don't retain your data
- Gemini API calls are stateless

---

## 🐛 Troubleshooting

### Backend won't start
```bash
# Check Python version
python --version  # Should be 3.11+

# Verify .env file exists with all required variables
cat .env

# Check Snowflake connection
# Try connecting manually with snowsql or Snowflake web UI
```

### Frontend can't connect to backend
```bash
# Verify backend is running on port 5000
curl http://localhost:5000/api/health

# Check CORS is enabled in backend/app.py
# Verify API_BASE_URL in frontend matches backend
```

### Upload fails
- Check file is valid CSV
- Verify Snowflake credentials
- Check Snowflake warehouse is running
- Ensure sufficient Snowflake credits

### Gemini API errors
- Verify `GEMINI_API_KEY` is set correctly
- Check API quota limits
- Test API key at: https://aistudio.google.com/

### Model training fails
- Check target variable has valid data
- Ensure target has sufficient unique values
- Verify no excessive missing values

---

## 📈 Roadmap

### Planned Features
- [ ] Support for more ML models (Random Forest, Neural Networks)
- [ ] Advanced hyperparameter tuning
- [ ] Time series forecasting support
- [ ] Interactive charts in reports
- [ ] Model deployment API
- [ ] Scheduled re-training
- [ ] User authentication
- [ ] Multi-user workspaces
- [ ] Excel file support
- [ ] Real-time model monitoring

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 for Python code
- Use ESLint rules for JavaScript/React
- Write docstrings for all functions
- Add tests for new features
- Update documentation

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

### Technologies
- **Snowflake** - Enterprise data platform
- **Google Gemini** - AI-powered insights
- **React** - Modern web framework
- **Flask** - Lightweight Python API
- **scikit-learn** & **XGBoost** - ML frameworks
- **ReportLab** - PDF generation

### Team
Built with ❤️ by the Foresee Team

---

## 📞 Support

For issues, questions, or suggestions:
- 📧 Email: support@foresee-app.com
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/foresee-app/issues)
- 📚 Documentation: [Wiki](https://github.com/yourusername/foresee-app/wiki)

---

**Happy Analyzing! 🚀**