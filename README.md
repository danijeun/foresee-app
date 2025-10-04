# Foresee App

**Automated ML Analysis & Reporting Platform**

An intelligent application that takes any dataset and automatically generates a comprehensive PDF report with analysis from 5 different machine learning models. Built with a multi-agent architecture where specialized AI agents collaborate to deliver professional, insightful reports.

---

## 🔄 Application Workflow

```mermaid
sequenceDiagram
    participant User
    participant Frontend
    participant Backend
    participant AI_Agent
    participant ML_Engine
    participant PDF_Generator

    Note over User,PDF_Generator: 1. Data Upload & Target Selection
    User->>Frontend: Upload CSV
    Frontend->>Backend: Send CSV file
    Backend->>AI_Agent: Analyze columns & suggest target
    AI_Agent-->>Backend: Suggested target variables
    Backend-->>Frontend: Display target suggestions
    Frontend-->>User: Show suggestions
    User->>Frontend: Select target variable
    Frontend->>Backend: Submit selected target

    Note over User,PDF_Generator: 2. Exploratory Data Analysis (EDA)
    Backend->>ML_Engine: Run EDA
    ML_Engine-->>AI_Agent: EDA results & plots
    AI_Agent-->>Backend: EDA insights

    Note over User,PDF_Generator: 3. Model Training & Evaluation
    Backend->>ML_Engine: Train 5 ML models
    Note over ML_Engine: Training in parallel:<br/>1. Linear Regression<br/>2. Random Forest<br/>3. XGBoost<br/>4. Neural Network<br/>5. Gradient Boosting
    ML_Engine-->>Backend: Model performance metrics

    Note over User,PDF_Generator: 4. Model Explainability
    Backend->>ML_Engine: Run SHAP/LIME explainability
    ML_Engine-->>AI_Agent: Global & local explanations
    AI_Agent-->>Backend: Interpretation insights

    Note over User,PDF_Generator: 5. Report Generation
    Backend->>AI_Agent: Summarize EDA + Model results
    AI_Agent-->>Backend: Natural language report
    Backend->>PDF_Generator: Generate PDF report
    PDF_Generator-->>Backend: Final PDF file

    Note over User,PDF_Generator: 6. View & Download
    Backend-->>Frontend: Show preview & download option
    Frontend-->>User: View / Download Report
```

## 🏗️ System Architecture

```mermaid
graph TB
    subgraph "User Interface"
        User[👤 User]
        UI[🖥️ Frontend<br/>React/Next.js]
    end
    
    subgraph "Backend Services"
        API[⚙️ Backend API<br/>FastAPI]
        Queue[📬 Task Queue<br/>Celery/Redis]
    end
    
    subgraph "Multi-Agent System"
        Orchestrator[🎯 Agent Orchestrator]
        Agent1[🔍 Data Analysis Agent]
        Agent2[📊 EDA Agent]
        Agent3[🤖 ML Training Agent]
        Agent4[💡 Explainability Agent]
        Agent5[📝 Report Synthesis Agent]
        Agent6[📄 PDF Generator Agent]
    end
    
    subgraph "ML & Data Platform"
        Snowflake[❄️ Snowflake<br/>Data Warehouse + ML]
        Storage[💾 Cloud Storage<br/>S3/Azure Blob]
    end
    
    subgraph "AI Services"
        Claude[🧠 Claude AI<br/>via MCP]
    end

    User --> UI
    UI <--> API
    API --> Queue
    Queue --> Orchestrator
    
    Orchestrator --> Agent1
    Agent1 --> Agent2
    Agent2 --> Agent3
    Agent3 --> Agent4
    Agent4 --> Agent5
    Agent5 --> Agent6
    
    Agent1 <--> Snowflake
    Agent2 <--> Snowflake
    Agent3 <--> Snowflake
    Agent4 <--> Snowflake
    
    Agent1 <--> Claude
    Agent2 <--> Claude
    Agent5 <--> Claude
    
    API <--> Storage
    Agent6 --> Storage
    
    Storage --> UI
    
    classDef userStyle fill:#e1bee7,stroke:#8e24aa,stroke-width:2px
    classDef backendStyle fill:#fff9c4,stroke:#f57f17,stroke-width:2px
    classDef agentStyle fill:#b3e5fc,stroke:#0277bd,stroke-width:2px
    classDef dataStyle fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px
    classDef aiStyle fill:#ffccbc,stroke:#d84315,stroke-width:2px
    
    class User,UI userStyle
    class API,Queue,Orchestrator backendStyle
    class Agent1,Agent2,Agent3,Agent4,Agent5,Agent6 agentStyle
    class Snowflake,Storage dataStyle
    class Claude aiStyle
```

## ✨ Key Features

- **🚀 Any Dataset Size**: Handles datasets from small to enterprise-scale
- **🤖 Multi-Agent Intelligence**: Specialized AI agents collaborate for comprehensive analysis
- **📊 5 ML Models**: Automatic training and comparison of 5 different algorithms
- **💡 Explainable AI**: SHAP/LIME explanations for model transparency
- **❄️ Snowflake Integration**: Leverages Snowflake's powerful ML and data processing
- **📄 Professional Reports**: Auto-generated PDF reports with insights and visualizations
- **⚡ Parallel Processing**: Efficient model training with parallel execution
- **🔍 Smart Target Detection**: AI suggests the best target variable for prediction

## 🛠️ Technology Stack

### Frontend
- **React 18** + **Next.js 14**
- **TypeScript**
- **TailwindCSS** + **shadcn/ui**
- **React Query** for state management
- **Chart.js** / **D3.js** for visualizations

### Backend
- **Python 3.11+**
- **FastAPI** for REST API
- **Celery** + **Redis** for task queue
- **PostgreSQL** for metadata storage

### Multi-Agent System
- **LangGraph** / **LangChain** for agent orchestration
- **Claude API** (via MCP) for intelligent agents
- Custom agent protocols for collaboration

### ML & Data
- **Snowflake** (Data Warehouse + ML Engine)
- **Snowpark ML** for model training
- **Scikit-learn**, **XGBoost**, **LightGBM**
- **TensorFlow** / **PyTorch**
- **SHAP** + **LIME** for explainability

### PDF Generation
- **WeasyPrint** or **Playwright**
- **Jinja2** templates
- **Matplotlib** / **Plotly** for charts

## 📋 Project Structure

```
foresee-app/
├── frontend/              # Next.js React application
│   ├── components/        # UI components
│   ├── pages/            # Application pages
│   └── styles/           # CSS and styling
├── backend/              # FastAPI backend
│   ├── api/             # API endpoints
│   ├── agents/          # Multi-agent system
│   ├── models/          # Database models
│   └── services/        # Business logic
├── agents/              # Agent implementations
│   ├── data_agent/      # Data analysis agent
│   ├── eda_agent/       # EDA agent
│   ├── ml_agent/        # ML training agent
│   ├── explain_agent/   # Explainability agent
│   ├── synthesis_agent/ # Report synthesis agent
│   └── pdf_agent/       # PDF generator agent
├── shared/              # Shared utilities
└── docs/               # Documentation
```

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- Node.js 18+
- Docker & Docker Compose
- Snowflake account
- Claude API key

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/foresee-app.git
   cd foresee-app
   ```

2. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your credentials
   ```

3. **Start with Docker Compose**
   ```bash
   docker-compose up -d
   ```

4. **Access the application**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

## 🔐 Configuration

Create a `.env` file with the following variables:

```env
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/foresee

# Redis
REDIS_URL=redis://localhost:6379

# Snowflake
SNOWFLAKE_ACCOUNT=your_account
SNOWFLAKE_USER=your_user
SNOWFLAKE_PASSWORD=your_password
SNOWFLAKE_WAREHOUSE=your_warehouse
SNOWFLAKE_DATABASE=your_database
SNOWFLAKE_SCHEMA=your_schema

# AI Services
ANTHROPIC_API_KEY=your_claude_api_key

# AWS (for S3 storage)
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_S3_BUCKET=your_bucket_name

# Security
JWT_SECRET=your_jwt_secret
```

## 📖 Usage

1. **Upload your dataset** (CSV format)
2. **Review suggested target variables** from the AI agent
3. **Select your target variable**
4. **Wait for analysis** - the system will:
   - Perform exploratory data analysis
   - Train 5 ML models in parallel
   - Generate explainability insights
   - Create a comprehensive report
5. **Download your PDF report** with all findings and recommendations

## 🎯 Supported ML Models

1. **Linear Regression / Logistic Regression**
2. **Random Forest**
3. **XGBoost**
4. **Neural Network (TensorFlow/PyTorch)**
5. **Gradient Boosting (LightGBM)**

## 📊 Report Contents

Each generated report includes:

- **Executive Summary**: Key findings and recommendations
- **Data Overview**: Dataset statistics and characteristics
- **EDA Findings**: Visual analysis and insights
- **Model Performance**: Comparison of all 5 models
- **Best Model Analysis**: Deep dive into top performer
- **Feature Importance**: Key drivers and SHAP values
- **Model Explainability**: Global and local explanations
- **Recommendations**: Actionable insights

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines before submitting PRs.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Snowflake for ML infrastructure
- Anthropic Claude for intelligent agents
- Open-source ML community

---

**Built with ❤️ by the Foresee Team**
