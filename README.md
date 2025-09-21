***

# Aquaverse

Aquaverse is an **end-to-end ocean data intelligence platform** developed under Smart India Hackathon (SIH). It streamlines the extraction, processing, search, and visualization of oceanographic datasets, enabling interactive research and discovery for scientists, students, and enthusiasts.

***

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Architecture Overview](#architecture-overview)
- [Repository Structure](#repository-structure)
- [Technology Stack](#technology-stack)
- [Setup and Installation](#setup-and-installation)
- [Typical Workflow](#typical-workflow)
- [API Endpoints](#api-endpoints)
- [Visualization and Usage](#visualization-and-usage)
- [License](#license)

***

## Project Overview

Aquaverse consolidates raw scientific float observations (e.g., ARGO NetCDF profiles) into searchable, summarized, and visual formats. It uses an **LLM-powered retrieval-augmented generation (RAG) backend** and a modern web frontend for interactive Q&A and analytics on ocean temperature, salinity, and other marine parameters.

***

## Key Features

- **Automated Data Pipeline:** Converts ARGO NetCDF profiles to Parquet format for fast, scalable processing.
- **Knowledge-Enhanced Search:** ChromaDB-powered semantic search for context-aware Q&A using large language models.
- **RESTful API:** FastAPI backend provides endpoints for authentication, user profiles, and chat-based data exploration.
- **Interactive Visualization:** Jupyter/Plotly-based visuals for scientific insights.
- **Geospatial and Temporal Queries:** Users can query by date, region, depth, temperature, salinity, and more.

***

## Architecture Overview

**Data Flow:**
1. **Data Ingestion:** Raw ARGO datasets are processed and stored in efficient formats (`data/` – `.nc` to `.parquet`).
2. **Database:** Core parameters loaded into a relational (PostgreSQL/SQLAlchemy) DB & ChromaDB for embeddings.
3. **RAG Backend:** FastAPI provides endpoints for queries. Natural language requests are translated into SQL, results interpreted with LLM summarization, and structured responses returned.
4. **Frontend:** Built on React + Vite, with Plotly.js, Recharts, React Leaflet for geospatial/graphical data display.

***

## Repository Structure

```
Aquaverse/
│
├── Data_processing_visuals/   # Jupyter notebooks – e.g., bgc_processing.ipynb for preprocessing/plotting
├── backend/                   # Python FastAPI app, DB access, RAG, API, requirements.txt
│   ├── main.py
│   ├── rag/
│   │   └── dataVisualisation.py
│   └── requirements.txt
├── data/                      # Raw & processed datasets (.nc, .parquet)
├── db/chroma_db/              # Persistent ChromaDB store (vector database)
├── frontend/                  # React + Tailwind + Plotly/Vite client
│   └── package.json
└── .gitignore                 # VCS exclusion rules
```

***

## Technology Stack

**Backend:**
- Python 3.10+, FastAPI, SQLAlchemy, psycopg2, ChromaDB, LangChain, Huggingface transformers, Jupyter, Streamlit, Plotly

**Frontend:**
- React 19, Vite, TailwindCSS, Recharts, Plotly.js, React-Leaflet, Framer Motion

**Data & Search:**
- NetCDF/Parquet processing (xarray, pandas, pyarrow)
- ChromaDB (semantic vector search)
- PostgreSQL (relational storage)

***

## Setup and Installation

### 1. Clone the Repository

```bash
git clone https://github.com/CodePhoenix-org/Aquaverse.git
cd Aquaverse
```

### 2. Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
# Set up .env with DB_URI and optional OpenRouter API key for LLM chat
```

- To run FastAPI (development):
  ```bash
  uvicorn main:app --reload
  ```

### 3. Frontend Setup

```bash
cd ../frontend
npm install
npm run dev  # Serves frontend at localhost:5173 by default
```

### 4. Data Preparation

- Place or link your ARGO `.nc` NetCDF files in `data/`.
- Use the provided Jupyter notebooks (e.g., `Data_processing_visuals/bgc_processing.ipynb`) for pre-processing and visualization.
- Data pipeline scripts convert `.nc` to Parquet; backend scripts populate ChromaDB automatically as needed.

***

## Typical Workflow

1. **Upload or sync new ARGO float files to `/data`.**
2. **Run the conversion notebook or script to produce `/data/processed/argo_profiles.parquet`.**
3. **Start the backend—on first run, ChromaDB is populated for fast context retrieval.**
4. **Use the frontend to search, analyze, and visualize the data:**
   - Ask natural language questions (e.g., "Show temperature near 9°N 68°E for 2025-03-11").
   - Results are interpreted and visualized with Plotly/Recharts.
5. **Advanced:** Use or fork the Jupyter notebooks for deeper stats or custom plots.

***

## API Endpoints (Backend)

- `GET /` – Health check
- `POST /chat` – RAG-powered Q&A; body: `{ "query": "Show salinity in Indian Ocean..." }`
- `GET /profiles`, `POST /auth` – Authentication and user management (extend as needed)

***

## Visualization and Usage

- **Map Visuals:** React-Leaflet shows spatial coverage of float data.
- **Profile Plots:** Query temperature/salinity by depth.
- **Custom Queries:** Filter by region, time, and parameter using interactive UI or API.
- **Summaries:** LLM generates accessible, non-technical data explanations for broader audiences.

***

## License

This repository is for SIH 2025 demonstration and educational purposes. For reproduction and full license terms, see the `LICENSE` file.

***

**Note:** Update `.env` with correct DB connection and API keys. For advanced cloud deployment, configure Docker and cloud storage as required.

***

Feel free to adjust the stack, API, and workflow sections as your implementation evolves!
