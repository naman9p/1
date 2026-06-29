# AI Candidate Recommendation Engine

A production-ready, scalable AI system for semantically matching job descriptions with candidate profiles using hybrid scoring and ML-based ranking.

## Overview

This system goes beyond simple keyword matching to understand job requirements and candidate profiles semantically. It combines multiple signals including:

- **Semantic Similarity**: Deep understanding using Sentence Transformers
- **Skill Matching**: Fuzzy matching of required and preferred skills
- **Experience Alignment**: Seniority and years of experience matching
- **Behavioral Signals**: Recruiter interaction and platform engagement data
- **Industry Fit**: Domain and industry alignment
- **Bonus Factors**: Leadership, startup experience, open source contributions

The system uses a two-stage ranking approach:
1. **Retrieval**: Fast semantic search to get top 300 candidates
2. **Re-ranking**: Cross-encoder model for precise ranking of top candidates

## Architecture

```
backend/
├── app/
│   ├── api/                    # FastAPI endpoints
│   │   ├── __init__.py
│   │   └── routes.py           # API route handlers
│   ├── core/                   # Core configuration
│   │   ├── __init__.py
│   │   ├── config.py           # Settings management
│   │   └── logging_config.py   # Logging setup
│   ├── schemas/                # Pydantic models
│   │   ├── __init__.py
│   │   ├── api.py              # API response schemas
│   │   ├── candidate.py        # Candidate schemas
│   │   ├── job.py              # Job schemas
│   │   ├── recommendation.py   # Recommendation schemas
│   │   └── scoring.py          # Scoring schemas
│   ├── services/               # Business logic
│   │   ├── __init__.py
│   │   ├── behavioral_parser.py    # Behavioral signal processing
│   │   ├── candidate_processor.py  # Candidate document processing
│   │   ├── embedding_service.py    # Embedding generation
│   │   ├── job_parser.py           # Job description parsing
│   │   ├── ranking_engine.py       # Cross-encoder re-ranking
│   │   ├── recommendation_pipeline.py  # Main pipeline
│   │   ├── scoring_engine.py       # Hybrid scoring
│   │   └── vector_store.py         # ChromaDB integration
│   ├── __init__.py
│   └── main.py                 # FastAPI application
├── data/                       # Data directory
│   ├── candidates.jsonl        # Sample candidate data
│   └── job_description.txt     # Sample job description
├── scripts/                    # CLI tools
│   ├── __init__.py
│   └── cli.py                  # Command-line interface
├── tests/                      # Unit tests
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_behavioral_parser.py
│   ├── test_embedding_service.py
│   ├── test_ranking_engine.py
│   ├── test_recommendation_pipeline.py
│   └── test_scoring_engine.py
├── .env.example                # Environment template
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Installation

### Prerequisites

- Python 3.10+
- 16GB RAM recommended (for model loading)
- CPU-only supported (no GPU required)

### Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd backend
```

2. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Configure environment**:
```bash
cp .env.example .env
# Edit .env with your settings
```

5. **Initialize the system**:
```bash
python -m scripts.cli init
```

## Usage

### Running the API Server

```bash
# Development mode
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

Access the API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Using the CLI

```bash
# Initialize system
python -m scripts.cli init

# Process a job description
python -m scripts.cli process_job data/job_description.txt --job-id job_001

# Index candidates
python -m scripts.cli index_candidates data/candidates.jsonl

# Generate recommendations
python -m scripts.cli recommend job_001 --top-k 100

# Export to CSV
python -m scripts.cli export job_001 --output recommendations.csv

# Check system status
python -m scripts.cli status
```

### API Endpoints

#### Health Check
```bash
curl http://localhost:8000/api/health
```

#### Upload Job Description
```bash
curl -X POST http://localhost:8000/api/upload-job \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "job_001",
    "title": "Senior ML Engineer",
    "description": "We are seeking...",
    "company": "TechCorp"
  }'
```

#### Index Candidates
```bash
curl -X POST http://localhost:8000/api/index-candidates \
  -F "file=@data/candidates.jsonl"
```

#### Generate Recommendations
```bash
curl -X POST http://localhost:8000/api/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "job_001",
    "top_k": 100,
    "include_reasoning": true,
    "use_reranking": true
  }'
```

#### Get Candidate Details
```bash
curl http://localhost:8000/api/candidate/cand_001
```

#### Get Metrics
```bash
curl http://localhost:8000/api/metrics
```

## Configuration

All settings are configurable via environment variables or the `.env` file:

### Scoring Weights

Weights must sum to 1.0:

| Weight | Default | Description |
|--------|---------|-------------|
| `WEIGHT_SEMANTIC` | 0.40 | Semantic similarity |
| `WEIGHT_SKILL` | 0.20 | Skill match |
| `WEIGHT_EXPERIENCE` | 0.10 | Experience match |
| `WEIGHT_BEHAVIOR` | 0.10 | Behavioral signals |
| `WEIGHT_INDUSTRY` | 0.10 | Industry match |
| `WEIGHT_EDUCATION` | 0.05 | Education match |
| `WEIGHT_BONUS` | 0.05 | Bonus features |

### Model Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `EMBEDDING_MODEL` | `BAAI/bge-large-en-v1.5` | Primary embedding model |
| `FALLBACK_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Fallback model |
| `CROSS_ENCODER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Re-ranking model |

### Retrieval Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `RETRIEVAL_TOP_K` | 300 | Candidates to retrieve |
| `RERANK_TOP_K` | 100 | Candidates to re-rank |
| `FINAL_OUTPUT_K` | 100 | Final recommendations |

## Design Decisions

### Why Hybrid Scoring?

Pure semantic search can miss important signals like specific skill requirements or behavioral indicators. Our hybrid approach combines:
- Semantic understanding (40%)
- Explicit skill matching (20%)
- Experience and seniority (10%)
- Behavioral engagement (10%)
- Industry and education fit (15%)
- Bonus factors (5%)

### Why Two-Stage Ranking?

1. **Retrieval**: Fast vector search to narrow down from thousands to hundreds
2. **Re-ranking**: Expensive cross-encoder only on top candidates for precision

This balances speed and accuracy, enabling sub-5-minute processing for thousands of candidates.

### Why ChromaDB?

- Simple, lightweight vector database
- Persistent storage
- No external dependencies
- Easy to deploy and maintain

### Model Selection

- **BAAI/bge-large-en-v1.5**: State-of-the-art embedding model with excellent semantic understanding
- **all-MiniLM-L6-v2**: Fast fallback for resource-constrained environments
- **ms-marco-MiniLM-L-6-v2**: Optimized for re-ranking tasks

## Performance

Expected performance on CPU (16GB RAM):

| Operation | Time |
|-----------|------|
| Model loading | 30-60s |
| Job processing | <1s |
| Candidate indexing (1000) | 60-120s |
| Recommendation generation | 30-60s |
| **Total (5000 candidates)** | **<5 minutes** |

## Testing

Run all tests:
```bash
pytest tests/ -v
```

Run specific test:
```bash
pytest tests/test_scoring_engine.py -v
```

Run with coverage:
```bash
pytest tests/ --cov=app --cov-report=html
```

## Output Format

The system generates a CSV with the following columns:

| Column | Description |
|--------|-------------|
| `candidate_id` | Unique identifier |
| `rank` | Final rank position |
| `score` | Final combined score |
| `reasoning` | Explanation for ranking |
| `name` | Candidate name |
| `headline` | Professional headline |
| `years_experience` | Total years |
| `seniority_level` | Inferred seniority |
| `semantic_score` | Semantic similarity |
| `skill_score` | Skill match |
| `experience_score` | Experience match |
| `behavior_score` | Behavioral signals |
| `hybrid_score` | Weighted hybrid score |
| `rerank_score` | Cross-encoder score |
| `top_skills` | Key matching skills |

## Extensibility

### Adding New Scoring Features

1. Add feature to `ScoringFeatures` schema
2. Implement computation in `ScoringEngine`
3. Add weight to configuration
4. Update hybrid score calculation

### Custom Job Parsing

Extend `JobParserService` to add:
- Custom skill extraction
- Industry-specific parsing
- Company-specific requirements

### Alternative Vector Stores

Implement `VectorStoreService` interface for:
- Pinecone
- Weaviate
- Qdrant
- Milvus

## Future Improvements

1. **Model Fine-tuning**: Fine-tune embedding model on hiring domain
2. **Active Learning**: Incorporate recruiter feedback
3. **A/B Testing**: Test different weight configurations
4. **Real-time Updates**: Stream processing for candidate updates
5. **Explainability**: SHAP values for score decomposition
6. **Multi-language**: Support for non-English job descriptions
7. **Bias Detection**: Monitor and mitigate hiring bias
8. **Integration**: ATS and HR platform integrations

## Troubleshooting

### Model Loading Issues

If models fail to download:
```bash
# Set HF home to writable directory
export HF_HOME=/path/to/cache
# Or use offline mode with pre-downloaded models
```

### Memory Issues

For low-memory environments:
```env
EMBEDDING_MODEL=all-MiniLM-L6-v2
BATCH_SIZE=16
```

### ChromaDB Errors

Reset the vector store:
```bash
python -m scripts.cli reset
```

## License

MIT License - See LICENSE file for details.

## Support

For issues and feature requests, please open a GitHub issue.
