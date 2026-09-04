import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


class Settings(BaseModel):
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    max_iterations_default: int = int(os.getenv("MAX_ITERATIONS_DEFAULT", "5"))
    output_dir: str = "outputs"
    upload_dir: str = "data/uploads"

    # Default quality thresholds — overridable per run via API/UI
    default_thresholds: dict = Field(
        default_factory=lambda: {
            "ks_stat_max": 0.15,          # lower is better; per-column, take mean
            "corr_diff_max": 0.20,        # Frobenius-norm difference, normalized
            "js_divergence_max": 0.10,    # class balance divergence
            "dcr_min_percentile": 5,      # privacy: 5th percentile of nearest-neighbor distance must exceed a safe floor
            "utility_auc_drop_max": 0.10  # max allowed drop of TSTR AUC vs TRTR AUC
        }
    )


settings = Settings()
