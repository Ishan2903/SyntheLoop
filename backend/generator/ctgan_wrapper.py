"""CTGAN Generator Wrapper for SyntheLoop.

Wraps ctgan.CTGAN conforming to Data Contract 4.2 and Section 5.2 of the
SyntheLoop Implementation Methodology.
"""

from typing import Any
import pandas as pd
import ctgan


class SyntheticGenerator:
    """Synthetic tabular data generator wrapping CTGAN.

    Conforms to Data Contract 4.2:
    - categorical_columns: list[str]
    - epochs: int (50-500)
    - batch_size: int (50-1000, multiple of 10)
    - generator_dim: list[int] / tuple[int]
    - discriminator_dim: list[int] / tuple[int]
    - pac: int (1-20)
    - reasoning: str (optional audit-trail justification)
    """

    def __init__(self, config: dict[str, Any]):
        """Initializes generator with configuration matching Data Contract 4.2.

        Args:
            config: Dictionary containing CTGAN hyperparameters and settings.
        """
        if not isinstance(config, dict):
            raise TypeError(f"config must be a dict, got {type(config).__name__}")

        self.config = config
        self.model: ctgan.CTGAN | None = None
        self._fitted: bool = False
        self._columns: list[str] | None = None
        self._dtypes: dict[str, Any] | None = None

    def fit(self, real_df: pd.DataFrame) -> None:
        """Trains CTGAN on the provided real DataFrame.

        Validates that categorical_columns is a subset of real_df.columns
        (primary defense against LLM hallucination per NFR-7).

        GPU is used automatically if available via ctgan's internal device
        detection; otherwise falls back cleanly to CPU.

        Args:
            real_df: Pandas DataFrame containing real training data.

        Raises:
            ValueError: If real_df is empty, invalid, or contains hallucinated column names.
            RuntimeError: If model training fails.
        """
        if not isinstance(real_df, pd.DataFrame):
            raise ValueError(f"real_df must be a pandas DataFrame, got {type(real_df).__name__}")

        if real_df.empty or len(real_df) < 1:
            raise ValueError("real_df cannot be empty. Must have at least 1 row.")

        if len(real_df.columns) == 0:
            raise ValueError("real_df has zero columns.")

        # Validate categorical columns against actual DataFrame columns (NFR-7 defense)
        configured_cat_cols = self.config.get("categorical_columns", [])
        if not isinstance(configured_cat_cols, list):
            raise ValueError(f"categorical_columns must be a list, got {type(configured_cat_cols).__name__}")

        missing_cols = [c for c in configured_cat_cols if c not in real_df.columns]
        if missing_cols:
            raise ValueError(
                f"Configured categorical_columns contains columns not present in dataset: {missing_cols}"
            )

        # Store schema metadata for post-sampling fidelity
        self._columns = list(real_df.columns)
        self._dtypes = {col: real_df[col].dtype for col in real_df.columns}

        # Extract and sanitize hyperparameters with safe defaults
        epochs = int(self.config.get("epochs", 150))
        pac = int(self.config.get("pac", 10))
        batch_size = int(self.config.get("batch_size", 500))

        # CTGAN requires batch_size to be a multiple of pac
        if batch_size % pac != 0:
            batch_size = max(pac, (batch_size // pac) * pac)

        gen_dim = tuple(self.config.get("generator_dim", [256, 256]))
        dis_dim = tuple(self.config.get("discriminator_dim", [256, 256]))

        try:
            self.model = ctgan.CTGAN(
                embedding_dim=128,
                generator_dim=gen_dim,
                discriminator_dim=dis_dim,
                batch_size=batch_size,
                epochs=epochs,
                pac=pac,
                verbose=False,
            )

            # Fit model with discrete/categorical columns
            self.model.fit(real_df, discrete_columns=configured_cat_cols)
            self._fitted = True

        except Exception as e:
            self._fitted = False
            raise RuntimeError(f"CTGAN model training failed: {str(e)}") from e

    def sample(self, n_rows: int) -> pd.DataFrame:
        """Generates synthetic rows matching real_df schema and dtypes.

        Args:
            n_rows: Number of synthetic rows to generate.

        Returns:
            pd.DataFrame: Synthetic dataset with matching column names and aligned types.

        Raises:
            RuntimeError: If sample() is called before fit().
            ValueError: If n_rows <= 0.
        """
        if not self._fitted or self.model is None or self._columns is None or self._dtypes is None:
            raise RuntimeError("SyntheticGenerator must be fitted before calling sample().")

        if not isinstance(n_rows, int) or n_rows <= 0:
            raise ValueError(f"n_rows must be a positive integer, got {n_rows}")

        try:
            synth_df = self.model.sample(n_rows)
        except Exception as e:
            raise RuntimeError(f"CTGAN sampling failed: {str(e)}") from e

        # Ensure column ordering strictly matches real_df
        synth_df = synth_df[self._columns].copy()

        # Re-align dtypes to match original data where feasible
        for col in self._columns:
            orig_dtype = self._dtypes[col]
            try:
                if pd.api.types.is_integer_dtype(orig_dtype):
                    # Round and cast to original integer dtype (GAN continuous approximation)
                    synth_df[col] = synth_df[col].round().astype(orig_dtype)
                elif pd.api.types.is_float_dtype(orig_dtype):
                    synth_df[col] = synth_df[col].astype(orig_dtype)
                elif isinstance(orig_dtype, pd.CategoricalDtype):
                    synth_df[col] = synth_df[col].astype(orig_dtype)
                elif pd.api.types.is_object_dtype(orig_dtype):
                    synth_df[col] = synth_df[col].astype(str)
            except Exception:
                # Keep sampled column unchanged if safe cast fails
                pass

        return synth_df
