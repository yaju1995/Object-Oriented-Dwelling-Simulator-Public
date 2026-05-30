import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from support.probability_lib import ProbabilityDistributions  # your file :contentReference[oaicite:0]{index=0}
from support.lib_config import CustomLogger

logger = CustomLogger(command=False)


class PatternGenerationHandler:
    """
    Generates 24h stochastic patterns from CSV-defined distributions
    """

    MODEL_COLUMNS = {
        "normal": ["mean", "std"],
        "lognormal": ["mu", "sigma"],
        "custom_lognormal": ["mu", "sigma", "alpha"],
        "exponential": ["lambda_"],
        "gamma": ["shape", "scale"],
        "uniform": ["a", "b"],
        "gmm": ["weights", "mus", "sigmas"]
    }

    def __init__(self, model_name: str, csv_path: str):
        self.model_name = model_name
        self.dist = ProbabilityDistributions()
        self.df = self._load_and_validate(csv_path)

    # ------------------------------------------------------------------
    def _load_and_validate(self, csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path)

        if "time" not in df.columns:
            raise ValueError("CSV must contain a 'time' column (hh:mm:ss)")

        df["time"] = pd.to_timedelta(df["time"])
        df = df.set_index("time")

        if self.model_name not in self.MODEL_COLUMNS:
            raise ValueError(f"Unsupported model: {self.model_name}")

        required = self.MODEL_COLUMNS[self.model_name]
        missing = set(required) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns for {self.model_name}: {missing}")

        return df.sort_index()

    # ------------------------------------------------------------------
    def _enforce_non_negative(self, value: float) -> float:
        """Clamp generated value to non-negative domain"""
        if value < 0:
            logger.commandline(
                f"[PatternGen] Negative value clipped: {value:.4f} → 0.0"
            )
            return 0.0
        return value

    # ------------------------------------------------------------------
    def generate_24h(
            self,
            resolution: timedelta,
            aggregate: str = "mean",
            seed: int | None = None,
            intra_noise_std: float = 0.1,  # <-- your white noise std
            intra_noise_mean: float = 0.0,  # <-- mean 0
    ) -> np.ndarray:
        """
        Generate 24h samples.

        If resolution is finer than the CSV base resolution:
          - draw ONE base sample per CSV interval
          - for each sub-step inside that interval, add white noise N(0, intra_noise_std)

        If resolution is equal/coarser:
          - keep existing behavior: one sample per target step after (optional) aggregation
        """
        rng = np.random.default_rng(seed)

        # Determine CSV base resolution
        if len(self.df.index) < 2:
            raise ValueError("CSV must contain at least 2 rows to infer base resolution.")
        base_resolution = self.df.index[1] - self.df.index[0]

        # Target 24h timeline
        target_index = pd.timedelta_range(
            start=timedelta(0),
            end=timedelta(hours=24) - resolution,
            freq=resolution
        )

        # -----------------------------
        # Case A: finer than CSV → base draw + white noise
        # -----------------------------
        if resolution < base_resolution:
            # Build a base timeline at CSV resolution for a full 24h
            base_index = pd.timedelta_range(
                start=timedelta(0),
                end=timedelta(hours=24) - base_resolution,
                freq=base_resolution
            )

            # Align CSV params to base timeline (ffill covers missing)
            base_params = self.df.reindex(base_index, method="ffill")

            # Draw ONE sample per base interval
            base_values = []
            for _, row in base_params.iterrows():
                # Use your distribution sampler, but keep RNG reproducible
                # by temporarily drawing u from rng and using inv_transform directly.
                # Easiest drop-in: call your existing method, but it uses np.random.rand().
                # To keep reproducible with rng, we emulate it here:
                u = rng.random()
                base_val = self.dist.get_inv_transform(self.model_name, u, **row.to_dict())
                base_val = self._enforce_non_negative(base_val)
                base_values.append(base_val)
            base_values = np.array(base_values)

            # Map each fine timestep to its base interval index
            samples = []
            steps_per_base = int(base_resolution / resolution)
            if steps_per_base * resolution != base_resolution:
                raise ValueError(
                    "resolution must divide the CSV base resolution exactly for intra-interval noise mode.")

            for i, t in enumerate(target_index):
                base_i = int(t / base_resolution)  # which base interval
                val = base_values[base_i] + rng.normal(loc=intra_noise_mean, scale=intra_noise_std)
                val = self._enforce_non_negative(val)
                samples.append(round(val, 3))

            return np.array(samples)

        # -----------------------------
        # Case B: equal/coarser than CSV → old behavior (one draw per target step)
        # -----------------------------
        if resolution > base_resolution:
            rule = pd.Timedelta(resolution)
            if aggregate == "mean":
                params = self.df.resample(rule).mean()
            elif aggregate == "sum":
                params = self.df.resample(rule).sum()
            else:
                raise ValueError("aggregate must be 'mean' or 'sum'")
            params = params.reindex(target_index, method="ffill")
        else:
            params = self.df.reindex(target_index, method="ffill")

        samples = []
        for _, row in params.iterrows():
            u = rng.random()
            val = self.dist.get_inv_transform(self.model_name, u, **row.to_dict())
            val = self._enforce_non_negative(val)
            samples.append(round(val, 3))

        return np.array(samples)

    def get_simulation_data(
            self,
            start_date: datetime,
            duration_days: timedelta,
            resolution: timedelta,
            seed: int | None = None,
            aggregate: str = "mean",
            column_name: str = "value"
    ) -> pd.DataFrame:
        """
        Generate simulation data aligned to calendar midnights:
        from 00:00:00 of start day to 00:00:00 of end day
        """
        if duration_days <= timedelta(0):
            raise ValueError("duration_days must be positive")

        if resolution <= timedelta(0):
            raise ValueError("resolution must be positive")

        # ---- Force midnight alignment
        start_date = datetime.combine(start_date.date(), datetime.min.time())
        end_date = start_date + duration_days

        # ---- Full datetime index
        datetime_index = pd.date_range(
            start=start_date,
            end=end_date,
            freq=resolution,
            # inclusive="left"
        )

        steps_per_day = int(timedelta(days=1) / resolution)
        if steps_per_day * resolution != timedelta(days=1):
            raise ValueError("resolution must divide 24h exactly")

        values = []

        current_day = None
        daily_profile = None

        for ts in datetime_index:
            day_start = datetime.combine(ts.date(), datetime.min.time())

            if current_day != day_start:
                current_day = day_start
                day_idx = (current_day - start_date).days
                day_seed = None if seed is None else seed + day_idx

                daily_profile = self.generate_24h(
                    resolution=resolution,
                    aggregate=aggregate,
                    seed=day_seed
                )

            step_in_day = int((ts - current_day) / resolution)
            values.append(daily_profile[step_in_day])

        return pd.DataFrame(
            {column_name: np.array(values)},
            index=datetime_index
        )

    def visualize_csv_pattern(self):
        """
        Visualize the parameter pattern defined in the CSV
        (no randomness, just model parameters over time)
        """

        df = self.df.copy()
        hours = df.index.total_seconds() / 3600

        plt.figure(figsize=(12, 5))

        if self.model_name == "normal":
            plt.plot(hours, df["mean"], label="Mean")
            plt.fill_between(
                hours,
                df["mean"] - df["std"],
                df["mean"] + df["std"],
                alpha=0.3,
                label="±1 Std"
            )

        elif self.model_name == "uniform":
            plt.plot(hours, df["a"], label="a (min)")
            plt.plot(hours, df["b"], label="b (max)")
            plt.fill_between(hours, df["a"], df["b"], alpha=0.3)

        elif self.model_name == "lognormal":
            plt.plot(hours, df["mu"], label="mu")
            plt.plot(hours, df["sigma"], label="sigma")

        elif self.model_name == "custom_lognormal":
            plt.plot(hours, df["mu"], label="mu")
            plt.plot(hours, df["sigma"], label="sigma")
            plt.plot(hours, df["alpha"], label="alpha")

        elif self.model_name == "gamma":
            mean = df["shape"] * df["scale"]
            plt.plot(hours, mean, label="Mean (shape×scale)")
            plt.plot(hours, df["shape"], "--", label="shape")
            plt.plot(hours, df["scale"], "--", label="scale")

        elif self.model_name == "exponential":
            mean = 1 / df["lambda_"]
            plt.plot(hours, mean, label="Mean (1/lambda)")

        elif self.model_name == "gmm":
            # assumes mus is stored as stringified list
            for i, mus in enumerate(df["mus"]):
                mus = np.array(eval(mus))
                plt.plot(hours[i], mus.mean(), "o", label="GMM mean" if i == 0 else "")

        else:
            raise NotImplementedError(
                f"Visualization not implemented for {self.model_name}"
            )

        plt.xlabel("Hour of Day")
        plt.ylabel("Parameter Value")
        plt.title(f"CSV Pattern Visualization ({self.model_name})")
        plt.xlim(0, 24)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
