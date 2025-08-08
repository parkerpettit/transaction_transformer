import argparse
from pathlib import Path
import wandb

from transaction_transformer.config.config import ConfigManager


def main():
    parser = argparse.ArgumentParser(description="Upload raw CSV as a W&B artifact")
    parser.add_argument("--config", type=str, default="pretrain.yaml", help="Path to config YAML (relative to config dir)")
    parser.add_argument("--artifact-name", type=str, default="raw-card-transactions-v1", help="Artifact name")
    parser.add_argument("--project", type=str, default=None, help="W&B project override")
    args = parser.parse_args()

    cfg = ConfigManager(config_path=args.config).config

    project = args.project or cfg.metrics.wandb_project
    run = wandb.init(project=project, job_type="upload_raw_data", name="upload_raw_data")

    csv_path = Path(cfg.model.data.raw_csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found at {csv_path}")

    art = wandb.Artifact(name=args.artifact_name, type="dataset", description="Raw card transactions CSV")
    art.add_file(str(csv_path), name=csv_path.name)
    run.log_artifact(art, aliases=["latest"])  # add more aliases if needed
    print(f"Logged artifact {args.artifact_name} with aliases [latest]")
    # Explicitly finish the run to ensure all data is uploaded and the run is closed
    wandb.finish()
    run.finish()

if __name__ == "__main__":
    main()


