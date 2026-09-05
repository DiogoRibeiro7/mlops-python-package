# prepare and train; evaluate an explicit candidate before separate promotion
[group('project')]
project: project-environment (project-run "tuning") (project-run "training")

# export both runtime environment files from the lockfile
[group('project')]
project-environment:
    python scripts/export_environment.py

# compatibility alias: both files must be regenerated together
[group('project')]
project-requirements: project-environment

# run project job using mlflow
[group('project')]
project-run job:
    uv run mlflow run --experiment-name={{REPOSITORY}} --run-name={{capitalize(job)}} -P conf_file=confs/{{job}}.yaml .
