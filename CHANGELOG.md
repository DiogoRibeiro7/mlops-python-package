# Independent maintenance: unreleased

- Generate consistent runtime environment exports from uv.lock, retaining platform markers and excluding default development groups.
- Check export freshness in CI and distinguish inherited capabilities from independent changes in the README.

- Establish independent maintenance by Diogo Ribeiro with preserved upstream attribution.
- Remove automatic upstream synchronization, active template linkage and inherited funding configuration.
- Replace upstream project links and container destination; make publication manual.
- Run locked CI installation on pull requests and main.
- Document known modelling and dependency limitations and their acceptance gates.

The entries below are retained historical release notes.

## v4.1.0 (2025-03-05)

### Feat

- **gemini**: add support for gemini code assist (#51)
- **dependabot**: add dependabot configuration file (#50)
- **github**: add default rulesets and installation (#47)

### Fix

- **workflows**: fix just in workflows

### Refactor

- **cruft**: update to new template version

## v4.0.0 (2025-03-04)

### Feat

- **tasks**: switch from pyinvoke to just (#42)
- **workflows**: bump GitHub action versions (#41)
- **versions**: bump python and package version (#40)
- **mindmap**: add mindmap of the package (#32)

### Fix

- **version**: ready to bump
- **datasets**: fix dtype backend (#44)

### Refactor

- **cruft**: update to new template version

## v2.0.0 (2024-07-28)

### Feat

- **cruft**: adopt cruft and link it to cookiecutter-mlops-package

## v1.1.3 (2024-07-28)

### Fix

- **mlproject**: fix calling mlflow run by adding project run in front

## v1.1.2 (2024-07-28)

### Fix

- **dependencies**: add setuptools to main dependency for mlflow

## v1.1.1 (2024-07-23)

### Fix

- **publish**: fix publication workflow by installing dev dependencies

## v1.0.1 (2024-06-28)

### Fix

- **version**: bump
