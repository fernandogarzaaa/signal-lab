# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- optional extras nlp (spaCy), tabpfn (TabPFN/torch), gdelt (aiohttp, tenacity) and dev (pytest)
- Dependabot for pip, npm and GitHub Actions (weekly)
- .env.example, SECURITY.md and this CHANGELOG

### Fixed

- pyproject now declares runtime dependencies; pip install -e . previously installed none, so tests could not import numpy/pandas

### Removed

- generated src/signal_lab.egg-info from version control
