# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.6.0] - 2026-10-13

### Security

- dashboard API: `POST /api/run/:stage` now accepts only the `--event {"ticker","date"}` argument the UI sends (validated and normalised); arbitrary engine argv from remote callers is rejected with 400
- only one pipeline stage runs at a time server-side (409 with the running job id otherwise), and background GDELT polls are single-flight, so a public demo box cannot be flooded with Python processes

### Added

- `GET /api/health` liveness probe and a cron self-heal in `scripts/auto-update.sh`
- node test suite for the dashboard server (`npm test`), run in CI
- accessible labels on the dashboard's ticker, date, event and watchlist inputs
- optional extras nlp (spaCy), tabpfn (TabPFN/torch), gdelt (aiohttp, tenacity) and dev (pytest)
- Dependabot for pip, npm and GitHub Actions (weekly)
- .env.example, SECURITY.md and this CHANGELOG

### Fixed

- Express 5 wildcard route crash that took the dashboard down
- gdelt tests skip cleanly when gdelt-client is not installed; Dependabot ignores TabPFN major bumps
- dogfood fixes from a live event run (#97)
- pyproject now declares runtime dependencies; pip install -e . previously installed none, so tests could not import numpy/pandas

### Removed

- generated src/signal_lab.egg-info from version control
