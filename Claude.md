# CLAUDE.md

## Self-Improvement

Whenever you make a mistake or misinterpret something and are corrected, add a section at the bottom of this file under `## Lessons Learned` to avoid repeating it.

-----

## Response Style

- Explain what you are doing and why, simply and clearly, as you go
- Prefer short explanations over long ones — but never skip the reasoning
- When making a code change, state in plain English: what you changed, why, and what to watch out for
- Do not pad responses with summaries of what you just did

-----

## Tools

Always use `uv` for Python. Never use `pip install` or run Python directly.

|Task             |Command                    |
|-----------------|---------------------------|
|Install a package|`uv add <package>`         |
|Remove a package |`uv remove <package>`      |
|Run a script     |`uv run <script.py>`       |
|Format code      |`uv run ruff format <file>`|
|Lint/check code  |`uv run ruff check <file>` |
|Type check       |`uv run ty <file>`         |
|Run tests        |`uv run pytest`            |

Run `ruff check` and `ty` on any new or edited file before running it. Fix all warnings and errors. Do not suppress types with `# noqa` or `# type: ignore`.

-----

## Code Guidelines

- Google docstring format
- Follow SOLID principles where practical
- Functionality over test coverage — but write pytest tests when logic is non-trivial
- Surgical additions preferred — do not refactor unless asked
- Discuss and justify design decisions before building

-----

## Project Context

### Repos

- `dailytrade` — daily macro trade idea generator, HTML email output
- `macro_claude` — macro regime classifier and RV trade idea generator (active development); `macro_claude.py` is the main file, evolved from `macro_engine.py`

### Data Sources & Constraints

- Yahoo Finance — equities and commodity proxies
- FRED — macro data
- Alpha Vantage — news sentiment; **free tier = 25 API calls/day, cache aggressively**
- exchangerate.host — FX pairs

### Design Principles

- Keep `dailytrade` and `macro_claude` architecturally separate
- FX trades always expressed as pairs, never single-leg directional
- Prefer incremental, non-disruptive additions over rewrites

-----

## Lessons Learned

<!-- New entries added here after corrections -->
