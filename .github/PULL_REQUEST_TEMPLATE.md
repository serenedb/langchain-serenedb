<!--
PR title will become the squash commit message.
Use conventional commits: feat:, fix:, refactor:, test:, docs:, chore:

This description becomes the commit body -- explain *why*, not *what*.

Before merging, please make sure:
- `pre-commit run --all-files` passes (ruff + mypy)
- the integration suite passes against a SereneDB instance
  (`tests/run_tests_docker.sh`, or `tests/run_tests_local.sh <serened-binary>`)
-->
