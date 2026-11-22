# Release cycle

`llmcall` ships on two tracks.

| Track | When | Version | Where |
|-------|------|---------|-------|
| Nightly | Daily, 03:00 UTC, only if `main` changed | `X.Y.Z.devYYYYMMDDHHMM` | [`nightly` GitHub pre-release](https://github.com/rihoneailabs/llmcall/releases/tag/nightly), optionally TestPyPI |
| Release | Every other Monday, 06:00 UTC | `X.Y.Z` from Conventional Commits | GitHub Release + PyPI |

## Nightly builds

[`nightly.yml`](../.github/workflows/nightly.yml) runs the full test matrix and `pip-audit` (via [`tests.yml`](../.github/workflows/tests.yml)), builds the wheel and sdist, smoke-tests the wheel, and replaces the rolling `nightly` pre-release. `X.Y.Z` is the version the next release would get, so a nightly always sorts below the release it precedes.

If `main` hasn't changed since the last successful nightly, the run is skipped. A failed nightly doesn't move the `nightly` tag, so it is retried the next night. To build on demand, run the workflow manually with `force`.

Install the latest nightly:

```bash
pip install "llmcall @ https://github.com/rihoneailabs/llmcall/releases/download/nightly/<wheel-file>"
# or, if TestPyPI publishing is enabled:
pip install --pre -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ llmcall
```

To enable TestPyPI publishing, add a trusted publisher on test.pypi.org for workflow `nightly.yml` and environment `testpypi`, then set the repository variable `NIGHTLY_TESTPYPI=true`.

## Biweekly releases

[`semantic-release.yml`](../.github/workflows/semantic-release.yml) runs every Monday. If the last `v*` tag is at least 7 days old, it computes the next version from Conventional Commits since that tag and opens (or updates) a `chore(release): bump version to X.Y.Z` PR. The 7-day threshold skips the off-week, so releases land every other Monday, and an off-cycle release re-anchors the cadence.

Only `feat`, `fix`, `perf` and breaking changes produce a release. A cycle with only `chore`/`ci`/`docs` commits (including `chore(deps)` bumps) ships nothing.

When the bump PR is merged (squash, keeping its title), the push to `main` tags `vX.Y.Z` and creates the GitHub Release with generated notes. Publishing that release triggers [`python-publish.yml`](../.github/workflows/python-publish.yml), which uploads to PyPI.

- With the `RELEASE_PAT` secret set, the release is published straight away and PyPI upload follows automatically.
- Without it, the release is created as a **draft**, because releases created by `GITHUB_TOKEN` do not trigger other workflows. Click **Publish** on the draft to upload to PyPI.

To cut a release off-cycle (for example, a security fix), run **Semantic Release** manually with `force`.
