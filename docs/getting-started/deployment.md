# Deployment Guide

## Automated Deployment (Recommended)

1. Push changes to the `main` branch on GitHub
2. The workflow in `.github/workflows/gh-pages.yml` will automatically:
   - Build the documentation with MkDocs
   - Deploy to the `gh-pages` branch
   - Publish at `https://frbennett.github.io/shapleyx/`

The workflow triggers on changes to `docs/**`, `mkdocs.yml`, or the
workflow file itself.

## Manual Deployment (Fallback)

If automated deployment fails:

1. Install documentation dependencies:
```bash
pip install mkdocs mkdocstrings[python] mkdocs-jupyter pymdown-extensions
```

2. Build and deploy:
```bash
mkdocs gh-deploy --force --remote-branch gh-pages
```

3. If permission errors persist, ensure your GitHub token has write
   access to the repository.

## Testing Locally

```bash
mkdocs serve
```

Then open `http://127.0.0.1:8000/` in your browser.  Changes to the
`docs/` directory are reflected in real time.

## Troubleshooting

### Permission Errors
- Ensure you have push access to the repository
- The automated workflow uses `GITHUB_TOKEN` with `contents: write`
- For manual deployment, configure a personal access token

### Build Errors
- Check for broken links in your documentation
- Verify all Python modules referenced in API docs are importable
  (the workflow runs `pip install .` before building)
- Run `mkdocs serve` locally to test before deploying

### Notebook Rendering Issues
- Notebooks are converted with `mkdocs-jupyter` using `execute=False`
- Large notebooks may take time to convert; the workflow caches
  converted notebooks in `.cache/mkdocs-jupyter/`
