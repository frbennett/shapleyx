# Deployment Guide

## Automated Deployment (Recommended)
1. Push changes to GitHub
2. The workflow in `.github/workflows/gh-pages.yml` will automatically:
   - Build the documentation
   - Deploy to the `gh-pages` branch
   - Make it available at `https://[your-username].github.io/shapleyx/`

## Manual Deployment (Fallback)
If automated deployment fails:

1. Install requirements:
```bash
pip install mkdocs-material mkdocstrings[python]
```

2. Build and deploy:
```bash
mkdocs gh-deploy --force
```

3. If permission errors persist:
```bash
git config --global user.name "Your Name"
git config --global user.email "your@email.com"
mkdocs gh-deploy --force --remote-branch gh-pages
```

## Troubleshooting

### Permission Errors
- Ensure you have push access to the repository
- Verify your GitHub token has repo permissions
- Try creating a personal access token with repo scope

### Build Errors
- Check for broken links in your docs
- Verify all Python modules referenced in API docs exist
- Run `mkdocs serve` locally to test before deploying