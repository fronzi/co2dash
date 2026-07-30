# Publishing the demo on Streamlit Community Cloud

A 15-minute path to a public URL, for a **public/demo-data** instance. Do **not**
put confidential or partner data on this public demo (see the data note at the
bottom).

## What's already prepared

- `requirements.txt` (repo root) — installs the `co2dash` package + Streamlit + Plotly.
- `app/streamlit_app.py` — the entry point.
- `.streamlit/config.toml` — theme + server settings.
- No system packages needed (`packages.txt` not required).

## Steps

1. **Put the code on GitHub.** Create a repository (public, or private — Streamlit
   Cloud supports both) and push the *contents of the `co2dash/` folder* so that
   `pyproject.toml`, `requirements.txt`, `app/`, and `src/` are at the repo root.

   ```bash
   cd co2dash
   git init && git add . && git commit -m "co2dash demo"
   git branch -M main
   git remote add origin https://github.com/<you>/co2dash.git
   git push -u origin main
   ```

2. **Sign in** at https://share.streamlit.io with your GitHub account and authorise it.

3. **New app → Deploy a public app from GitHub.** Set:
   - Repository: `<you>/co2dash`
   - Branch: `main`
   - Main file path: `app/streamlit_app.py`

4. **Deploy.** Streamlit runs `pip install -r requirements.txt` (installs the
   package) and launches the app. First build takes a few minutes.

5. **Secrets (only if you enable live connectors).** The demo needs none. If you
   later wire ElectricityMaps, add the token under *App → Settings → Secrets*
   (never in the repo):
   ```toml
   ELECTRICITYMAPS_TOKEN = "..."
   ```

6. **Share the URL** (`https://<app-name>.streamlit.app`).

## Expected behaviour / limits of the free tier

- The app runs on slider + YAML input out of the box (no data files needed).
- Free resources are modest (~1 GB RAM); the app **sleeps when idle** and wakes on
  the next visit (a few seconds). Fine for a consortium demo, not for heavy public
  traffic.
- WebSockets and HTTPS are handled by Streamlit Cloud — nothing to configure.

## Updating

Push to `main`; the app redeploys automatically.

## Data & privacy note (important)

This is a **public** endpoint on third-party (non-USyd) infrastructure. Use it only
with public or synthetic inputs. The "Your data" tab processes uploads **in the
running session** and the app persists nothing by default — but do not upload
confidential catalyst data or partner data to the public demo. For real/partner
data, use the USyd-hosted instance (see `DEPLOY.md`) after the data-governance
review.
