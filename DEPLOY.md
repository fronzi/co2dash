# Deploying co2dash

co2dash's dashboard is a running Python (Streamlit) process, not a static site —
it computes on the server (Monte-Carlo, Sobol, calibration). So it needs a host
that runs a process, and a place for secrets and cached data. Two paths:

## A. Fast partner demo (Streamlit Community Cloud / Hugging Face Spaces)

Best for sharing a link with the AJF partners. No ops.

1. Push the repo to GitHub.
2. Point share.streamlit.io (or a HF Space) at `app/streamlit_app.py`.
3. Dependencies install from `requirements.txt` / `pyproject.toml`.
4. Put any API token (e.g. `ELECTRICITYMAPS_TOKEN`) in the platform's **secrets**
   manager — never in the repo.
5. The app runs on slider/YAML input out of the box. Live connectors and the
   descriptor/FE data steps run separately (see the examples) and their cached
   outputs are read by the app.

## B. Durable, controlled (container on a USyd research VM)

Best once it's infrastructure rather than a demo.

1. `docker build -t co2dash .`
2. `docker run -p 8501:8501 --env-file .env co2dash` (tokens live in `.env`,
   which is **not** committed).
3. Front with a reverse proxy (HTTPS) on a USyd subdomain; a Nectar/research-
   cloud VM works. Persist the cached descriptor/data files as a mounted volume
   (version them with DVC so the deployed app and your laptop agree).

## Secrets & data (both paths)

- Tokens → host secret store / `.env`, never the image or repo.
- The app reads a **cached** descriptor/data file; the fetch + calibration steps
  (`examples/…`) run beforehand and produce that cache.
- No user data is persisted by the app unless you add it deliberately.

## Before any public URL — institutional clearance (do NOT skip)

This is an Australia–Japan bilateral project that may ingest users' data. A
USyd-hosted, internet-facing instance touches USyd's **foreign-interference and
privacy** review. Clear this with ICT / research office **before** exposing a
public endpoint — it is a sign-off, not a code change. A private, link-shared
partner demo (path A) is the low-risk first step.

## Readiness checklist

- [x] Package installs, 80+ tests pass, engine validated (see docs/VALIDATION.md)
- [x] Container + deploy paths documented
- [ ] Hosted instance stood up and load-tested (never done yet)
- [ ] Secrets configured on the host
- [ ] Data-governance / foreign-interference review cleared
- [ ] Descriptor→activity model finished on real (public) data before outputs are
      presented as trustworthy to others
