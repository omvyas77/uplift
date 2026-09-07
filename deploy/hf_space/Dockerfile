FROM python:3.12-slim

# HF Spaces run as uid 1000. Creating the user explicitly keeps file ownership
# sane and avoids Streamlit trying to write config into a root-owned home.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

WORKDIR $HOME/app

COPY --chown=user requirements.txt ./
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user app.py ./
COPY --chown=user src/ src/
COPY --chown=user evals/ evals/

# 7860 is the port HF Spaces expects; app_port in the README frontmatter must
# match. Headless + no CORS is required for the Space's reverse proxy.
EXPOSE 7860
CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false", \
     "--browser.gatherUsageStats=false"]
