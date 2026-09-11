FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY pyproject.toml README.md requirements.lock ./
COPY mare ./mare
COPY mare_web ./mare_web
COPY mare_planner ./mare_planner
ARG INSTALL_PLANNER=false
RUN if [ "$INSTALL_PLANNER" = "true" ]; then pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu; fi
RUN pip install -c requirements.lock '.[openai]' && useradd --uid 10001 --create-home mare
COPY migrations ./migrations
COPY alembic.ini ./
COPY deploy/entrypoint.py ./deploy/entrypoint.py
USER 10001
ENTRYPOINT ["python", "deploy/entrypoint.py"]
CMD ["uvicorn", "mare_web.api:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
