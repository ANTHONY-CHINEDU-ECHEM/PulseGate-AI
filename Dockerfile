# HTTP service image. Inference only: no PyTorch, no training data.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY pulsegate ./pulsegate
RUN pip install --no-cache-dir .

COPY manage.py ./
COPY configs ./configs
COPY models ./models

RUN useradd --create-home pulsegate && chown -R pulsegate /app
USER pulsegate
EXPOSE 8000
CMD ["python", "manage.py", "serve", "api.host=0.0.0.0"]
