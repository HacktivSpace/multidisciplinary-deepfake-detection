FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO

WORKDIR /app

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    curl \
    && apt-get clean

RUN python -m nltk.downloader punkt stopwords wordnet

COPY . /app/

RUN mkdir -p /app/logs

EXPOSE 8000

COPY entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh

ENV DATABASE_URL=${**[]**}
ENV SECRET_KEY=${**[]**}

RUN /app/entrypoint.sh python manage.py migrate
RUN /app/entrypoint.sh python manage.py collectstatic --noinput

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 CMD curl -f http://localhost:8000/health || exit 1

RUN apt-get purge -y --auto-remove build-essential libssl-dev libffi-dev python3-dev curl && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python", "src/main.py"]
