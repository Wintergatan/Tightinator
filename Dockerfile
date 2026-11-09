# Pinned by digest so rebuilds are reproducible. To update, pull the newest
# python:3.12-slim and replace the digest.
FROM python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b

# FORWARDED_ALLOW_IPS: gunicorn only trusts X-Forwarded-* headers from these
# addresses. The app builds relative URLs only, so it doesn't rely on them.
# Override at deploy time with the reverse proxy's network if needed.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FORWARDED_ALLOW_IPS=127.0.0.1,::1

WORKDIR /opt

# curl is used by the Coolify container health check.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

# Install hash-verified dependencies (see requirements.lock).
COPY requirements.lock /opt/requirements.lock
RUN pip3 install --require-hashes -r requirements.lock

# Copy files. Code stays owned by root so the app user can't modify it.
COPY web /opt/web/
COPY main.py /opt/web/main.py

# We'll run everything under a regular user acct, not root.
RUN adduser wilson --home /opt --disabled-password
# Uploads/results live here; mount a persistent volume at this path.
# This is the only directory the app user can write to.
RUN mkdir -p /opt/web/static/upload \
 && chown wilson:wilson /opt/web/static/upload
USER wilson

# Start up the app!
WORKDIR /opt/web
EXPOSE 5000/tcp
# One worker keeps the in-process job limit (MAX_CONCURRENT_JOBS) global;
# threads keep slow uploads from blocking other requests.
CMD [ "gunicorn", "--bind", "0.0.0.0:5000", "--worker-class", "gthread", "--workers", "1", "--threads", "8", "--timeout", "120", "app:app"]
