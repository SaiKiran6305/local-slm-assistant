FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# PaaS platforms inject the listen port and health-check the bound address.
# Hardcoding one makes the container start and then fail its health check,
# which surfaces as a deploy timeout with a clean application log.
ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
