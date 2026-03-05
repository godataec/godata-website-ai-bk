# 1. Use the official Microsoft image (it already has Playwright + Python)
FROM mcr.microsoft.com/playwright/python:v1.48.0-jammy

# 2. Set working directory
WORKDIR /app

# 3. Copy only requirements first to leverage Docker Cache
COPY requirements.txt .

# 4. Install dependencies (This layer will be cached unless you change requirements.txt)
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of the app (The .dockerignore makes this instant)
COPY . .

# 6. Final settings
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--loop", "asyncio"]