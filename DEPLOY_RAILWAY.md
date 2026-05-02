# Deploying ForeSee to Railway

This project should be deployed as **two Railway services**:

1. `backend`: Flask API
2. `frontend`: Vite static site

## 1. Backend service

Create a Railway service from this repo and set its **root directory** to `.`.

Use these settings:

- Build command:
  `pip install -r requirements.txt`
- Start command:
  `gunicorn --chdir backend wsgi:app --bind 0.0.0.0:$PORT --timeout 300`

Set these environment variables in Railway:

- `SNOWFLAKE_ACCOUNT`
- `SNOWFLAKE_USER`
- `SNOWFLAKE_PASSWORD`
- `SNOWFLAKE_DATABASE`
- `SNOWFLAKE_SCHEMA`
- `INGESTION_WAREHOUSE`
- `GEMINI_API_KEY`
- `FLASK_DEBUG=false`

After deploy, copy the backend public URL, for example:

`https://foresee-backend-production.up.railway.app`

## 2. Frontend service

Create a second Railway service from the same repo and set its **root directory** to `frontend`.

Use these settings:

- Build command:
  `npm ci && npm run build`
- Start command:
  `npm run start:railway`

Set this environment variable:

- `VITE_API_BASE_URL=https://your-backend-url/api`

Replace `https://your-backend-url` with the real backend Railway URL.

## 3. Important notes

- Railway filesystem is ephemeral. Generated PDFs inside `backend/pdf/` are fine for short-term access, but they are not durable across redeploys or restarts.
- Large ML dependencies can make the backend build slower.
- Your frontend must point to the backend Railway URL. It cannot stay on `localhost` in production.
- If Snowflake or Gemini credentials are missing, upload and analysis endpoints will fail at runtime.

## 4. Local dev vs production

Local development still works without extra config because the frontend falls back to:

`http://localhost:5000/api`

Production should always set:

`VITE_API_BASE_URL`
