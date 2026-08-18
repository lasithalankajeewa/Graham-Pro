# Graham Pro — Next.js

Next.js migration of the Streamlit Graham-Bot. The original Python project remains unchanged.

## Local development

```bash
npm install
npm run dev
```

Without an environment file, local development opens `../graham_bot.db`, so existing users and analyses remain available. Copy `.env.example` to `.env.local` to override keys or database settings.

## Vercel deployment

Vercel's filesystem is ephemeral, so production must use the PostgreSQL `DATABASE_URL` supported by both this app and `core/db.py`. Set these project variables in Vercel:

- `DATABASE_URL`
- `SECRET_KEY`
- `GEMINI_API_KEY` or `OPENROUTER_API_KEY`
- SMTP variables if the existing Python alert pipeline sends email

Set the Vercel project root directory to `graham-next`. The existing Python pipeline can keep using the same PostgreSQL URL; automatic analyses will appear in the Next.js dashboard.

## Verification

```bash
npm run build
npm audit --omit=dev
```
