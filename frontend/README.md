# React + TypeScript + Vite + shadcn/ui

The chat UI calls the FastAPI companion API (`POST /api/chat`). Copy `.env.example` to `.env` and set `VITE_API_URL` if the API is not on `http://127.0.0.1:8000`.

This is a template for a new Vite project with React, TypeScript, and shadcn/ui.

## Adding components

To add components to your app, run the following command:

```bash
npx shadcn@latest add button
```

This will place the ui components in the `src/components` directory.

## Using components

To use the components in your app, import them as follows:

```tsx
import { Button } from "@/components/ui/button"
```
