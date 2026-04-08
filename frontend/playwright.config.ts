import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  timeout: 60_000,
  fullyParallel: false,
  use: {
    baseURL: "http://127.0.0.1:3100",
    trace: "retain-on-failure"
  },
  webServer: [
    {
      command:
        "EPNET_CORS_ORIGINS=http://127.0.0.1:3100,http://localhost:3100 PYTHONPATH=src .venv/bin/uvicorn epnet_api.app.main:app --host 127.0.0.1 --port 8100",
      cwd: "..",
      url: "http://127.0.0.1:8100/api/v1/health",
      reuseExistingServer: false,
      timeout: 90_000
    },
    {
      command: "NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8100/api/v1 npm run dev -- --hostname 127.0.0.1 --port 3100",
      cwd: ".",
      url: "http://127.0.0.1:3100",
      reuseExistingServer: false,
      timeout: 90_000
    }
  ],
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] }
    }
  ]
});
