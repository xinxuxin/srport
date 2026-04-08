import path from "node:path";

import { expect, test } from "@playwright/test";

const sampleImagePath = path.resolve(process.cwd(), "../data/samples/demo_input.png");
const invalidFilePath = path.resolve(process.cwd(), "tests/fixtures/not-image.txt");

test("upload success renders output and updates usage dashboard", async ({ page }) => {
  await page.goto("/");

  await expect(page.getByText("EPNet Demo")).toBeVisible();
  await expect(page.getByText("Usage and runtime history")).toBeVisible();

  const requestsMetric = page.getByTestId("requests-metric");
  const beforeText = (await requestsMetric.textContent()) ?? "";
  const beforeCount = Number(beforeText.match(/\d+/)?.[0] ?? "0");

  await page.getByTestId("upload-input").setInputFiles(sampleImagePath);

  await expect(page.getByTestId("output-image")).toBeVisible({ timeout: 60_000 });
  await expect(page.getByTestId("compare-slider")).toBeVisible();
  await page.waitForTimeout(1800);
  await page.getByTestId("compare-slider").fill("75");
  await expect(page.getByTestId("compare-slider")).toHaveValue("75");

  await expect
    .poll(async () => {
      const afterText = (await requestsMetric.textContent()) ?? "";
      return Number(afterText.match(/\d+/)?.[0] ?? "0");
    })
    .toBeGreaterThanOrEqual(beforeCount + 1);
});

test("invalid upload surfaces a friendly error", async ({ page }) => {
  await page.goto("/");

  await page.getByTestId("upload-input").setInputFiles(invalidFilePath);
  await expect(page.getByTestId("upload-error")).toContainText(
    "Please upload a PNG, JPEG, WEBP, or BMP image."
  );
});

test("refresh retains dashboard without crashing after inference", async ({ page }) => {
  await page.goto("/");
  await page.getByTestId("upload-input").setInputFiles(sampleImagePath);
  await expect(page.getByTestId("output-image")).toBeVisible({ timeout: 60_000 });

  await page.reload();

  await expect(page.getByText("Usage and runtime history")).toBeVisible();
  await expect(page.getByText("Inference activity")).toBeVisible();
});

test("batch mode renders aggregate stats", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Batch queue" }).click();
  await page
    .getByTestId("upload-input")
    .setInputFiles([sampleImagePath, sampleImagePath]);

  await expect(page.getByText("2/2")).toBeVisible({ timeout: 60_000 });
  await expect(page.getByText("Batch aggregate")).toBeVisible();
  await expect(page.getByText("Output MP")).toBeVisible();
});
