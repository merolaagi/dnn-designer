import { test, expect } from "@playwright/test";

test("create research, inspect every view, graph keyboard access and mobile layout", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: "Research dashboard" }),
  ).toBeVisible();
  await page.screenshot({
    path: "../docs/screenshots/dashboard.png",
    fullPage: true,
  });
  await page
    .getByRole("link", { name: "Create run", exact: true })
    .first()
    .click();
  await page.getByLabel("Run title").fill("Browser test · Burgers benchmark");
  await page.getByLabel("Rounds", { exact: true }).fill("4");
  await page.getByRole("button", { name: "Queue research run" }).click();
  await expect(page).toHaveURL(/\/runs\/RUN-/);
  await expect(page.locator(".run-meta .badge.completed").first()).toBeVisible({
    timeout: 45000,
  });
  await page.screenshot({
    path: "../docs/screenshots/run-overview.png",
    fullPage: true,
  });
  for (const tab of [
    "Branches",
    "Claims & evidence",
    "Failures",
    "Questions",
    "Proof DAG",
    "Neural policy",
    "Budgets & costs",
    "Logs & events",
  ]) {
    await page.getByRole("link", { name: tab, exact: true }).click();
    await expect(page.locator("main")).not.toContainText(
      "Internal service error",
    );
    if (tab === "Claims & evidence") {
      await expect(page.locator(".claim-card").first()).toBeVisible();
      await page.locator(".claim-card summary").first().click();
      await expect(page.locator(".claim-card .evidence").first()).toBeVisible();
      await page.getByLabel("Search claims").fill("no-such-result-zzz");
      await expect(page.getByText("No matching claims.")).toBeVisible();
    }
    if (tab === "Proof DAG") {
      await expect(
        page.getByRole("img", { name: "Interactive proof dependency graph" }),
      ).toBeVisible();
      await page.locator(".graph-node").first().focus();
      await page.keyboard.press("Enter");
      await expect(page.locator(".claim-card")).toBeVisible();
      await page.screenshot({
        path: "../docs/screenshots/proof-dag.png",
        fullPage: true,
      });
    }
    if (tab === "Neural policy")
      await page.screenshot({
        path: "../docs/screenshots/neural-policy.png",
        fullPage: true,
      });
  }
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: "Research dashboard" }),
  ).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBeTruthy();
  await page.screenshot({
    path: "../docs/screenshots/mobile.png",
    fullPage: true,
  });
  expect(errors).toEqual([]);
});

test("API error state offers retry", async ({ page }) => {
  await page.route("**/api/runs", (route) =>
    route.fulfill({
      status: 503,
      contentType: "application/json",
      body: JSON.stringify({
        error: { message: "Temporarily unavailable", request_id: "test-error" },
      }),
    }),
  );
  await page.goto("/");
  await expect(page.getByRole("alert")).toContainText(
    "Temporarily unavailable",
  );
  await expect(page.getByRole("button", { name: "Try again" })).toBeVisible();
});
