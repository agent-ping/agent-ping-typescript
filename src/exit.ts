import { getState } from "./state.js";

let registered = false;

export function registerExitHandler(): void {
  if (registered) return;
  if (typeof process === "undefined" || !process.on) return;
  registered = true;

  const handler = async (): Promise<void> => {
    try {
      const state = getState();
      if (!state) return;
      await state.worker.flushAll(5_000);
    } catch {
      // never throw from exit handler
    }
  };

  process.on("beforeExit", () => {
    void handler();
  });
}
