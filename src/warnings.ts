const warned = new Set<string>();

export function warnOnce(cls: string, message: string): void {
  if (warned.has(cls)) return;
  warned.add(cls);
  try {
    // eslint-disable-next-line no-console
    console.warn(`[agentping] ${cls}: ${message}`);
  } catch {
    // Never let logging crash user code.
  }
}

export function _resetWarningsForTests(): void {
  warned.clear();
}
