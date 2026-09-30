import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api";

/** A result is valid only for the inputs that started its request. */
export function useOfflineTool<T>(token: string, path: string) {
  const [result, setResult] = useState<T | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const current = useRef<AbortController | null>(null);

  const reset = useCallback(() => {
    current.current?.abort();
    current.current = null;
    setResult(null);
    setError("");
    setBusy(false);
  }, []);

  useEffect(() => {
    reset();
    return () => current.current?.abort();
  }, [token, path, reset]);

  async function run(input: unknown) {
    reset();
    const controller = new AbortController();
    current.current = controller;
    setBusy(true);
    try {
      const data = await api<T>(token, path, {
        method: "POST",
        body: JSON.stringify(input),
        signal: controller.signal,
      });
      if (current.current === controller && !controller.signal.aborted)
        setResult(data);
    } catch (exc) {
      if (current.current === controller && !controller.signal.aborted)
        setError(exc instanceof Error ? exc.message : "分析失败");
    } finally {
      if (current.current === controller) {
        current.current = null;
        setBusy(false);
      }
    }
  }

  return { result, error, busy, reset, run };
}
