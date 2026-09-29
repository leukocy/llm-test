import { useState } from "react";
import { Mark } from "../components";

export function Login({
  onLogin,
}: {
  onLogin: (value: string) => Promise<void>;
}) {
  const [input, setInput] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  return (
    <div className="login-shell">
      <div className="login-art">
        <div className="login-grid" />
        <div className="login-brand">
          <Mark />
          <span>
            LLM TEST <small>测量控制台</small>
          </span>
        </div>
        <div className="login-art-copy">
          <span className="eyebrow">MEASUREMENT INFRASTRUCTURE</span>
          <h1>
            让每一次
            <br />
            模型评估
            <br />
            <em>都有依据。</em>
          </h1>
          <p>稳定的任务执行、可追溯的统计口径与清晰的结果展示。</p>
        </div>
        <div className="login-art-foot">CONTROL PLANE / SINGLE TENANT</div>
      </div>
      <main className="login-main">
        <div className="login-form">
          <span className="eyebrow">SECURE ACCESS</span>
          <h2>进入工作台</h2>
          <p>
            输入管理员配置的访问令牌。令牌保存在当前标签页，关闭标签页后清除。
          </p>
          <form
            onSubmit={async (event) => {
              event.preventDefault();
              setBusy(true);
              setError("");
              try {
                await onLogin(input.trim());
              } catch (exc) {
                setError(exc instanceof Error ? exc.message : "无法连接服务");
              } finally {
                setBusy(false);
              }
            }}
          >
            <label htmlFor="token">访问令牌</label>
            <input
              id="token"
              type="password"
              autoComplete="off"
              value={input}
              onChange={(event) => setInput(event.target.value)}
              placeholder="粘贴访问令牌"
              required
            />
            <button className="button primary login-button" disabled={busy}>
              {busy ? "验证中…" : "进入工作台 →"}
            </button>
            {error && (
              <p className="form-error" role="alert">
                {error}
              </p>
            )}
          </form>
          <div className="login-note">
            内网单租户 · 独立 worker · 持久化测量记录
          </div>
        </div>
      </main>
    </div>
  );
}
