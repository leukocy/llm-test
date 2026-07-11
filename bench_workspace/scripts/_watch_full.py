"""监视 rounds=3 完整矩阵:每 120s 查进度,完成(all done)/异常(进程消失+行数停)/超时 即退出。
退出后触发通知 → 自动重生报告 + 提交。"""

import glob
import os
import subprocess
import time

LOG = "raw_data/full_run.log"
CSV_GLOB = "raw_data/full_*.csv"
TIMEOUT = 4 * 3600  # 4h 上限
PHASE_DONE_MARK = "all done"


def csv_rows():
    n = 0
    for f in glob.glob(CSV_GLOB):
        if os.path.exists(f):
            try:
                with open(f) as fh:
                    n += sum(1 for _ in fh) - 1
            except Exception:
                pass
    return n


def process_alive():
    try:
        r = subprocess.run(["pgrep", "-f", "_live_test_full"], capture_output=True, text=True)
        return bool(r.stdout.strip())
    except Exception:
        return True


start = time.time()
last_rows = 0
stable = 0
print("[watch] start, 监视 rounds=3 矩阵(完成/异常/4h超时即退出)", flush=True)
while time.time() - start < TIMEOUT:
    time.sleep(120)
    t = int(time.time() - start)
    log = open(LOG, errors="ignore").read() if os.path.exists(LOG) else ""  # noqa: SIM115
    rows = csv_rows()
    alive = process_alive()
    print(
        f"[watch] t={t//60}min rows={rows}/1482 proc={'alive' if alive else 'gone'}",
        flush=True,
    )

    if PHASE_DONE_MARK in log:
        print(f"[watch] DONE (all done) @ t={t//60}min rows={rows}", flush=True)
        break
    if not alive:
        if rows == last_rows:
            stable += 1
            if stable >= 2:  # 进程消失且行数 2 次不变 → 视为结束(可能崩溃/中止)
                print(
                    f"[watch] [WARN] PROCESS GONE + rows stable({rows}) → 结束(查 log 看是否正常 all done)",
                    flush=True,
                )
                break
        else:
            stable = 0
    last_rows = rows
else:
    print(f"[watch] TIMEOUT 4h rows={csv_rows()}", flush=True)
