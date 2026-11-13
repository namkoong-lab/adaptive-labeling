"""Tiny job launcher: runs a list of shell commands with at most `per_gpu` concurrent jobs on each
of the given GPUs (round-robin assignment), logging stdout/stderr per job.

    python -m adaptive_sampling_lib.launch jobs.txt --gpus 0 4 7 --per_gpu 3 --log_dir logs/

Each line of jobs.txt is a command; the token {GPU} is replaced by the assigned GPU index.
Jobs whose expected output file (optional, after ' ## ' on the line) already exists are skipped.

Optional, for shared GPUs:
  --gpus_file F     lines "<gpu> [<slots>]", re-read every 30 s, so GPUs can be added, removed or resized while the
                    launcher runs (a removed GPU gets no new jobs; its running jobs finish).  Overrides --gpus.
  --min_free_mb N   start a job on a GPU only when it has at least N MiB free (nvidia-smi), and wait 30 s after a start
                    on that GPU before the next, so the new job's memory shows up first.
  --lock_dir D      share one job file between several launchers (e.g. on different nodes with the same file system):
                    a job is started only after atomically creating D/<hash>.lock (O_EXCL) and only if its output does not
                    exist yet; a failed job's lock is removed so that a later launcher can retry it.
  --cap_by_running P  apply each GPU's slot count to ALL python processes on this machine whose command line contains P
                    and whose CUDA_VISIBLE_DEVICES is that GPU (e.g. jobs left running by an earlier launcher), not just
                    to this launcher's own jobs.
"""
import argparse
import hashlib
import os
import socket
import subprocess
import sys
import time


def read_gpus_file(path, default_slots):
    try:
        cap = {}
        with open(path) as f:
            for line in f:
                line = line.split("#")[0].strip()
                if not line:
                    continue
                parts = line.split()
                cap[int(parts[0])] = int(parts[1]) if len(parts) > 1 else default_slots
        return cap
    except (OSError, ValueError):
        return None  # keep the previous capacities if the file is missing or half-written


def free_mib():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
        return {int(a): int(b) for a, b in (ln.split(",") for ln in out.strip().splitlines())}
    except Exception:
        return None


def running_per_gpu(pattern):
    """python processes on this machine whose command line contains `pattern`, counted per CUDA_VISIBLE_DEVICES."""
    out = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            argv = open(f"/proc/{d}/cmdline", "rb").read().split(b"\0")
            if not argv or b"python" not in os.path.basename(argv[0]) or pattern.encode() not in b" ".join(argv):
                continue
            env = open(f"/proc/{d}/environ", "rb").read().split(b"\0")
        except OSError:
            continue
        for e in env:
            if e.startswith(b"CUDA_VISIBLE_DEVICES="):
                try:
                    g = int(e.split(b"=", 1)[1].split(b",")[0])
                    out[g] = out.get(g, 0) + 1
                except ValueError:
                    pass
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("jobs_file")
    p.add_argument("--gpus", type=int, nargs="+", default=[0])
    p.add_argument("--per_gpu", type=int, default=2)
    p.add_argument("--gpus_file", default=None)
    p.add_argument("--min_free_mb", type=int, default=0)
    p.add_argument("--lock_dir", default=None)
    p.add_argument("--cap_by_running", default=None)
    p.add_argument("--log_dir", default="launch_logs")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--dry", action="store_true")
    args = p.parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    jobs = []
    with open(args.jobs_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cmd, _, expect = line.partition(" ## ")
            if expect and os.path.exists(expect.strip()):
                continue
            jobs.append((cmd, expect.strip()))
    cap = {g: args.per_gpu for g in args.gpus}
    if args.gpus_file:
        cap = read_gpus_file(args.gpus_file, args.per_gpu) or cap
    print(f"{len(jobs)} jobs to run on gpus {cap} (gpu: slots)", flush=True)
    if args.dry:
        for j, _ in jobs:
            print(j)
        return
    slots = {g: [] for g in cap}  # gpu -> list of (Popen, name, lock path or None)
    if args.lock_dir:
        os.makedirs(args.lock_dir, exist_ok=True)
    i = 0
    t0 = time.time()
    n_done = 0
    t_cap = t_free = 0.0
    free = None
    next_ok = {}  # gpu -> earliest time of the next start (min_free_mb mode)
    while i < len(jobs) or any(slots[g] for g in slots):
        now = time.time()
        if args.gpus_file and now - t_cap > 30:
            new = read_gpus_file(args.gpus_file, args.per_gpu)
            if new is not None and new != cap:
                print(f"[{now - t0:7.0f}s] gpu slots now {new}", flush=True)
                cap = new
            for g in cap:
                slots.setdefault(g, [])
            t_cap = now
        if args.min_free_mb and now - t_free > 10:
            free = free_mib() or free
            t_free = now
        busy = running_per_gpu(args.cap_by_running) if args.cap_by_running else None
        for g in list(slots):
            # reap
            still = []
            for proc, name, lock in slots[g]:
                if proc.poll() is None:
                    still.append((proc, name, lock))
                else:
                    n_done += 1
                    status = "ok" if proc.returncode == 0 else f"FAIL({proc.returncode})"
                    if proc.returncode != 0 and lock and os.path.exists(lock):
                        os.remove(lock)  # let a later launcher retry the job
                    print(f"[{time.time() - t0:7.0f}s] done {n_done}/{len(jobs)} {status}: {name}", flush=True)
            slots[g] = still
            while (busy.get(g, 0) if busy is not None else len(slots[g])) < cap.get(g, 0) and i < len(jobs):
                if args.min_free_mb:
                    if time.time() < next_ok.get(g, 0.0) or free is None or free.get(g, 0) < args.min_free_mb:
                        break
                    next_ok[g] = time.time() + 30.0
                    free[g] = free.get(g, 0) - args.min_free_mb  # until the next query sees the new job
                lock = None
                if args.lock_dir:  # claim the next job that no launcher has taken and whose output does not exist
                    while i < len(jobs):
                        raw, expect = jobs[i]
                        if expect and os.path.exists(expect):
                            i += 1; continue
                        lock = os.path.join(args.lock_dir, hashlib.md5((expect or raw).encode()).hexdigest() + ".lock")
                        try:
                            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                            os.write(fd, f"{socket.gethostname()} {os.getpid()} {raw}\n".encode()); os.close(fd)
                            break
                        except FileExistsError:
                            lock = None; i += 1
                    if i >= len(jobs):
                        break
                cmd = jobs[i][0].replace("{GPU}", str(g)).replace("{PY}", args.python)
                name = f"job{i:04d}"
                log = open(os.path.join(args.log_dir, name + ".log"), "w")
                log.write(cmd + "\n\n")
                log.flush()
                proc = subprocess.Popen(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT,
                                        env={**os.environ, "CUDA_VISIBLE_DEVICES": str(g),
                                             "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
                slots[g].append((proc, name, lock))
                if busy is not None:
                    busy[g] = busy.get(g, 0) + 1
                print(f"[{time.time() - t0:7.0f}s] start {name} on gpu{g}: {cmd}", flush=True)
                i += 1
        time.sleep(2)
    print("all done", flush=True)


if __name__ == "__main__":
    main()
