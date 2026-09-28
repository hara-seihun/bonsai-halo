#!/usr/bin/env python3
"""Drive the HTTP server with concurrent chat completions and report what it delivers.

Every other measurement in this repository calls the engine directly, which is why the served
concurrency bug survived: a panel sets its own route and drives one sequence. This one speaks to the
product over HTTP, the way the agents on this machine do.

One run starts each arm's server in turn, so both arms share a clock and a lock hold:

    tools/serve_load.py --exec ./bonsai-halo --dflash FILE --context 8192 \
        --arm serial:1 --arm batch:8 --concurrency 1,2,4,8 --max-tokens 48 \
        --prompts bench/serve-prompts.txt --out run.json

An arm is NAME:SLOTS[:VAR=VALUE[,VAR=VALUE...]], where SLOTS is `serve --slots` and the optional
third field sets environment variables for that arm's server alone. That is how one lock hold
prices two routes against one clock, because the route a decode step takes is a setting of the
serving process and not of the request:

    --arm sliced:16:HALO_SERVE_ROUTE_MIN=999 --arm wide:16

Requests are greedy, so the completion text of a prompt must not depend on how many requests shared
the step that produced it, nor on which route carried it; `--check` compares every arm against the
first and reports the first divergence.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def post_chat(base, prompt, max_tokens, timeout):
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(base + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read().decode())
    t1 = time.monotonic()
    usage = out.get("usage", {})
    halo = usage.get("halo", {})
    choice = out["choices"][0]
    return {
        "seconds": t1 - t0,
        "cached_tokens": usage.get("prompt_tokens_details", {}).get("cached_tokens", 0),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "prefill_seconds": halo.get("prefill_seconds", 0.0),
        "accepted": halo.get("accepted", 0),
        "drafted": halo.get("drafted", 0),
        "finish_reason": choice.get("finish_reason"),
        "content": choice["message"].get("content") or "",
        "reasoning": choice["message"].get("reasoning_content") or "",
    }


def stream_chat(base, prompt, max_tokens, timeout):
    """The streaming path agents use: returns (text, seconds to first token, total seconds)."""
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(base + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    text, first, t0 = "", None, time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            delta = chunk["choices"][0].get("delta", {}) if chunk.get("choices") else {}
            piece = (delta.get("reasoning_content") or "") + (delta.get("content") or "")
            if piece and first is None:
                first = time.monotonic() - t0
            text += piece
    return text, first, time.monotonic() - t0


def wait_ready(base, proc, seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("server exited with %d before becoming ready" % proc.returncode)
        try:
            with urllib.request.urlopen(base + "/models", timeout=2) as r:
                r.read()
            return
        except (urllib.error.URLError, ConnectionError, OSError):
            time.sleep(0.25)
    raise RuntimeError("server did not become ready in %.0f s" % seconds)


def run_point(base, prompts, concurrency, max_tokens, timeout):
    """One wave of `concurrency` requests, fired together, waited for together."""
    picked = [prompts[i % len(prompts)] for i in range(concurrency)]
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(lambda p: post_chat(base, p, max_tokens, timeout), picked))
    wall = time.monotonic() - t0
    tokens = sum(r["completion_tokens"] for r in results)
    return {
        "concurrency": concurrency,
        "wall_seconds": wall,
        "completion_tokens": tokens,
        "aggregate_tps": tokens / wall if wall > 0 else 0.0,
        "per_request_tps": [r["completion_tokens"] / r["seconds"] if r["seconds"] > 0 else 0.0 for r in results],
        "latency_seconds": [r["seconds"] for r in results],
        "prefill_seconds": [r["prefill_seconds"] for r in results],
        "prompt_tokens": [r["prompt_tokens"] for r in results],
        "cached_tokens": [r["cached_tokens"] for r in results],
        "drafted": [r["drafted"] for r in results],
        "accepted": [r["accepted"] for r in results],
        "finish_reasons": [r["finish_reason"] for r in results],
        "texts": [r["reasoning"] + r["content"] for r in results],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exec", dest="binary", default="./bonsai-halo")
    ap.add_argument("--model", default=None)
    ap.add_argument("--dflash", default=None)
    ap.add_argument("--context", type=int, default=0)
    ap.add_argument("--arm", action="append", default=[], help="NAME:SLOTS[:VAR=VALUE,...]")
    ap.add_argument("--base-url", default=None,
                    help="drive a server that is already running (the resident service) instead of starting arms")
    ap.add_argument("--concurrency", default="1,2,4,8")
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--prompts", default="bench/serve-prompts.txt")
    ap.add_argument("--port", type=int, default=8571)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--ready-seconds", type=float, default=90.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--check", action="store_true", help="compare each arm's completions against the first arm")
    ap.add_argument("--repeat-control", action="store_true",
                    help="repeat the first arm's waves to measure the served greedy identity floor")
    ap.add_argument("--stream-check", action="store_true",
                    help="also run one streaming request per arm and compare it with the buffered one")
    ap.add_argument("--extra", action="append", default=[], help="extra server argument")
    ap.add_argument("--shared-prefix", type=int, default=0,
                    help="prepend this many tokens of identical preamble to every prompt, which is what an "
                         "agent workload looks like and what the prompt-prefix snapshots exist for")
    args = ap.parse_args()

    prompts = [l.strip() for l in open(args.prompts) if l.strip()]
    if args.shared_prefix:
        # deterministic filler: one token per word is close enough for this model's tokenizer
        words = ["section %d discusses the measured behaviour of the system under load." % i
                 for i in range(args.shared_prefix // 10 + 1)]
        preamble = "Read the following notes, then answer the question at the end.\n" + " ".join(words) + "\n\nQuestion: "
        prompts = [preamble + p for p in prompts]
    points = [int(c) for c in args.concurrency.split(",") if c]
    run = {"arms": [], "max_tokens": args.max_tokens, "concurrency": points,
           "prompts": os.path.abspath(args.prompts), "binary": os.path.abspath(args.binary),
           "context": args.context, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with open(args.binary, "rb") as binary:
        digest = hashlib.sha256()
        for chunk in iter(lambda: binary.read(1024 * 1024), b""):
            digest.update(chunk)
        run["binary_sha256"] = digest.hexdigest()

    if args.base_url:
        base = args.base_url.rstrip("/")
        arm = {"name": "resident", "slots": None, "base_url": base, "points": []}
        post_chat(base, prompts[0], 4, args.timeout)
        for c in points:
            point = run_point(base, prompts, c, args.max_tokens, args.timeout)
            arm["points"].append(point)
            print("resident c=%-3d %7.2f s  %6.1f tok/s aggregate  %5.1f tok/s per request  %d/%d drafted"
                  % (c, point["wall_seconds"], point["aggregate_tps"],
                     sum(point["per_request_tps"]) / len(point["per_request_tps"]),
                     sum(point["accepted"]), sum(point["drafted"])), flush=True)
        run["arms"].append(arm)

    for spec in args.arm:
        name, _, rest = spec.partition(":")
        slots, _, envspec = rest.partition(":")
        slots = int(slots or 1)
        env = dict(os.environ)
        armenv = {}
        for kv in envspec.split(",") if envspec else []:
            if not kv:
                continue
            k, _, v = kv.partition("=")
            armenv[k] = v
            env[k] = v
        port = args.port + len(run["arms"])
        base = "http://127.0.0.1:%d/v1" % port
        cmd = [args.binary, "serve", "--port", str(port), "--slots", str(slots)]
        if args.model:
            cmd += ["-m", args.model]
        if args.dflash:
            cmd += ["--dflash", args.dflash]
        if args.context:
            cmd += ["--context", str(args.context)]
        cmd += args.extra
        log_path = os.path.join(os.path.dirname(os.path.abspath(args.out)), "serve-%s.log" % name) if args.out else "/tmp/serve_load-%s.log" % name
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        log = open(log_path, "w")
        t0 = time.monotonic()
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        arm = {"name": name, "slots": slots, "port": port, "command": cmd, "env": armenv, "server_log": log_path, "points": []}
        try:
            wait_ready(base, proc, args.ready_seconds)
            arm["startup_seconds"] = time.monotonic() - t0
            # one short request so the first timed point does not pay lazy initialisation
            post_chat(base, prompts[0], 4, args.timeout)
            if args.stream_check:
                text, ttft, total = stream_chat(base, prompts[1], args.max_tokens, args.timeout)
                buffered = post_chat(base, prompts[1], args.max_tokens, args.timeout)
                arm["stream"] = {"first_token_seconds": ttft, "seconds": total,
                                 "matches_buffered": text == (buffered["reasoning"] + buffered["content"])}
                print("%-8s stream: first token %.3f s, %.2f s total, matches buffered %s"
                      % (name, ttft or -1, total, arm["stream"]["matches_buffered"]), flush=True)
            for c in points:
                point = run_point(base, prompts, c, args.max_tokens, args.timeout)
                arm["points"].append(point)
                if args.repeat_control and len(run["arms"]) == 0:
                    repeat = run_point(base, prompts, c, args.max_tokens, args.timeout)
                    arm.setdefault("control_repeats", []).append({
                        "concurrency": c,
                        "identical_completions": sum(a == b for a, b in zip(point["texts"], repeat["texts"])),
                        "total_completions": c,
                        "repeat": repeat,
                    })
                print("%-8s c=%-3d %7.2f s  %6.1f tok/s aggregate  %5.1f tok/s per request  %d/%d drafted  %d/%d prompt tokens cached"
                      % (name, c, point["wall_seconds"], point["aggregate_tps"],
                         sum(point["per_request_tps"]) / len(point["per_request_tps"]),
                         sum(point["accepted"]), sum(point["drafted"]),
                         sum(point["cached_tokens"]), sum(point["prompt_tokens"])), flush=True)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
            log.close()
        run["arms"].append(arm)

    if args.check and len(run["arms"]) > 1:
        ref = run["arms"][0]
        for arm in run["arms"][1:]:
            same, diff = 0, []
            for rp, ap_ in zip(ref["points"], arm["points"]):
                for i, (a, b) in enumerate(zip(rp["texts"], ap_["texts"])):
                    if a == b:
                        same += 1
                    else:
                        cut = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))
                        diff.append({"concurrency": rp["concurrency"], "request": i, "at_char": cut,
                                     "ref": a[max(0, cut - 40):cut + 40], "arm": b[max(0, cut - 40):cut + 40]})
            arm["identical_completions"] = same
            arm["divergences"] = diff
            print("%s against %s: %d identical completions, %d divergent"
                  % (arm["name"], ref["name"], same, len(diff)), flush=True)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(run, f, indent=1)
        print("wrote " + args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
