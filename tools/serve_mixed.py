#!/usr/bin/env python3
"""Drive the HTTP server with a workload that has prompts and generation in it at the same time.

`tools/serve_load.py` fires N requests together and waits for all of them, which is the right
instrument for a route: every request ingests, then every request generates, and the steps in
between all carry N rows. It cannot see the scheduler, because in that pattern there is never a
request generating while another one's prompt is being read.

That is the pattern this one breaks. Requests arrive on a schedule, each with its own long prompt,
so at any moment some are ingesting and some are decoding - the shape a served agent workload
actually has. Two things come out of it that a synchronised burst cannot show:

  * **aggregate tokens per second over the whole window**, which moves when a request is held out
    of the batch, because a served step is priced per weight stream and a narrow step wastes it;
  * **the inter-token stall**, the longest gap between two tokens of one response, which is what a
    prompt admitted as a single call does to everybody else.

Arms are the same `NAME:SLOTS[:VAR=VALUE,...]` as `serve_load.py`, so one lock hold prices two
schedulers against one clock:

    tools/serve_mixed.py --exec ./bonsai-halo --dflash FILE --context 4096 \
        --arm blocking:8:HALO_SERVE_PREFILL_CHUNK=-1 --arm chunked:8 \
        --requests 8 --stagger 1.5 --prompt-tokens 1200 --max-tokens 64 --check --out run.json

Requests are greedy, so an arm that changes only *when* work runs must return the same text;
`--check` compares every arm against the first and reports the first divergent character.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from serve_load import post_chat, wait_ready  # noqa: E402  (the HTTP shapes are theirs)


WORDS = ("measurement latency throughput kernel operand register bandwidth schedule residual "
         "projection attention recurrence quantiser occupancy barrier workgroup accumulator "
         "dispatch pipeline coalesced ternary activation gradient inference weights").split()


def make_prompt(index, approx_tokens, shared_prefix=0):
    """A distinct prompt of about `approx_tokens` tokens, plus an optional shared preamble.

    Distinct is the point: a workload whose prompts share a prefix is served from the snapshot
    cache and does almost no ingestion, which is the case where a blocking admission costs nothing.
    """
    out = []
    if shared_prefix:
        out.append("You are reading an engineering log. Notes follow.")
        for i in range(shared_prefix // 8):
            out.append("Entry %d records the %s of the %s stage."
                       % (i, WORDS[i % len(WORDS)], WORDS[(i * 7 + 3) % len(WORDS)]))
    out.append("Request %d. Private notes follow." % index)
    i = 0
    while sum(len(s.split()) for s in out) < approx_tokens * 0.75:
        out.append("Observation %d-%d: the %s of run %d was %d units, against %d for %s."
                   % (index, i, WORDS[(i * 3 + index * 5) % len(WORDS)], i * 7 + index,
                      (i * 131 + index * 17) % 997, (i * 71 + index * 13) % 499,
                      WORDS[(i * 11 + index) % len(WORDS)]))
        i += 1
    out.append("In one short paragraph, say what the notes above are about.")
    return "\n".join(out)


def stream_tokens(base, prompt, max_tokens, timeout):
    """Stream one completion and keep the arrival time of every chunk."""
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(base + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    text, stamps, usage = "", [], {}
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            if chunk.get("usage"):
                usage = chunk["usage"]           # the server's own exact token count, last chunk
            choices = chunk.get("choices") or [{}]
            delta = choices[0].get("delta", {})
            piece = (delta.get("reasoning_content") or "") + (delta.get("content") or "")
            if piece:
                text += piece
                stamps.append(time.monotonic())
    end = time.monotonic()
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    halo = usage.get("halo", {})
    return {
        "text": text,
        "submitted": t0,
        "finished": end,
        "seconds": end - t0,
        "first_token_seconds": (stamps[0] - t0) if stamps else None,
        "completion_tokens": usage.get("completion_tokens", len(stamps)),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "cached_tokens": usage.get("prompt_tokens_details", {}).get("cached_tokens", 0),
        "prefill_seconds": halo.get("prefill_seconds", 0.0),
        "accepted": halo.get("accepted", 0),
        "drafted": halo.get("drafted", 0),
        "chunks": len(stamps),
        "max_gap_seconds": max(gaps) if gaps else 0.0,
        "p50_gap_seconds": sorted(gaps)[len(gaps) // 2] if gaps else 0.0,
    }


def run_mixed(base, prompts, stagger, max_tokens, timeout):
    """Post every request on a schedule and wait for all of them.

    The window is measured from the first submission to the last completion, so a scheduler that
    finishes the same work sooner reports a higher aggregate rate even though no request asked for
    fewer tokens.
    """
    results = [None] * len(prompts)
    errors = []

    def one(i):
        try:
            results[i] = stream_tokens(base, prompts[i], max_tokens, timeout)
        except Exception as ex:  # a failed request must not look like a fast one
            errors.append("request %d: %s" % (i, ex))

    threads = []
    t0 = time.monotonic()
    for i in range(len(prompts)):
        due = t0 + i * stagger
        now = time.monotonic()
        if due > now:
            time.sleep(due - now)
        th = threading.Thread(target=one, args=(i,))
        th.start()
        threads.append(th)
    for th in threads:
        th.join()
    if errors:
        raise RuntimeError("; ".join(errors))
    wall = max(r["finished"] for r in results) - t0
    tokens = sum(r["completion_tokens"] for r in results)
    return {
        "requests": len(prompts),
        "stagger_seconds": stagger,
        "wall_seconds": wall,
        "completion_tokens": tokens,
        "aggregate_tps": tokens / wall if wall > 0 else 0.0,
        "prompt_tokens": [r["prompt_tokens"] for r in results],
        "cached_tokens": [r["cached_tokens"] for r in results],
        "prefill_seconds": [r["prefill_seconds"] for r in results],
        "accepted": [r["accepted"] for r in results],
        "drafted": [r["drafted"] for r in results],
        "first_token_seconds": [r["first_token_seconds"] for r in results],
        "seconds": [r["seconds"] for r in results],
        "max_gap_seconds": [r["max_gap_seconds"] for r in results],
        "p50_gap_seconds": [r["p50_gap_seconds"] for r in results],
        "texts": [r["text"] for r in results],
    }


def summarise(name, point):
    gaps = sorted(point["max_gap_seconds"])
    ttft = [t for t in point["first_token_seconds"] if t is not None]
    print("%-10s %2d req stagger %.1fs  %7.2f s window  %6.1f tok/s aggregate  "
          "TTFT med %.2f max %.2f  stall med %.2f max %.2f"
          % (name, point["requests"], point["stagger_seconds"], point["wall_seconds"],
             point["aggregate_tps"],
             sorted(ttft)[len(ttft) // 2] if ttft else -1, max(ttft) if ttft else -1,
             gaps[len(gaps) // 2], gaps[-1]), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exec", dest="binary", default="./bonsai-halo")
    ap.add_argument("--model", default=None)
    ap.add_argument("--dflash", default=None)
    ap.add_argument("--context", type=int, default=0)
    ap.add_argument("--arm", action="append", default=[], help="NAME:SLOTS[:VAR=VALUE,...]")
    ap.add_argument("--base-url", default=None, help="measure an already running server instead")
    ap.add_argument("--requests", type=int, default=8)
    ap.add_argument("--stagger", type=float, default=1.5, help="seconds between arrivals")
    ap.add_argument("--prompt-tokens", type=int, default=1200)
    ap.add_argument("--shared-prefix", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--port", type=int, default=8671)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--ready-seconds", type=float, default=120.0)
    ap.add_argument("--extra", action="append", default=[], help="extra server argument")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    prompts = [make_prompt(i, args.prompt_tokens, args.shared_prefix) for i in range(args.requests)]
    run = {"requests": args.requests, "stagger": args.stagger, "prompt_tokens": args.prompt_tokens,
           "shared_prefix": args.shared_prefix, "max_tokens": args.max_tokens,
           "rounds": args.rounds, "binary": os.path.abspath(args.binary),
           "context": args.context, "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "arms": []}
    try:
        run["binary_sha256"] = subprocess.run(["sha256sum", args.binary], capture_output=True,
                                              text=True, check=True).stdout.split()[0]
    except Exception:
        pass

    def measure(name, base, arm):
        for _ in range(args.rounds):
            point = run_mixed(base, prompts, args.stagger, args.max_tokens, args.timeout)
            arm["points"].append(point)
            summarise(name, point)

    if args.base_url:
        base = args.base_url.rstrip("/")
        arm = {"name": "resident", "slots": None, "base_url": base, "points": []}
        post_chat(base, prompts[0][:200], 4, args.timeout)
        measure("resident", base, arm)
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
        log = open("/tmp/serve_mixed-%s.log" % name, "w")
        t0 = time.monotonic()
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        arm = {"name": name, "slots": slots, "port": port, "command": cmd, "env": armenv, "points": []}
        try:
            wait_ready(base, proc, args.ready_seconds)
            arm["startup_seconds"] = time.monotonic() - t0
            post_chat(base, prompts[0][:200], 4, args.timeout)   # pay lazy initialisation first
            measure(name, base, arm)
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
                        cut = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]),
                                   min(len(a), len(b)))
                        diff.append({"request": i, "at_char": cut,
                                     "ref": a[max(0, cut - 40):cut + 40],
                                     "arm": b[max(0, cut - 40):cut + 40]})
            arm["identical_completions"] = same
            arm["divergences"] = diff
            print("%s against %s: %d identical completions, %d divergent"
                  % (arm["name"], ref["name"], same, len(diff)), flush=True)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(run, f, indent=1)
        print("wrote %s" % args.out, flush=True)


if __name__ == "__main__":
    main()
