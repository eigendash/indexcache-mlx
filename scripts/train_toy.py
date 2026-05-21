"""Train a small IndexCache model on multi-key associative recall.

Configurations, all sharing one seeded initialisation so the comparison is
apples to apples:

  all_Full         every layer keeps its indexer (the DSA baseline)
  shared_1_in_2    uniform interleaved pattern, weights from all-Full training
  shared_1_in_4    uniform interleaved pattern, weights from all-Full training
  greedy_selected  the training-free greedy pattern (Algorithm 1), found by
                   minimising LM loss on a fixed calibration set using the
                   all-Full weights
  trained_1_in_2   the same 1-in-2 pattern, trained from scratch with the
                   multi-layer distillation term of Section 3.2
  trained_1_in_4   the same 1-in-4 pattern, trained from scratch with it

Writes ``results/experiment.json`` and ``results/run_log.txt``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indexcache import (  # noqa: E402
    VOCAB_SIZE,
    IndexCacheModel,
    LayerPattern,
    ModelConfig,
    TaskConfig,
    answer_accuracy,
    answer_loss,
    build_task,
    filter_queries,
    greedy_layer_selection,
    overlap_ratio,
    pairwise_jaccard,
)
from indexcache.greedy import calibration_batches  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")


def build_corpus(n: int, seq_len: int, seed: int, n_pairs: int = 6) -> np.ndarray:
    """(N, L) token ids: ``n`` examples, generated with consecutive seeds."""
    per_call = n_pairs
    chunks = [
        build_task(TaskConfig(n_pairs=n_pairs, seq_len=seq_len, seed=seed + i))
        for i in range((n + per_call - 1) // per_call)
    ]
    return np.concatenate(chunks, axis=0)[:n]


def batches(data: np.ndarray, batch_size: int, *, shuffle: bool, seed: int):
    rng = np.random.default_rng(seed)
    order = np.arange(data.shape[0])
    if shuffle:
        rng.shuffle(order)
    for start in range(0, len(order) - batch_size + 1, batch_size):
        yield mx.array(data[order[start : start + batch_size]])


def evaluate(model, data, *, batch_size=16, with_indices=False, pattern=None):
    """Held-out task loss and accuracy at the answer position."""
    losses, correct = [], 0
    indices: dict[int, list] = {}
    for start in range(0, data.shape[0], batch_size):
        tok = mx.array(data[start : start + batch_size])
        out = model.forward(tok, pattern, collect_indices=with_indices)
        logits = out["logits"]
        losses.append(float(answer_loss(logits, tok).item()))
        correct += round(answer_accuracy(logits, tok) * tok.shape[0])
        if with_indices:
            for layer, idx in out["indices"].items():
                mx.eval(idx)
                indices.setdefault(layer, []).append(np.asarray(idx))
    result = {"loss": float(np.mean(losses)), "accuracy": correct / data.shape[0]}
    if with_indices:
        result["indices"] = {
            layer: mx.array(np.concatenate(chunks, axis=0)) for layer, chunks in indices.items()
        }
    return result


def train(
    model,
    train_data,
    *,
    pattern,
    steps,
    batch_size,
    lr,
    seed,
    distill_weight=0.0,
    full_sequence=False,
    log_every=100,
    say=print,
    label="",
):
    """Train ``model`` under ``pattern``; returns the per-step loss history."""
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=0.0)
    state = [model.state, optimizer.state]
    history = []
    t0 = time.time()
    data_iter = batches(train_data, batch_size, shuffle=True, seed=seed)
    for step in range(steps):
        try:
            tok = next(data_iter)
        except StopIteration:
            data_iter = batches(train_data, batch_size, shuffle=True, seed=seed + step + 1)
            tok = next(data_iter)

        def loss_fn(mdl):
            return mdl.training_loss(
                tok,
                pattern,
                distill_weight=distill_weight,
                mode="per_layer",
                full_sequence=full_sequence,
            )

        loss, grads = mx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        mx.eval(loss, model.state, optimizer.state)
        history.append(float(loss.item()))
        if log_every and (step + 1) % log_every == 0:
            say(
                f"    {label:14s} step {step + 1:4d}/{steps}  loss {history[-1]:.4f}"
                f"  ({time.time() - t0:.1f}s)"
            )
    return history


def timed_forward(model, tokens, pattern, *, repeats=5, warmup=2):
    """Mean wall-clock seconds for one forward pass over ``tokens``."""
    for _ in range(warmup):
        mx.eval(model.forward(tokens, pattern)["logits"])
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        mx.eval(model.forward(tokens, pattern)["logits"])
        times.append(time.perf_counter() - t0)
    return float(np.mean(times))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--n-pairs", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--top-k", type=int, default=32)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--n-train", type=int, default=800)
    ap.add_argument("--n-val", type=int, default=200)
    ap.add_argument("--n-calib", type=int, default=72)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--distill-weight", type=float, default=0.001)
    ap.add_argument("--overlap-min-candidates", type=int, default=16)
    ap.add_argument("--out", type=str, default=RESULTS_DIR)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    log_file = open(os.path.join(args.out, "run_log.txt"), "w")

    def say(msg=""):
        print(msg, flush=True)
        log_file.write(str(msg) + "\n")
        log_file.flush()

    cfg = ModelConfig(
        vocab_size=VOCAB_SIZE,
        d_model=64,
        n_heads=4,
        n_layers=args.n_layers,
        top_k=args.top_k,
        max_seq_len=args.seq_len,
        window=-1,
    )
    seeds = [args.seed + i for i in range(args.seeds)]
    say("IndexCache toy experiment -- cross-layer top-k index reuse")
    say(f"  model   {asdict(cfg)}")
    say(f"  task    multi-key associative recall, {args.n_pairs} key/value pairs"
        f" drawn from {128} keys, answer at the final position")
    say(f"  vocab   {VOCAB_SIZE} tokens")
    say(f"  data    train {args.n_train}  val {args.n_val}  calib {args.n_calib},"
        f" sequence {args.seq_len}")
    say(f"  training {args.steps} steps, batch {args.batch_size}, lr {args.lr}")
    say(f"  seeds   {seeds}")
    say(f"  distillation weight {args.distill_weight}")
    say()

    per_seed = []
    for seed in seeds:
        say(f"================ seed {seed} ================")
        train_data = build_corpus(args.n_train, args.seq_len, 1000 + seed, args.n_pairs)
        val_data = build_corpus(args.n_val, args.seq_len, 9000 + seed, args.n_pairs)
        calib_data = build_corpus(args.n_calib, args.seq_len, 5000 + seed, args.n_pairs)

        def fresh_model(seed=seed):
            mx.random.seed(seed)
            return IndexCacheModel(cfg)

        row = {"seed": seed}

        # ------------------------------------------------------------ all-Full
        say("[1/5] all-Full: every layer keeps its own indexer (DSA baseline)")
        all_full = fresh_model()
        t0 = time.time()
        train(
            all_full,
            train_data,
            pattern=LayerPattern.all_full(args.n_layers),
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=seed,
            log_every=max(1, args.steps // 4),
            say=say,
            label="all-Full",
        )
        row["all_Full_seconds"] = time.time() - t0
        ev = evaluate(all_full, val_data, with_indices=True)
        full_indices = ev.pop("indices")
        row["all_Full"] = {**ev, "pattern": "F" * args.n_layers, "indexer_saving": 0.0}
        say(
            f"  all-Full held-out loss {ev['loss']:.4f}  answer accuracy {ev['accuracy']:.3f}"
            f"  ({row['all_Full_seconds']:.1f}s)"
        )
        jac = pairwise_jaccard(full_indices)
        ovl = overlap_ratio(full_indices)
        deep = filter_queries(full_indices, args.overlap_min_candidates)
        jac_deep = pairwise_jaccard(deep)
        ovl_deep = overlap_ratio(deep)
        row["overlap"] = {
            "k": args.top_k,
            "jaccard_mean_adjacent": jac["mean_adjacent"],
            "jaccard_pairs": {key: val for key, val in jac.items() if key != "mean_adjacent"},
            "overlap_ratio_mean_adjacent": ovl["mean_adjacent"],
            "min_candidates": args.overlap_min_candidates,
            "jaccard_mean_adjacent_filtered": jac_deep["mean_adjacent"],
            "overlap_ratio_mean_adjacent_filtered": ovl_deep["mean_adjacent"],
        }
        say(
            f"  adjacent-layer top-{args.top_k} overlap: all queries Jaccard"
            f" {jac['mean_adjacent']:.3f}, overlap/k {ovl['mean_adjacent']:.3f}"
        )
        say(
            f"  adjacent-layer overlap, queries with >="
            f" {args.overlap_min_candidates} candidates: Jaccard"
            f" {jac_deep['mean_adjacent']:.3f}, overlap/k {ovl_deep['mean_adjacent']:.3f}"
        )
        for pair, value in jac.items():
            if pair != "mean_adjacent":
                say(f"    jaccard {pair}: {value:.3f}")
        say()

        # ------------------------------------- training-free (shared weights)
        say("[2/5] training-free: reuse the all-Full weights, share indices")
        for name, every in (("shared_1_in_2", 2), ("shared_1_in_4", 4)):
            pattern = LayerPattern.interleaved(args.n_layers, every)
            free = evaluate(all_full, val_data, pattern=pattern)
            free["pattern"] = pattern.pattern
            free["indexer_saving"] = pattern.indexer_saving()
            row[name] = free
            say(
                f"  {name:14s} {pattern.pattern}  loss {free['loss']:.4f}"
                f"  accuracy {free['accuracy']:.3f}  indexers -{pattern.indexer_saving():.0%}"
            )
        say()

        # -------------------------------------------------------- greedy search
        say("[3/5] training-free greedy search (Algorithm 1) on the calibration set")
        calib_batches = calibration_batches(calib_data, args.batch_size, 6)
        calls = []

        def eval_for_greedy(pattern):
            calls.append(pattern.pattern)
            total = 0.0
            for tok in calib_batches:
                total += float(
                    answer_loss(all_full.forward(tok, pattern)["logits"], tok).item()
                )
            return total / len(calib_batches)

        greedy_pattern, trace = greedy_layer_selection(
            eval_for_greedy, args.n_layers, args.n_layers - 1, log=say
        )
        greedy = evaluate(all_full, val_data, pattern=greedy_pattern)
        greedy["pattern"] = greedy_pattern.pattern
        greedy["indexer_saving"] = greedy_pattern.indexer_saving()
        greedy["trace"] = trace
        greedy["calibration_forward_passes"] = len(calls)
        row["greedy_selected"] = greedy
        say(
            f"  greedy pattern {greedy_pattern.pattern}  loss {greedy['loss']:.4f}"
            f"  accuracy {greedy['accuracy']:.3f}  ({len(calls)} calibration forwards)"
        )
        say()

        # --------------------------------------------------------- train-aware
        say("[4/5] training-aware: train retained indexers against the layers they serve")
        for name, every in (("trained_1_in_2", 2), ("trained_1_in_4", 4)):
            pattern = LayerPattern.interleaved(args.n_layers, every)
            model = fresh_model()
            t0 = time.time()
            train(
                model,
                train_data,
                pattern=pattern,
                steps=args.steps,
                batch_size=args.batch_size,
                lr=args.lr,
                seed=seed,
                distill_weight=args.distill_weight,
                log_every=max(1, args.steps // 4),
                say=say,
                label=name,
            )
            elapsed = time.time() - t0
            trained = evaluate(model, val_data, pattern=pattern)
            trained["pattern"] = pattern.pattern
            trained["indexer_saving"] = pattern.indexer_saving()
            trained["train_seconds"] = elapsed
            row[name] = trained
            row[f"{name}_model_pattern"] = pattern.pattern
            say(
                f"  {name:14s} {pattern.pattern}  loss {trained['loss']:.4f}"
                f"  accuracy {trained['accuracy']:.3f}  ({elapsed:.1f}s)"
            )
        say()

        per_seed.append(row)
        say()

    # The timed forward pass is model- and pattern-dependent, so it is measured
    # once at the end with freshly built models (same seed) rather than inside
    # the aggregation loop.
    say("[5/5] cost: measured forward time, 16 x %d tokens" % args.seq_len)
    sample = mx.array(build_corpus(16, args.seq_len, 9000 + args.seed, args.n_pairs))
    cost = {}
    timing_models = {}
    mx.random.seed(args.seed)
    timing_models["all_Full"] = IndexCacheModel(cfg)
    for name in ("shared_1_in_2", "shared_1_in_4", "greedy_selected"):
        timing_models[name] = timing_models["all_Full"]
    for name in ("trained_1_in_2", "trained_1_in_4"):
        pattern = LayerPattern(per_seed[0][name]["pattern"])
        model = IndexCacheModel(cfg)
        train(
            model,
            build_corpus(args.n_train, args.seq_len, 1000 + args.seed, args.n_pairs),
            pattern=pattern,
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            distill_weight=args.distill_weight,
            log_every=0,
            say=lambda *_a, **_k: None,
            label=name,
        )
        timing_models[name] = model

    for name, row in per_seed[0].items():
        if name not in ("seed", "overlap") and isinstance(row, dict) and "pattern" in row:
            pattern = LayerPattern(row["pattern"])
            prefill = timed_forward(timing_models[name], sample, pattern)
            row["prefill_seconds"] = prefill
            cost[name] = {
                "pattern": pattern.pattern,
                "indexer_invocations": pattern.n_full,
                "indexer_saving": pattern.indexer_saving(),
                "prefill_seconds_16x%d" % args.seq_len: prefill,
                "prefill_ms": prefill * 1000.0,
            }
            say(
                f"  {name:14s} {pattern.pattern}  indexers {pattern.n_full}/{args.n_layers}"
                f"  prefill {prefill * 1000:8.2f} ms"
            )

    # ---------------------------------------------------------------- aggregate
    names = [name for name in per_seed[0] if isinstance(per_seed[0][name], dict) and "pattern" in per_seed[0][name]]
    summary = {}
    for name in names:
        losses = [row[name]["loss"] for row in per_seed]
        accs = [row[name]["accuracy"] for row in per_seed]
        summary[name] = {
            "pattern": per_seed[0][name]["pattern"],
            "loss_mean": float(np.mean(losses)),
            "loss_std": float(np.std(losses)),
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs)),
            "indexer_saving": per_seed[0][name]["indexer_saving"],
        }
    jac_adj = [row["overlap"]["jaccard_mean_adjacent"] for row in per_seed]
    ovl_adj = [row["overlap"]["overlap_ratio_mean_adjacent"] for row in per_seed]
    jac_filt = [row["overlap"]["jaccard_mean_adjacent_filtered"] for row in per_seed]
    ovl_filt = [row["overlap"]["overlap_ratio_mean_adjacent_filtered"] for row in per_seed]

    results = {
        "config": asdict(cfg),
        "args": vars(args),
        "task": (
            f"multi-key associative recall, {args.n_pairs} pairs,"
            " answer read at the final position"
        ),
        "seeds": seeds,
        "per_seed": per_seed,
        "summary": summary,
        "cost": cost,
        "cross_layer_overlap": {
            "k": args.top_k,
            "jaccard_mean_adjacent": float(np.mean(jac_adj)),
            "jaccard_mean_adjacent_std": float(np.std(jac_adj)),
            "overlap_ratio_mean_adjacent": float(np.mean(ovl_adj)),
            "overlap_ratio_mean_adjacent_std": float(np.std(ovl_adj)),
            "min_candidates_filter": args.overlap_min_candidates,
            "jaccard_mean_adjacent_filtered": float(np.mean(jac_filt)),
            "jaccard_mean_adjacent_filtered_std": float(np.std(jac_filt)),
            "overlap_ratio_mean_adjacent_filtered": float(np.mean(ovl_filt)),
            "overlap_ratio_mean_adjacent_filtered_std": float(np.std(ovl_filt)),
            "per_seed": [
                {"seed": row["seed"], **row["overlap"]} for row in per_seed
            ],
        },
    }

    with open(os.path.join(args.out, "experiment.json"), "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)

    say()
    say("summary over %d seeds" % len(seeds))
    say(
        f"  {'configuration':16s} {'pattern':9s} {'loss':>14s} {'accuracy':>14s} {'indexers':>9s}"
    )
    for name, value in summary.items():
        say(
            f"  {name:16s} {value['pattern']:9s}"
            f" {value['loss_mean']:7.4f}+-{value['loss_std']:.4f}"
            f" {value['accuracy_mean']:7.3f}+-{value['accuracy_std']:.3f}"
            f" {value['indexer_saving'] * 100:8.0f}%"
        )
    say()
    say(
        f"cross-layer top-{args.top_k} overlap between adjacent layers (all queries):"
        f" Jaccard {np.mean(jac_adj):.3f} +- {np.std(jac_adj):.3f},"
        f" overlap/k {np.mean(ovl_adj):.3f}"
    )
    say(
        f"cross-layer overlap on queries with >= {args.overlap_min_candidates}"
        f" candidates: Jaccard {np.mean(jac_filt):.3f} +- {np.std(jac_filt):.3f},"
        f" overlap/k {np.mean(ovl_filt):.3f}"
    )
    say(f"wrote {os.path.join(args.out, 'experiment.json')}")
    log_file.close()


if __name__ == "__main__":
    main()
