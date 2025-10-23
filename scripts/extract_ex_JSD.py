#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_examples_JSD.py

Collect example sentences for JSD top-contributing lexical types, per model,
from DELPH-IN itsdb profiles.

Adds progress reporting:
- Per-model start/heartbeat every N processed items (--progress-every)
- Early-stop message when caps are reached
- Per-model summary of how many types got at least one example
"""

import argparse
import json
import os
from collections import defaultdict as dd

from delphin import itsdb, derivation
from delphin.tokens import YYTokenLattice


def parse_group_models(jsd_payload):
    models = set()
    for t in jsd_payload.get('types', []):
        md = t.get('models', {})
        for side in ('A', 'B'):
            s = md.get(side)
            if not s or 'GRP{' not in s:
                continue
            inside = s[s.find('{') + 1 : s.rfind('}')]
            for name in inside.split(','):
                name = name.strip()
                if name:
                    models.add(name)
    return models


def find_constituent(lattice, start, end, ex_text):
    try:
        left = lattice.tokens[start].lnk.data[0]
        right = lattice.tokens[end-1].lnk.data[1]
        return ex_text[left:right]
    except Exception:
        return ""


def collect_for_types_in_suite(suite_dir, types_of_interest, cap_per_type, progress_every=200, model_name=None, quiet=False):
    ex_by_type = {t: [] for t in types_of_interest}
    def all_full():
        return all(len(v) >= cap_per_type for v in ex_by_type.values())

    db = itsdb.TestSuite(suite_dir)
    processed = 0
    if not quiet:
        print(f"[scan] {model_name or suite_dir}: start", flush=True)

    for response in db.processed_items():
        processed += 1
        if not quiet and processed % max(1, progress_every) == 0:
            remaining = sum(max(0, cap_per_type - len(v)) for v in ex_by_type.values())
            print(f"[scan] {model_name or suite_dir}: {processed} items, remaining_slots={remaining}", flush=True)

        if all_full():
            if not quiet:
                print(f"[scan] {model_name or suite_dir}: done (early stop; caps reached)", flush=True)
            break

        results = response.get('results', [])
        if not results:
            continue
        sent = response.get('i-input', '')
        lat_str = response.get('p-input', '')
        try:
            lattice = YYTokenLattice.from_string(lat_str)
        except Exception:
            lattice = None
        try:
            deriv = derivation.from_string(results[0].get('derivation', ''))
        except Exception:
            continue

        for pt in deriv.preterminals():
            tname = getattr(pt, 'type', None)
            if not tname or tname not in types_of_interest:
                continue
            if len(ex_by_type[tname]) >= cap_per_type:
                continue
            cons = ""
            if lattice is not None and hasattr(pt, 'start') and hasattr(pt, 'end'):
                cons = find_constituent(lattice, pt.start, pt.end, sent)
            ex_by_type[tname].append({"sentence": sent, "constituent": cons})

    return ex_by_type


def main():
    ap = argparse.ArgumentParser(description="Collect examples for JSD top-contributing lexical types, per model.")
    ap.add_argument("jsd_files", nargs="+", help="Input JSD top-contributors JSON file(s).")
    ap.add_argument("--data-dir", required=True, help="Directory with itsdb suites; one subdir per model.")
    ap.add_argument("--output-dir", required=True, help="Where to write the enriched JSON(s).")
    ap.add_argument("--max-per-model", type=int, default=10, help="Max examples per type per model (default 10).")
    ap.add_argument("--progress-every", type=int, default=200, help="Heartbeat every N processed items (default 200).")
    ap.add_argument("--quiet", action="store_true", help="Suppress progress output.")
    ap.add_argument("--restrict-sides", choices=["A","B","both"], default="both",
                    help="Restrict to models on side A, side B, or both (default both).")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        suites = [d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))]
    except Exception as e:
        raise SystemExit(f"Could not list --data-dir: {e}")

    for jsd_path in args.jsd_files:
        with open(jsd_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        types_list = [t.get("type") for t in payload.get("types", []) if t.get("type")]
        types_set = set(types_list)

        union_models = parse_group_models(payload)

        side_models = set()
        if args.restrict_sides in ("A", "B"):
            for t in payload.get("types", []):
                md = t.get("models", {})
                tag = md.get(args.restrict_sides)
                if tag and 'GRP{' in tag:
                    inside = tag[tag.find('{') + 1 : tag.rfind('}')]
                    side_models.update([s.strip() for s in inside.split(',') if s.strip()])

        if side_models:
            allow = side_models
        else:
            allow = union_models if union_models else set(suites)

        examples = {t: {} for t in types_set}

        for model in suites:
            if model not in allow:
                continue
            print(f"[scan] Processing model: {model}", flush=True)
            suite_dir = os.path.join(args.data_dir, model)
            ex_by_type = collect_for_types_in_suite(
                suite_dir, types_set, args.max_per_model,
                progress_every=args.progress_every, model_name=model, quiet=args.quiet
            )
            if not args.quiet:
                filled = sum(1 for lst in ex_by_type.values() if lst)
                print(f"[scan] {model}: collected types with examples = {filled}/{len(types_set)}", flush=True)
            for t, lst in ex_by_type.items():
                if lst:
                    examples[t][model] = lst

        enriched = dict(payload)
        enriched_types = []
        for t in payload.get("types", []):
            tcopy = dict(t)
            tname = t.get("type")
            tcopy["examples"] = examples.get(tname, {})
            enriched_types.append(tcopy)
        enriched["types"] = enriched_types

        stem = os.path.splitext(os.path.basename(jsd_path))[0]
        out_path = os.path.join(args.output_dir, f"{stem}-examples.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(enriched, f, ensure_ascii=False, indent=2)
        if not args.quiet:
            print(f"[done] Wrote: {out_path}", flush=True)


if __name__ == "__main__":
    main()
