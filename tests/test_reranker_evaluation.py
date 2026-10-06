"""Portable evaluation tests: stdlib only, no search, models, or catalogue required.

Run: python -m unittest discover -s tests -p test_reranker_evaluation.py -v
"""
from __future__ import annotations

import copy
import ast
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import reranker_evaluation as ev


def fixture(count=1, duplicate=False):
    rows = [{"id": f"q{i:03d}", "query": "theme" if duplicate else f"theme {i}",
             "duplicate_of": "q001" if duplicate and i > 1 else None}
            for i in range(1, count + 1)]
    manifest = {"version": 1, "queries": rows, "query_count": count,
                "unique_query_count": 1 if duplicate else count}
    protocol = {
        "query_manifest_digest": ev.digest(manifest), "assessment_mode": "assistant_draft",
        "duplicate_policy": "all", "include_bm25": False,
        "k": 10, "candidate_depth": 60, "pool_depth": 20, "pool_seed": 55,
        "relevant_threshold": 2, "bootstrap_samples": 200, "bootstrap_seed": 7,
        "query_intents": {r["id"]: r["query"] for r in rows},
        "categories": {}, "excluded_queries": {}, "evidence_policy": "catalogue only",
    }
    books = {b: {"title": b, "description": f"Documented theme for book {b}."}
             for b in "abcde"}
    run = {
        "query_manifest": manifest, "status": "complete", "catalogue_fingerprint": "catalogue-v1",
        "index_generation": "index-v1", "candidate_depth": 60, "systems": list(ev.SYSTEMS),
        "include_bm25": False, "books": books,
        "queries": [{"id": r["id"], "query": r["query"], "status": "ok",
                     "rankings": {"fusion": list("abcde"), "without_reranker": list("abcde"),
                                  "reranked": list("bdcae")}, "candidates": list("abcde"),
                     "latency_ms": 100 + i, "timings_ms": {"rerank": 25},
                     "rerank_backend": "test-reranker", "rerank_fallback": ""}
                    for i, r in enumerate(rows)],
    }
    return run, protocol


def judge(pool, reviewed=True, grade_map=None):
    grade_map = grade_map or dict(zip("abcde", [0, 3, 1, 2, 0]))
    pool = copy.deepcopy(pool)
    for row in pool["queries"]:
        for item in row["books"]:
            item["judgment"] = {
                "status": "graded", "grade": grade_map[item["book_id"]], "reason": "Fixture judgment.",
                "evidence": [{"field": "description", "quote": item["metadata"]["description"]}],
                "assessor": "test assistant", "assessor_type": "assistant",
                "human_reviewed": reviewed, "reviewer": "test human" if reviewed else "",
            }
    return pool


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.run, self.protocol = fixture()
        self.pool = ev.make_pool(self.run, self.protocol)

    def test_numbered_bengali_queries_bom_comments_and_duplicates(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "queries.txt"
            path.write_text("\ufeff# ignore\n১। একটি বিষয়\n2. another\n৩। একটি বিষয়\n", encoding="utf-8")
            manifest = ev.read_queries(path)
        self.assertEqual(manifest["query_count"], 3)
        self.assertEqual(manifest["unique_query_count"], 2)
        self.assertEqual(manifest["queries"][0]["query"], "একটি বিষয়")
        self.assertEqual(manifest["queries"][2]["duplicate_of"], "q001")

    def test_cross_pc_formatting_changes_keep_the_same_queries(self):
        with tempfile.TemporaryDirectory() as temp:
            original = Path(temp) / "queries.txt"
            copied = Path(temp) / "copied-queries.txt"
            original.write_bytes("1. একটি বিষয়\n2. another query\n".encode("utf-8"))
            copied.write_bytes(("\ufeff# Comment added on the other PC\r\n\r\n"
                                "১। একটি বিষয়\r\n২। another query\r\n").encode("utf-8"))
            prepared, current = ev.read_queries(original), ev.read_queries(copied)
            self.assertNotEqual(prepared["source_sha256"], current["source_sha256"])
            self.assertNotEqual(prepared, current)
            protocol = ev.fresh_protocol(prepared, None, None)
            protocol["query_intents"]["q001"] = "Confirmed meaning"
            refreshed = ev.fresh_protocol(current, prepared, protocol)
            ev.validate_protocol(current, refreshed)
            self.assertEqual(refreshed["query_intents"]["q001"], "Confirmed meaning")

    def test_lf_and_crlf_are_accepted_in_both_directions(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "queries.txt"
            path.write_bytes(b"1. first\n2. second\n")
            lf = ev.read_queries(path)
            path.write_bytes(b"1. first\r\n2. second\r\n")
            crlf = ev.read_queries(path)
        ev.validate_protocol(crlf, ev.fresh_protocol(crlf, lf, ev.fresh_protocol(lf, None, None)))
        ev.validate_protocol(lf, ev.fresh_protocol(lf, crlf, ev.fresh_protocol(crlf, None, None)))

    def test_changed_added_removed_or_reordered_queries_get_fresh_snapshots(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "queries.txt"
            path.write_bytes(b"1. first\n2. second\n")
            prepared = ev.read_queries(path)
            previous = ev.fresh_protocol(prepared, None, None)
            previous["query_intents"]["q001"] = "Confirmed first intent"
            previous["query_guidance"]["q001"] = "Guidance for first only"
            variants = [b"1. changed\n2. second\n", b"1. first\n", b"1. first\n2. second\n3. third\n",
                        b"1. second\n2. first\n"]
            for content in variants:
                with self.subTest(content=content):
                    path.write_bytes(content)
                    current = ev.read_queries(path)
                    protocol = ev.fresh_protocol(current, prepared, previous)
                    ev.validate_protocol(current, protocol)
                    self.assertEqual(protocol["query_manifest_digest"], ev.digest(current))
                    for row in current["queries"]:
                        if row["query"] == "first":
                            self.assertEqual(protocol["query_intents"][row["id"]], "Confirmed first intent")
                        else:
                            self.assertNotIn(row["id"], protocol["query_guidance"])

    def test_graded_ndcg_known_example_and_shared_ideal(self):
        grades = dict(zip("abcde", [0, 3, 1, 2, 0]))
        ideal = 7 + 3 / math.log2(3) + 1 / math.log2(4)
        actual = 7 / math.log2(3) + 1 / math.log2(4) + 3 / math.log2(5)
        self.assertAlmostEqual(ev.ndcg(list("abcde"), grades, 5), actual / ideal)
        self.assertAlmostEqual(ev.ndcg(list("bdcae"), grades, 5), 1)
        # A run retrieving just the weak match must not normalize its own list to 1.
        self.assertLess(ev.ndcg(["c"], grades), 0.2)

    def test_fixed_precision_denominator_and_relevance_threshold(self):
        metrics = ev.query_metrics(["a", "b"], ["a", "b"], {"a": 1, "b": 2}, 10, 60)
        self.assertEqual(metrics["precision_at_10"], 0.1)
        self.assertEqual(metrics["mrr_at_10"], 0.5)
        self.assertEqual(metrics["candidate_pooled_recall_at_60"], 1)

    def test_zero_relevance_has_explicit_conventions(self):
        self.assertEqual(ev.ndcg(["a"], {"a": 0}), 0)
        self.assertIsNone(ev.query_metrics(["a"], ["a"], {"a": 0}, 10, 60)["candidate_pooled_recall_at_60"])
        assessment = judge(self.pool, grade_map={b: 0 for b in "abcde"})
        payload = ev.report(self.run, self.protocol, assessment)
        self.assertEqual(payload["scored_query_rows"], 1)
        self.assertEqual(payload["no_known_relevant_queries"], ["q001"])

    def test_pool_is_deterministic_shuffled_and_blinded(self):
        self.assertEqual(self.pool, ev.make_pool(self.run, self.protocol))
        row = self.pool["queries"][0]
        self.assertNotEqual([b["book_id"] for b in row["books"]], list("abcde"))
        self.assertNotIn("rankings", row)
        self.assertNotIn("rerank_backend", row)
        self.assertTrue(all(b["judgment"]["grade"] is None for b in row["books"]))

    def test_pool_includes_union_of_all_systems_at_fixed_depth(self):
        ids = [f"book-{i}" for i in range(30)]
        self.run["books"] = {b: {"title": b} for b in ids}
        row = self.run["queries"][0]
        row["rankings"] = {"fusion": ids, "without_reranker": ids,
                           "reranked": list(reversed(ids))}
        row["candidates"] = ids
        pool = ev.make_pool(self.run, self.protocol)
        self.assertEqual({b["book_id"] for b in pool["queries"][0]["books"]}, set(ids))

    def test_unresolved_decisions_block_pool(self):
        self.protocol["assessment_mode"] = None
        with self.assertRaisesRegex(ValueError, "unresolved decisions"):
            ev.make_pool(self.run, self.protocol)

    def test_paired_candidate_mismatch_and_duplicate_ids_are_rejected(self):
        self.run["queries"][0]["rankings"]["reranked"] = ["a"]
        with self.assertRaisesRegex(ValueError, "different candidates"):
            ev.make_pool(self.run, self.protocol)
        self.run["queries"][0]["rankings"]["reranked"] = list("aabcde")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            ev.make_pool(self.run, self.protocol)

    def test_incomplete_run_cannot_be_scored(self):
        self.run["status"] = "failed"
        with self.assertRaisesRegex(ValueError, "incomplete/failed"):
            ev.make_pool(self.run, self.protocol)

    def test_pending_and_unjudgeable_are_not_irrelevant(self):
        for status in ("pending", "unjudgeable"):
            pool = copy.deepcopy(self.pool)
            pool["queries"][0]["books"][0]["judgment"].update(status=status, reason="Insufficient metadata.")
            with self.assertRaisesRegex(ValueError, "no final metrics"):
                ev.report(self.run, self.protocol, pool)
            provisional = ev.report(self.run, self.protocol, pool, allow_partial=True)
            self.assertEqual(provisional["status"], "provisional")
            self.assertEqual(provisional["scored_query_rows"], 0)

    def test_unreviewed_assistant_grades_cannot_count_in_final_metrics(self):
        assessment = judge(self.pool, reviewed=False)
        with self.assertRaisesRegex(ValueError, "no final metrics"):
            ev.report(self.run, self.protocol, assessment)
        self.assertEqual(ev.check_assessment(self.run, self.protocol, assessment)[1][0]["awaiting_human_review"], 5)

    def test_quotes_must_exist_in_catalogue_not_model_memory(self):
        assessment = judge(self.pool)
        assessment["queries"][0]["books"][0]["judgment"]["evidence"][0]["quote"] = "Invented historical fact."
        with self.assertRaisesRegex(ValueError, "quote is absent"):
            ev.report(self.run, self.protocol, assessment)

    def test_metadata_protocol_and_run_provenance_cannot_drift(self):
        assessment = judge(self.pool)
        assessment["queries"][0]["books"][0]["metadata"]["description"] = "Changed after annotation"
        with self.assertRaisesRegex(ValueError, "evidence changed"):
            ev.report(self.run, self.protocol, assessment)
        assessment = judge(self.pool)
        self.protocol["pool_seed"] += 1
        with self.assertRaisesRegex(ValueError, "stale/altered"):
            ev.report(self.run, self.protocol, assessment)

    def test_boolean_grades_are_not_valid_integers(self):
        assessment = judge(self.pool)
        assessment["queries"][0]["books"][0]["judgment"]["grade"] = True
        with self.assertRaisesRegex(ValueError, "integer 0-3"):
            ev.report(self.run, self.protocol, assessment)

    def test_duplicate_policy_changes_units_not_reported_rows(self):
        run, protocol = fixture(count=2, duplicate=True)
        # Simulate two different rankings for the same query.
        run["queries"][1]["rankings"]["reranked"] = list("abcde")
        assessment = judge(ev.make_pool(run, protocol))
        payload = ev.report(run, protocol, assessment)
        self.assertEqual(payload["evaluation_units"], 2)
        protocol["duplicate_policy"] = "unique"
        assessment = judge(ev.make_pool(run, protocol))
        payload = ev.report(run, protocol, assessment)
        self.assertEqual(payload["evaluation_units"], 1)
        self.assertEqual(payload["scored_query_rows"], 2)

    def test_same_query_book_must_have_consistent_grades(self):
        run, protocol = fixture(count=2, duplicate=True)
        assessment = judge(ev.make_pool(run, protocol))
        assessment["queries"][1]["books"][0]["judgment"]["grade"] = 0
        if assessment["queries"][1]["books"][0]["book_id"] in "ae":
            assessment["queries"][1]["books"][0]["judgment"]["grade"] = 3
        with self.assertRaisesRegex(ValueError, "consistent grades"):
            ev.report(run, protocol, assessment)

    def test_bootstrap_is_paired_reproducible_and_zero_for_ties(self):
        self.assertEqual(ev.bootstrap([0.1, -0.2, 0.3], 500, 7), ev.bootstrap([0.1, -0.2, 0.3], 500, 7))
        self.assertEqual(ev.bootstrap([0, 0], 200, 7)["ci_95"], [0, 0])

    def test_fallback_blocks_positive_gain_claim(self):
        self.run["queries"][0]["rerank_fallback"] = "model unavailable"
        payload = ev.report(self.run, self.protocol, judge(ev.make_pool(self.run, self.protocol)))
        self.assertFalse(payload["positive_mean_gain_ci_excludes_zero"])
        self.assertEqual(payload["reranker_fallback_queries"], ["q001"])

    def test_completed_thirty_query_report_and_outputs(self):
        run, protocol = fixture(count=30)
        payload = ev.report(run, protocol, judge(ev.make_pool(run, protocol)))
        self.assertEqual(payload["scored_query_rows"], 30)
        self.assertAlmostEqual(payload["overall"]["reranked"]["ndcg_at_10"], 1)
        self.assertGreater(payload["comparisons"]["reranked_vs_without_reranker"]["mean_delta"], 0)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "report.json"
            ev.write_report(payload, path)
            self.assertTrue(path.with_suffix(".csv").exists())
            self.assertIn("assistant_draft", path.with_suffix(".md").read_text(encoding="utf-8"))

    def test_offline_cli_does_not_need_third_party_packages(self):
        with tempfile.TemporaryDirectory() as temp:
            run_path, protocol_path, assessment_path = [Path(temp) / name for name in
                                                       ("run.json", "protocol.json", "assessment.json")]
            ev.write_json(run_path, self.run)
            ev.write_json(protocol_path, self.protocol)
            ev.write_json(assessment_path, judge(self.pool))
            result = subprocess.run([sys.executable, "-S", str(ROOT / "reranker_evaluation.py"), "score",
                                     "--run", str(run_path), "--protocol", str(protocol_path),
                                     "--assessment", str(assessment_path), "--out", str(Path(temp) / "report.json")],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_search_pc_export_uses_one_search_per_query_and_keeps_raw_metadata(self):
        # Mock the search boundary only; no real query or model is executed on this PC.
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            query_path = temp / "queries.txt"
            query_path.write_text("1. theme one\n2. theme two\n", encoding="utf-8")
            manifest = ev.read_queries(query_path)
            _, protocol = fixture(count=2)
            protocol["query_manifest_digest"] = ev.digest(manifest)
            protocol["query_intents"] = {r["id"]: r["query"] for r in manifest["queries"]}
            protocol["query_intents"]["q001"] = "Confirmed meaning of theme one"
            protocol["query_guidance"] = {"q001": "Only applies to theme one"}
            manifest_path, protocol_path, out = [temp / n for n in ("queries.json", "protocol.json", "run.json")]
            ev.write_json(manifest_path, manifest)
            ev.write_json(protocol_path, protocol)
            index_path = temp / "index.json"
            index_path.write_text("{}", encoding="utf-8")
            book = SimpleNamespace(model_dump=lambda **kw: {
                "title": "Source title", "description": "Full original catalogue description.",
                "author_bio": "Source author biography.", "popularity": 0.9, "metadata_quality": 1.0,
            })
            calls = []

            def search(query, **kwargs):
                calls.append((query, kwargs))
                return SimpleNamespace(
                    candidates=["b"], rerank_ablation=["b"], hits=[SimpleNamespace(book=SimpleNamespace(book_id="b"))],
                    timings_ms={"rerank": 12}, applied_filters=SimpleNamespace(model_dump=lambda **kw: {}),
                    rerank_backend="fixture", rerank_fallback="",
                )

            engine = SimpleNamespace(search=search, records={"b": SimpleNamespace(book=book)},
                                     reranker=SimpleNamespace(model_name="fixture-model"))
            index = SimpleNamespace(catalogue_fingerprint="fp", generation="generation",
                                    model_dump=lambda **kw: {"generation": "generation"})
            settings = SimpleNamespace(index_manifest_path=index_path,
                                       model_dump=lambda **kw: {"use_reranker": True})
            modules = {
                "config": SimpleNamespace(settings=settings),
                "search.core.schemas": SimpleNamespace(SearchOptions=lambda **kw: SimpleNamespace(**kw)),
                "search.engine": SimpleNamespace(SearchEngine=SimpleNamespace(load=lambda settings: engine)),
                "search.indexing.manifest": SimpleNamespace(IndexManifest=SimpleNamespace(model_validate_json=lambda raw: index)),
            }
            # The target PC checkout has different line endings, BOM, and labels.
            query_path.write_bytes("\ufeff১। theme one\r\n২। theme two\r\n".encode("utf-8"))
            with patch.dict(sys.modules, modules):
                self.assertEqual(ev.run_searches(manifest_path, protocol_path, query_path, out), 0)
            self.assertEqual(len(calls), 3)  # one discarded warmup, then two queries
            self.assertTrue(all(call[1]["options"].compare_rerank for call in calls))
            self.assertTrue(all(call[1]["options"].trace is False for call in calls))
            bundle = ev.read_json(out)
            refreshed = ev.read_json(protocol_path)
            ev.validate_bundle(bundle, refreshed)
            self.assertEqual(bundle["books"]["b"]["description"], "Full original catalogue description.")
            self.assertNotIn("popularity", bundle["books"]["b"])
            self.assertNotIn("metadata_quality", bundle["books"]["b"])
            self.assertEqual(bundle["effective_reranker_model"], "fixture-model")
            self.assertEqual(bundle["query_source"]["sha256"], ev.read_queries(query_path)["source_sha256"])
            self.assertEqual(bundle["query_manifest"], ev.read_queries(query_path))
            self.assertEqual(refreshed["query_intents"]["q001"], "Confirmed meaning of theme one")
            old_archive = Path(bundle["previous_evaluation_archive"])
            self.assertEqual(ev.read_json(old_archive / "queries.json"), manifest)
            self.assertEqual(bundle["protocol"], refreshed)
            # A repeated run uses changed/added queries and archives all old assessment/results.
            (temp / "assessment.json").write_text("Old reviewed grades", encoding="utf-8")
            (temp / "report.json").write_text("Old report", encoding="utf-8")
            query_path.write_text("1. theme changed\n2. theme two\n3. extra theme\n", encoding="utf-8")
            with patch.dict(sys.modules, modules):
                self.assertEqual(ev.run_searches(manifest_path, protocol_path, query_path, out), 0)
            rerun, fresh = ev.read_run_protocol(out)
            ev.validate_bundle(rerun, fresh)
            self.assertEqual([r["query"] for r in rerun["queries"]], ["theme changed", "theme two", "extra theme"])
            self.assertEqual(len(calls), 7)  # two-query run, then three-query run; warmup each time
            self.assertNotIn("q001", fresh["query_guidance"])
            self.assertEqual(fresh["assessment_mode"], "assistant_draft")
            self.assertFalse((temp / "assessment.json").exists())
            archive = Path(rerun["previous_evaluation_archive"])
            self.assertEqual(ev.read_json(archive / "run.json"), bundle)
            self.assertEqual((archive / "assessment.json").read_text(encoding="utf-8"), "Old reviewed grades")
            self.assertEqual((archive / "report.json").read_text(encoding="utf-8"), "Old report")

    def test_run_protocol_can_be_created_without_prepare(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "queries.txt"
            path.write_text("1. first query", encoding="utf-8")
            manifest = ev.read_queries(path)
            protocol = ev.fresh_protocol(manifest, None, None)
            ev.validate_protocol(manifest, protocol)
            self.assertEqual(protocol["assessment_mode"], "assistant_draft")
            self.assertFalse(protocol["include_bm25"])

    def test_transferred_run_embeds_protocol_and_ignores_a_previous_runs_settings(self):
        run = copy.deepcopy(self.run)
        protocol = ev.fresh_protocol(run["query_manifest"], run["query_manifest"], self.protocol)
        run["protocol"] = protocol
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "run.json"
            ev.write_json(out, run)
            loaded_run, loaded_protocol = ev.read_run_protocol(out)
            self.assertEqual(loaded_protocol, protocol)
            ev.make_pool(loaded_run, loaded_protocol)
            stale = dict(protocol, run_id="previous-run", assessment_mode="assistant_final")
            ev.write_json(Path(temp) / "protocol.json", stale)
            self.assertEqual(ev.read_run_protocol(out)[1], protocol)
            # Confirmed edits associated with this run are used by subsequent pooling/scoring.
            updated = dict(protocol, categories={"q001": "semantic"})
            ev.write_json(Path(temp) / "protocol.json", updated)
            self.assertEqual(ev.read_run_protocol(out)[1], updated)

    def test_missing_snapshot_can_reuse_confirmed_intents_from_protocol_sources(self):
        old = ev.fresh_protocol(self.run["query_manifest"], self.run["query_manifest"], self.protocol)
        old["query_intents"]["q001"] = "Previously confirmed meaning"
        refreshed = ev.fresh_protocol(self.run["query_manifest"], None, old)
        self.assertEqual(refreshed["query_intents"]["q001"], "Previously confirmed meaning")

    def test_unfinished_replacement_flag_is_an_invalid_protocol(self):
        self.protocol["pending_query_replacements"] = {"q001": None}
        with self.assertRaisesRegex(ValueError, "finalize query replacements"):
            ev.validate_protocol(self.run["query_manifest"], self.protocol, require_decisions=False)

    def test_run_output_cannot_overwrite_the_query_source(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "queries.txt"
            path.write_text("1. first query", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "separate from"):
                ev.run_searches(Path(temp) / "queries.json", Path(temp) / "protocol.json", path, path)
            self.assertEqual(path.read_text(encoding="utf-8"), "1. first query")

    def test_actual_score_blend_ablation_keeps_other_signals_and_availability(self):
        # Load the actual two pure ranking functions without importing model dependencies.
        # This verifies the production formula used by the engine's paired export.
        source = ast.parse((ROOT / "search/ranking/rerank.py").read_text(encoding="utf-8"))
        functions = [node for node in source.body if isinstance(node, ast.FunctionDef)
                     and node.name in {"final_scores", "_graph_confidence"}]
        module = ast.parse("from __future__ import annotations")
        module.body.extend(functions)
        settings = SimpleNamespace(score_weights={"semantic": .55, "fusion": .25, "graph": .1,
                                                  "quality": .05, "popularity": .05},
                                   unavailable_penalty=.2)
        namespace = {"default_settings": settings}
        exec(compile(module, "production_score_blend", "exec"), namespace)
        score = namespace["final_scores"]
        candidates = [SimpleNamespace(book_id="a", fusion_score=.1, scores={"graph": .4}),
                      SimpleNamespace(book_id="b", fusion_score=.9, scores={"graph": .8})]
        records = {"a": SimpleNamespace(book=SimpleNamespace(metadata_quality=.4, popularity=.2, available=False)),
                   "b": SimpleNamespace(book=SimpleNamespace(metadata_quality=.9, popularity=.8, available=True))}
        actual = score(candidates, records, [1.0, .01], settings)
        ablation = score(candidates, records, [0.0, 0.0], settings)
        self.assertEqual([r[0].book_id for r in actual], ["a", "b"])
        self.assertEqual([r[0].book_id for r in ablation], ["b", "a"])
        actual_by_id, base_by_id = [{r[0].book_id: r for r in rows} for rows in (actual, ablation)]
        for book_id, semantic in (("a", 1), ("b", .01)):
            before, after = base_by_id[book_id], actual_by_id[book_id]
            for signal in ("fusion", "graph", "quality", "popularity", "availability"):
                self.assertEqual(before[2].get(signal), after[2].get(signal))
            availability = .8 if book_id == "a" else 1
            self.assertAlmostEqual(after[1] - before[1], .55 * semantic * availability)


if __name__ == "__main__":
    unittest.main()
