#!/usr/bin/env python3
"""Evaluate existing memories only. Default: dry-run; no API calls."""

import argparse
import ast
import contextlib
import datetime as dt
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import sys
import threading
import tempfile
import time
import types
import typing
import uuid

UPSTREAM_SHA = "db80b6a7c591e0ea730a058e9f5fc4eb06572299"

SOURCE_HASHES = {
    "main.py": "358cab67aebf68174f260202524ffaff430e882cedf73c89a91b14a6ade54cd9",
    "test_locomo10.py": "d5fef37a21700e3c721b1556e70bef2cf53e35f8c8baca24f175a9755af02978",
    "simplemem/core/hybrid_retriever.py": "63d6a3522e8a4c5d3a073b5dbf8ac003844c37993bf16033bb5c9a3807db5381",
    "simplemem/core/answer_generator.py": "ba8946134e9db0de63da00a9f223229800d2487c6fd34d22d7c5296093d0470c",
    "simplemem/core/utils/embedding.py": "da72ef639c5c0f6a6d9a0ecd5f4f45f65019da52e32191571ac718ff0b7c6d68",
    "simplemem/core/utils/llm_client.py": "19533ebf5b3a12096c70ae35df54305d8b307b22240377ce1977e2aaff527bd8",
    "simplemem/core/database/vector_store.py": "a7519fa27cfcfdceb4e146780c948c96e2840cea11965a94c296b786623fff29",
    "simplemem/core/database/vector_store_backend.py": "2406f5c267f9013af0dc34c8668def8b20b3e66326827c57d975f6129f502c26",
    "simplemem/core/models/memory_entry.py": "487209f5d6981a12568ec3f6f4597d2bd1437a82a0ef9af85c5dc755bc2947c8",
    "simplemem/core/settings.py": "3ca77f1d035fd309a541fdc7555e8519ddb9d77ce4f82161f1fa027619784de2",
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest(root):
    """Read-only fingerprint; reject symlinks and special files."""
    result = {}
    for p in sorted(root.rglob("*")):
        if p.is_symlink():
            raise ValueError("Database symlinks are not supported")
        if p.is_dir():
            continue
        if not p.is_file():
            raise ValueError("Database contains a special file")
        st = p.stat()
        result[str(p.relative_to(root))] = {
            "size": st.st_size,
            "sha256": digest(p),
        }
    if not result:
        raise ValueError("Database directory has no files")
    return result


def verify_source(root):
    for name, expected in SOURCE_HASHES.items():
        if digest(root / name) != expected:
            raise ValueError(
                f"Upstream source differs: {name}. "
                f"Use --source-root with a clean checkout of {UPSTREAM_SHA}"
            )


def save(path, data):
    """Atomic checkpoint."""
    temp = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix="." + path.name + ".",
            delete=False,
        ) as f:
            temp = Path(f.name)
            json.dump(
                data, f, ensure_ascii=False, indent=2, allow_nan=False
            )
            f.write("\n")
        os.replace(temp, path)
    finally:
        if temp is not None and temp.exists():
            temp.unlink()


def contained(a, b):
    return a == b or b in a.parents


def make_plan(args):
    db = args.db_path.resolve()
    dataset = args.dataset.resolve()

    if args.db_path.is_symlink() or not db.is_dir():
        raise ValueError(
            "A real local LanceDB directory is required; "
            "Parquet is not a database"
        )

    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.table_name):
        raise ValueError("Unsafe table name")

    tables = sorted(
        p.name[:-6] for p in db.glob("*.lance") if p.is_dir()
    )
    if args.table_name not in tables:
        raise ValueError(
            f"Table absent. Existing table directories: {tables}"
        )

    data = json.loads(dataset.read_text(encoding="utf-8"))

    if not 0 <= args.sample_index < len(data):
        raise ValueError("Invalid sample index")

    raw = data[args.sample_index]
    end = (
        len(raw["qa"])
        if args.question_end is None
        else args.question_end
    )

    if not 0 <= args.question_start < end <= len(raw["qa"]):
        raise ValueError(
            "Question range is zero-based [start,end), "
            "nonempty and within sample"
        )

    n = end - args.question_start

    return {
        "upstream_sha": UPSTREAM_SHA,
        "db_path": str(db),
        "table_name": args.table_name,
        "dataset": str(dataset),
        "dataset_sha256": digest(dataset),
        "sample_index": args.sample_index,
        "sample_id": raw.get("sample_id"),
        "sample_question_count": len(raw["qa"]),
        "question_range": [args.question_start, end],
        "selected_questions": n,
        "expected_memory_count": args.expected_memory_count,
        "estimated_logical_calls_without_retries": [4 * n, 8 * n],
        "memory_building_calls": 0,
        "note": (
            "Estimate only; category 5/empty retrieval can reduce calls; "
            "JSON and SDK retries increase them. "
            "No upstream output-token limit."
        ),
    }


def snapshot_database(source, destination):
    before = manifest(source)

    # Separate files: never hardlinks or symlinks.
    shutil.copytree(source, destination, symlinks=False)

    if (
        manifest(source) != before
        or manifest(destination) != before
    ):
        raise RuntimeError(
            "Database changed while copying; "
            "stop all writers and retry"
        )

    return before


class ForbiddenWrite(BaseException):
    pass


class ReadOnlyBackend:
    """Delegate only existing upstream read operations."""

    READS = frozenset(
        [
            "count",
            "get_all",
            "semantic_search",
            "keyword_search",
            "structured_search",
            "semantic_score_order",
            "keyword_score_order",
        ]
    )

    def __init__(self, backend):
        self._backend = backend

    def __getattr__(self, name):
        if name not in self.READS:
            raise ForbiddenWrite(
                "Non-read backend operation blocked: " + name
            )
        return getattr(self._backend, name)


def open_existing_store(
    snapshot, table_name, embedding, expected_count
):
    import lancedb
    from simplemem.core.database.vector_store_backend import (
        LanceDBVectorStoreBackend,
    )
    from simplemem.core.database.vector_store import VectorStore

    # Only the copied database is passed to LanceDB.
    connection = lancedb.connect(str(snapshot))

    if table_name not in connection.table_names():
        raise ValueError(
            "Table does not exist in snapshot; "
            "no table will be created"
        )

    table = connection.open_table(table_name)

    required = {
        "entry_id",
        "lossless_restatement",
        "keywords",
        "timestamp",
        "location",
        "persons",
        "entities",
        "topic",
        "vector",
    }

    if not required <= set(table.schema.names):
        raise ValueError("Incompatible original memory schema")

    dimension = getattr(
        table.schema.field("vector").type, "list_size", None
    )

    if dimension != embedding.dimension:
        raise ValueError(
            f"Vector dimension mismatch: "
            f"DB={dimension}, model={embedding.dimension}"
        )

    count = table.count_rows()

    if not count or (
        expected_count is not None and count != expected_count
    ):
        raise ValueError(
            f"Unexpected memory count: {count}, "
            f"expected={expected_count}"
        )

    # Bypass the constructor that could create a table.
    # Original search methods are retained unchanged.
    backend = LanceDBVectorStoreBackend.__new__(
        LanceDBVectorStoreBackend
    )
    backend.db_path = str(snapshot)
    backend.table_name = table_name
    backend.vector_dimension = dimension
    backend.db = connection
    backend.table = table
    backend._is_cloud_storage = False
    backend._fts_initialized = True

    store = VectorStore(
        db_path=str(snapshot),
        table_name=table_name,
        embedding_model=embedding,
        backend_factory=lambda dim: ReadOnlyBackend(backend),
    )

    return store, {
        "count": count,
        "dimension": dimension,
        "table_version": table.version,
        "snapshot": str(snapshot),
        "table_name": table_name,
    }


class Tee:
    def __init__(self, *streams, secret=""):
        self.streams = streams
        self.secret = secret

    def write(self, s):
        clean = (
            s.replace(self.secret, "[REDACTED]")
            if self.secret
            else s
        )
        for stream in self.streams:
            stream.write(clean)
            stream.flush()
        return len(s)

    def flush(self):
        for stream in self.streams:
            if not getattr(stream, "closed", False):
                stream.flush()

    def close(self):
        self.flush()

    def isatty(self):
        return False


def load_evaluator(root):
    """
    Load unchanged definitions without upstream module-level
    NLTK downloads or automatic model loading.
    """
    import dataclasses
    import collections
    import nltk
    from nltk.translate.bleu_score import (
        sentence_bleu,
        SmoothingFunction,
    )
    from nltk.translate.meteor_score import meteor_score
    from rouge_score import rouge_scorer

    module = types.ModuleType("_simplemem_existing_locomo")
    ns = module.__dict__

    ns.update(
        {
            name: getattr(typing, name)
            for name in ["List", "Dict", "Optional", "Union", "Any"]
        }
    )
    ns.update(
        Path=Path,
        time=time,
        json=json,
        statistics=statistics,
        dataclass=dataclasses.dataclass,
        defaultdict=collections.defaultdict,
        nltk=nltk,
        sentence_bleu=sentence_bleu,
        SmoothingFunction=SmoothingFunction,
        meteor_score=meteor_score,
        rouge_scorer=rouge_scorer,
        SimpleMemSystem=typing.Any,
        Dialogue=typing.Any,
        sentence_model=None,
    )

    reasons = {}

    try:
        from bert_score import score
        ns["bert_score"] = score
    except Exception as exc:
        reasons["bert"] = type(exc).__name__

        def missing_bert(*args, **kwargs):
            raise RuntimeError("BERTScore dependency unavailable")

        ns["bert_score"] = missing_bert

    try:
        from sentence_transformers import SentenceTransformer
        from sentence_transformers.util import pytorch_cos_sim

        ns["sentence_model"] = SentenceTransformer(
            "all-MiniLM-L6-v2", local_files_only=True
        )
        ns["pytorch_cos_sim"] = pytorch_cos_sim
    except Exception as exc:
        reasons["sbert"] = type(exc).__name__

    wanted = {
        "QA",
        "Turn",
        "Session",
        "Conversation",
        "EventSummary",
        "Observation",
        "LoCoMoSample",
        "parse_session",
        "parse_conversation",
        "load_locomo_dataset",
        "simple_tokenize",
        "calculate_rouge_scores",
        "calculate_bleu_scores",
        "calculate_bert_scores",
        "calculate_meteor_score",
        "calculate_sentence_similarity",
        "calculate_metrics",
        "aggregate_metrics",
        "LoCoMoTester",
    }

    parsed = ast.parse(
        (root / "test_locomo10.py").read_text(encoding="utf-8")
    )
    definitions = [
        node
        for node in parsed.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        and node.name in wanted
    ]

    if {node.name for node in definitions} != wanted:
        raise ValueError("Unexpected upstream evaluator structure")

    sys.modules[module.__name__] = module

    exec(
        compile(
            ast.Module(body=definitions, type_ignores=[]),
            str(root / "test_locomo10.py"),
            "exec",
        ),
        ns,
    )

    module.metric_load_failures = reasons
    return module


class MetricAudit:
    """
    Preserve original values during upstream printing;
    replace failed metric placeholders with null afterwards.
    """

    GROUPS = {
        "calculate_bert_scores": (
            "bert_precision",
            "bert_recall",
            "bert_f1",
        ),
        "calculate_sentence_similarity": ("sbert_similarity",),
        "calculate_meteor_score": ("meteor",),
        "calculate_bleu_scores": (
            "bleu1", "bleu2", "bleu3", "bleu4"
        ),
    }

    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.failures = {}

        for name, keys in self.GROUPS.items():
            original = getattr(evaluator, name)

            def audited(
                *args,
                _fn=original,
                _keys=keys,
                _name=name,
                **kwargs,
            ):
                reason = None

                if (
                    _name == "calculate_sentence_similarity"
                    and evaluator.sentence_model is None
                ):
                    reason = "SBERT model unavailable"

                if (
                    _name == "calculate_bert_scores"
                    and "bert" in evaluator.metric_load_failures
                ):
                    reason = "BERTScore dependency unavailable"

                buffer = io.StringIO()

                try:
                    with contextlib.redirect_stdout(buffer):
                        result = _fn(*args, **kwargs)
                except Exception as exc:
                    reason = type(exc).__name__
                    result = (
                        {key: 0.0 for key in _keys}
                        if len(_keys) > 1
                        else 0.0
                    )

                if "Error calculating" in buffer.getvalue():
                    reason = "Upstream metric computation failed"

                if reason:
                    for key in _keys:
                        self.failures[key] = reason

                return result

            setattr(evaluator, name, audited)

    def reset(self):
        self.failures = {}

        if self.evaluator.sentence_model is None:
            self.failures["sbert_similarity"] = (
                "SBERT model unavailable"
            )

        if "bert" in self.evaluator.metric_load_failures:
            for key in self.GROUPS["calculate_bert_scores"]:
                self.failures[key] = (
                    "BERTScore dependency unavailable"
                )

    def sanitize(self, result):
        if "llm_judge_score" in result["metrics"]:
            self.failures["llm_judge_score"] = (
                "LLM judge disabled; no judge API calls"
            )

        for key, value in list(result["metrics"].items()):
            if (
                isinstance(value, (float, int))
                and not math.isfinite(value)
            ):
                self.failures[key] = "Non-finite metric result"

        for key in self.failures:
            if key in result["metrics"]:
                result["metrics"][key] = None

        result["missing_metrics"] = dict(self.failures)
        result["metric_status"] = {
            key: ("missing" if value is None else "ok")
            for key, value in result["metrics"].items()
        }


class BudgetExceeded(BaseException):
    pass


class APIAudit:
    def __init__(self, budget, path, include_usage=False):
        self.budget = budget
        self.path = path
        self.include_usage = include_usage
        self.requests = 0
        self.events = []
        self.http_attempts = []
        self.question = None
        self.lock = threading.Lock()

    def summary(self, events=None):
        events = self.events if events is None else events
        known = [
            event["usage"]
            for event in events
            if event["usage"] is not None
        ]
        keys = ("prompt_tokens", "completion_tokens", "total_tokens")

        totals = (
            {
                key: sum(usage.get(key, 0) or 0 for usage in known)
                for key in keys
            }
            if known
            else None
        )

        return {
            "logical_sdk_calls": len(events),
            "calls_with_usage": len(known),
            "failed_sdk_calls": sum(
                event["status"] == "failed" for event in events
            ),
            "observed_usage_sum": totals,
            "all_sdk_calls_have_usage": (
                bool(events) and len(known) == len(events)
            ),
            "billing_total_verified": False,
            "note": (
                "Sum covers observed usage only; failed HTTP retries "
                "may have unreported billable tokens."
            ),
        }

    def persist(self):
        save(
            self.path,
            {
                "http_requests": self.requests,
                "request_budget": self.budget,
                "include_stream_usage": self.include_usage,
                "events": self.events,
                "http_attempts": self.http_attempts,
                **self.summary(),
            },
        )

    def request(self, request):
        with self.lock:
            if self.requests >= self.budget:
                raise BudgetExceeded(
                    "HTTP request budget exhausted"
                )

            self.requests += 1
            request.extensions["simplemem_audit_index"] = len(
                self.http_attempts
            )
            self.http_attempts.append(
                {
                    "attempt": self.requests,
                    "question_index": self.question,
                    "host": request.url.host,
                    "method": request.method,
                    "response_status": None,
                }
            )
            self.persist()

    def response(self, response):
        with self.lock:
            index = response.request.extensions.get(
                "simplemem_audit_index"
            )
            if index is not None:
                self.http_attempts[index]["response_status"] = (
                    response.status_code
                )
            self.persist()

    def install(self):
        from openai import OpenAI
        from openai.resources.chat.completions import Completions

        self.original_init = OpenAI.__init__
        self.original_create = Completions.create
        audit = self

        def init(client, *args, **kwargs):
            audit.original_init(client, *args, **kwargs)
            client._client.event_hooks.setdefault(
                "request", []
            ).append(audit.request)
            client._client.event_hooks.setdefault(
                "response", []
            ).append(audit.response)

        def create(resource, *args, **kwargs):
            if audit.include_usage and kwargs.get("stream"):
                kwargs["stream_options"] = {
                    **kwargs.get("stream_options", {}),
                    "include_usage": True,
                }

            event = {
                "question_index": audit.question,
                "requested_model": kwargs.get("model"),
                "response_models": [],
                "usage": None,
                "status": "started",
                "stream": bool(kwargs.get("stream")),
            }
            start = time.perf_counter()

            with audit.lock:
                audit.events.append(event)
                audit.persist()

            def observe(obj):
                model = getattr(obj, "model", None)
                if model and model not in event["response_models"]:
                    event["response_models"].append(model)

                usage = getattr(obj, "usage", None)
                if usage is not None:
                    event["usage"] = usage.model_dump()

            def done(status):
                event["status"] = status
                event["seconds"] = time.perf_counter() - start
                with audit.lock:
                    audit.persist()

            try:
                result = audit.original_create(
                    resource, *args, **kwargs
                )

                if kwargs.get("stream"):
                    class Stream:
                        def __iter__(self):
                            try:
                                for chunk in result:
                                    observe(chunk)
                                    yield chunk
                            except BaseException:
                                done("failed")
                                raise
                            else:
                                done("completed")

                        def __getattr__(self, name):
                            return getattr(result, name)

                    return Stream()

                observe(result)
                done("completed")
                return result

            except BaseException:
                done("failed")
                raise

        OpenAI.__init__ = init
        Completions.create = create

    def restore(self):
        from openai import OpenAI
        from openai.resources.chat.completions import Completions

        OpenAI.__init__ = self.original_init
        Completions.create = self.original_create


def local_components(args, snapshot):
    from simplemem.core.utils.embedding import EmbeddingModel
    from simplemem.core.settings import settings

    name = args.embedding_path or settings.EMBEDDING_MODEL

    if (
        name != "Qwen/Qwen3-Embedding-0.6B"
        and not Path(name).is_dir()
    ):
        raise ValueError(
            "Use official Qwen/Qwen3-Embedding-0.6B "
            "or its existing local model directory"
        )

    embedding = EmbeddingModel(model_name=name)

    if embedding.model_name != name or embedding.dimension != 1024:
        raise ValueError("Embedding fallback/dimension mismatch")

    config = embedding.model[0].auto_model.config

    if (
        getattr(config, "model_type", None) != "qwen3"
        or getattr(config, "hidden_size", None) != 1024
    ):
        raise ValueError(
            "Local model is not the expected "
            "Qwen3-Embedding-0.6B architecture"
        )

    store, info = open_existing_store(
        snapshot,
        args.table_name,
        embedding,
        args.expected_memory_count,
    )

    # Exercise original read paths without LLM calls.
    embedding.encode_single(
        "Read-only local embedding validation.", is_query=True
    )
    backend = store.backend
    first = backend.get_all()[0]

    backend.semantic_search(
        embedding.encode_single(
            "Memory validation.", is_query=True
        ).tolist(),
        top_k=1,
    )
    backend.keyword_search(["memory"], top_k=1)
    backend.structured_search(
        persons=first.metadata.get("persons")
        or ["__local_check__"],
        top_k=1,
    )

    info["embedding"] = {
        "model_name": embedding.model_name,
        "dimension": embedding.dimension,
        "model_type": embedding.model_type,
        "device": str(embedding.model.device),
        "query_prompt": embedding.supports_query_prompt,
        "max_seq_length": embedding.model.max_seq_length,
    }

    config_path = (
        Path(name) / "config.json"
        if Path(name).is_dir()
        else None
    )

    if config_path is None:
        from huggingface_hub import try_to_load_from_cache
        cached = try_to_load_from_cache(name, "config.json")
        if isinstance(cached, str):
            config_path = Path(cached)

    if config_path is not None and config_path.is_file():
        info["embedding"]["config_sha256"] = digest(config_path)
        info["embedding"]["cached_revision"] = (
            config_path.parent.name
            if config_path.parent.parent.name == "snapshots"
            else None
        )

    return store, info


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument(
        "--db-path",
        type=Path,
        required=True,
        help="Existing local LanceDB directory, NOT parquet export",
    )
    p.add_argument("--table-name", default="memory_entries")
    p.add_argument("--sample-index", type=int, default=0)
    p.add_argument(
        "--question-start",
        type=int,
        default=0,
        help="Zero-based inclusive",
    )
    p.add_argument(
        "--question-end",
        type=int,
        default=None,
        help="Zero-based exclusive; omit for rest of sample",
    )
    p.add_argument(
        "--result-file",
        type=Path,
        default=None,
        help="Must not exist; parent must already exist",
    )
    p.add_argument(
        "--source-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    p.add_argument(
        "--embedding-path",
        default=None,
        help="Existing local Qwen directory; use SAME weights",
    )
    p.add_argument("--expected-memory-count", type=int, default=None)

    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--check-local",
        action="store_true",
        help="Offline local validation; zero API calls",
    )
    mode.add_argument(
        "--allow-paid",
        action="store_true",
        help="Execute only after explicit cost approval",
    )

    p.add_argument("--max-api-requests", type=int, default=0)
    p.add_argument(
        "--include-stream-usage",
        action="store_true",
        help="Opt in to stream_options.include_usage metadata",
    )

    return p


def main(argv=None):
    args = parser().parse_args(argv)
    root = args.source_root.resolve()
    verify_source(root)
    plan = make_plan(args)

    if not args.check_local and not args.allow_paid:
        print(
            json.dumps(
                {"mode": "dry-run", "api_requests": 0, **plan},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.allow_paid and args.max_api_requests < 1:
        raise ValueError(
            "Explicit positive --max-api-requests is required"
        )

    # Local HF resources only; never upgrade or download models here.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, str(root))

    from simplemem.core.settings import settings

    secret = settings.OPENAI_API_KEY if args.allow_paid else ""

    if args.allow_paid:
        from urllib.parse import urlsplit

        url = urlsplit(settings.OPENAI_BASE_URL or "")

        if not secret:
            raise ValueError(
                "API key missing; use environment/protected secrets"
            )

        if (
            url.scheme != "https"
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise ValueError(
                "Use an HTTPS API base URL without embedded secrets"
            )

    result = (
        args.result_file
        or Path("existing-memory-" + uuid.uuid4().hex[:12] + ".json")
    ).resolve()

    source = args.db_path.resolve()
    run = result.with_name(result.name + ".artifacts")

    if (
        contained(result, source)
        or contained(run, source)
        or contained(source, run)
    ):
        raise ValueError(
            "Result/run directory must be disjoint "
            "from source database"
        )

    if run.exists():
        raise ValueError("Artifacts directory must NOT exist")

    if not result.parent.is_dir() or result.exists():
        raise ValueError(
            "Result parent must exist and result file must NOT exist"
        )

    with result.open("x", encoding="utf-8") as f:
        f.write("{}\n")

    run.mkdir(mode=0o700, exist_ok=False)
    os.chmod(result, 0o600)

    source_manifest = None
    audit = None
    start = time.perf_counter()

    output = {
        "mode": "check-local" if args.check_local else "paid",
        "status": "started",
        "plan": plan,
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "detailed_results": [],
        "source_db_unchanged": None,
        "memory_builder_instantiated": False,
        "versions": {
            dist.metadata["Name"]: dist.version
            for dist in importlib.metadata.distributions()
            if dist.metadata["Name"]
        },
    }

    try:
        with (
            (run / "console.log").open("x", encoding="utf-8") as log,
            contextlib.redirect_stdout(
                Tee(sys.stdout, log, secret=secret)
            ),
            contextlib.redirect_stderr(
                Tee(sys.stderr, log, secret=secret)
            ),
        ):
            source_manifest = snapshot_database(
                source, run / "database_snapshot"
            )
            save(run / "source_manifest.json", source_manifest)

            store, info = local_components(
                args, run / "database_snapshot"
            )
            output["memory"] = info

            evaluator = load_evaluator(root)
            output["metric_model_load_failures"] = (
                evaluator.metric_load_failures
            )

            samples = evaluator.load_locomo_dataset(args.dataset)
            sample = samples[args.sample_index]

            if len(sample.qa) != plan["sample_question_count"]:
                raise ValueError("Upstream loader changed QA count")

            metric_audit = MetricAudit(evaluator)

            # Local synthetic metric probe: no LLM calls.
            metric_audit.reset()
            probe = evaluator.calculate_metrics(
                "local validation", "local validation"
            )
            checked = {"metrics": probe}
            metric_audit.sanitize(checked)
            output["local_metric_probe"] = checked

            if args.allow_paid:
                from simplemem.core.utils.llm_client import LLMClient
                from simplemem.core.hybrid_retriever import (
                    HybridRetriever,
                )
                from simplemem.core.answer_generator import (
                    AnswerGenerator,
                )

                # No SimpleMemSystem or MemoryBuilder construction.
                llm = LLMClient()

                retriever = HybridRetriever(
                    llm,
                    store,
                    enable_planning=None,
                    enable_reflection=None,
                    max_reflection_rounds=None,
                    enable_parallel_retrieval=None,
                    max_retrieval_workers=None,
                )

                system = types.SimpleNamespace(
                    llm_client=llm,
                    hybrid_retriever=retriever,
                    answer_generator=AnswerGenerator(llm),
                )

                if "config" not in sys.modules:
                    cfg = types.ModuleType("config")
                    cfg.USE_JSON_FORMAT = settings.USE_JSON_FORMAT
                    sys.modules["config"] = cfg

                keys = [
                    "LLM_MODEL",
                    "USE_STREAMING",
                    "USE_JSON_FORMAT",
                    "ENABLE_THINKING",
                    "ENABLE_PLANNING",
                    "ENABLE_REFLECTION",
                    "MAX_REFLECTION_ROUNDS",
                    "SEMANTIC_TOP_K",
                    "KEYWORD_TOP_K",
                    "STRUCTURED_TOP_K",
                    "ENABLE_PARALLEL_RETRIEVAL",
                    "MAX_RETRIEVAL_WORKERS",
                ]
                output["config"] = {
                    key: getattr(settings, key) for key in keys
                }
                output["config"]["api_host"] = url.hostname

                tester = evaluator.LoCoMoTester(
                    system,
                    str(args.dataset),
                    use_llm_judge=False,
                )

                audit = APIAudit(
                    args.max_api_requests,
                    run / "api_calls.json",
                    args.include_stream_usage,
                )
                audit.install()

                # LLMClient existed before observer installation.
                llm.client._client.event_hooks.setdefault(
                    "request", []
                ).append(audit.request)
                llm.client._client.event_hooks.setdefault(
                    "response", []
                ).append(audit.response)

                for index in range(*plan["question_range"]):
                    audit.question = index
                    metric_audit.reset()
                    event_start = len(audit.events)
                    request_start = audit.requests

                    answer = tester._process_single_question(
                        sample.qa[index], index
                    )
                    metric_audit.sanitize(answer)

                    answer["question_index"] = index
                    answer["api"] = {
                        **audit.summary(audit.events[event_start:]),
                        "http_requests": (
                            audit.requests - request_start
                        ),
                        "failed_sdk_calls": sum(
                            event["status"] == "failed"
                            for event in audit.events[event_start:]
                        ),
                    }

                    output["detailed_results"].append(answer)
                    print(
                        "Audited missing metrics:",
                        answer["missing_metrics"],
                    )
                    save(result, output)

                valid = [
                    {
                        key: value
                        for key, value in row["metrics"].items()
                        if value is not None
                    }
                    for row in output["detailed_results"]
                ]
                categories = [
                    row["category"]
                    for row in output["detailed_results"]
                ]
                output["aggregated_metrics"] = (
                    evaluator.aggregate_metrics(valid, categories)
                )

            output["status"] = "completed"

    except BaseException as exc:
        output["status"] = "failed"
        output["error_type"] = type(exc).__name__
        print(
            "Evaluation stopped:", type(exc).__name__,
            file=sys.stderr,
        )

    finally:
        if audit is not None:
            audit.persist()
            audit.restore()
            output["api"] = {
                "http_requests": audit.requests,
                **audit.summary(),
            }
        else:
            output["api"] = {
                "http_requests": 0,
                "observed_usage_sum": None,
            }

        if source_manifest is not None:
            try:
                output["source_db_unchanged"] = (
                    manifest(source) == source_manifest
                )
            except Exception:
                output["source_db_unchanged"] = False

            if not output["source_db_unchanged"]:
                output["status"] = "failed"
                output["integrity_error"] = (
                    "Source DB changed externally"
                )

        output["dataset_unchanged"] = (
            digest(args.dataset) == plan["dataset_sha256"]
        )
        if not output["dataset_unchanged"]:
            output["status"] = "failed"

        output["elapsed_seconds"] = time.perf_counter() - start
        output["questions_completed"] = len(
            output["detailed_results"]
        )

        save(result, output)
        print("Result:", result, "status:", output["status"])

    return 0 if output["status"] == "completed" else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print("Validation failed:", str(exc), file=sys.stderr)
        sys.exit(2)