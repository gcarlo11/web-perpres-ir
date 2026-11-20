"""Core search engine for Indonesian Presidential Regulations (Perpres).

Implements the Stage 1 (BM25 + Dense RRF) and Stage 2 (QDA + Convex Fusion)
pipeline matching perpres_hybrid_qda_kaggle_v3.ipynb.
"""

from collections import defaultdict
import json
from pathlib import Path
import re
import time
from typing import Literal

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer


def tokenize(text: str) -> list[str]:
    """Lowercase word tokenization matching notebook preprocessing."""
    return re.findall(r"\w+", str(text).lower(), flags=re.UNICODE)


class QDA(nn.Module):
    """Query-Document Attention reranker (4 heads, 1.77M parameters)."""

    def __init__(self, dim: int = 384, heads: int = 4, dropout: float = 0.2) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.score = nn.Linear(dim, 1)

    def forward(
        self,
        q: torch.Tensor,
        qm: torch.Tensor,
        d: torch.Tensor,
        dm: torch.Tensor,
    ) -> torch.Tensor:
        x, _ = self.attn(q, d, d, key_padding_mask=~dm, need_weights=False)
        x = self.norm1(q + x)
        x = self.norm2(x + self.ffn(x))
        pooled = (x * qm.unsqueeze(-1)).sum(1) / qm.sum(1, keepdim=True).clamp_min(1)
        return self.score(pooled).squeeze(-1)


class PerpresSearchEngine:
    """Two-stage hybrid retrieval and QDA reranking engine."""

    def __init__(self, base_dir: Path | None = None) -> None:
        root = base_dir or Path(__file__).resolve().parent
        self.root = root
        self.artifacts_dir = root / "artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        corpus_candidates = [
            root / "data" / "dataset_korpus_per_ayat.csv",
            root / "dataset_korpus_per_ayat.csv",
            root.parent / "dataset" / "dataset_korpus_per_ayat.csv",
        ]
        self.corpus_path = next((p for p in corpus_candidates if p.is_file()), None)
        if self.corpus_path is None:
            raise FileNotFoundError(
                f"Corpus CSV not found in any expected location: {corpus_candidates}"
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config_path = self._resolve_file("config.json")
        cfg = json.loads(config_path.read_text(encoding="utf-8")) if config_path else {}

        self.model_name: str = cfg.get("model_name", "intfloat/multilingual-e5-small")
        self.top_k_each: int = int(cfg.get("top_k_each", 25))
        self.rrf_k: int = int(cfg.get("rrf_k", 60))
        self.max_q_len: int = int(cfg.get("max_q_len", 64))
        self.max_d_len: int = int(cfg.get("max_d_len", 256))
        self.dropout: float = float(cfg.get("dropout", 0.2))
        self.rerank_alpha: float = float(cfg.get("rerank_alpha", 0.05))

        self.corpus = pd.read_csv(self.corpus_path, sep=";")
        self.doc_ids = self.corpus["row_id"].astype(str).tolist()
        self.doc_texts = self.corpus["value"].astype(str).tolist()
        self.doc_index = {d: i for i, d in enumerate(self.doc_ids)}
        self.bm25 = BM25Okapi([tokenize(t) for t in self.doc_texts])

        local_model_dir = root / "models" / "multilingual-e5-small"
        if local_model_dir.is_dir() and (local_model_dir / "config.json").is_file():
            self.tokenizer = AutoTokenizer.from_pretrained(str(local_model_dir), local_files_only=True)
            self.encoder = AutoModel.from_pretrained(str(local_model_dir), local_files_only=True).to(self.device).eval()
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.encoder = AutoModel.from_pretrained(self.model_name).to(self.device).eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        self.doc_states, self.doc_masks, self.doc_dense = self._load_or_build_doc_artifacts()

        self.qda: QDA | None = None
        self.qda_loaded = False
        ckpt_path = self._resolve_file("qda_best.pt")
        if ckpt_path is not None:
            saved = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            model = QDA(dim=int(self.doc_states.shape[-1]), heads=4, dropout=self.dropout).to(self.device)
            model.load_state_dict(saved["state_dict"])
            model.eval()
            self.qda = model
            self.qda_loaded = True

    def _resolve_file(self, filename: str) -> Path | None:
        """Find artifact in artifacts/ or root directory."""
        for candidate in (self.artifacts_dir / filename, self.root / filename):
            if candidate.is_file():
                return candidate
        return None

    @torch.inference_mode()
    def _encode_texts(
        self, texts: list[str], max_length: int, prefix: str = "", batch_size: int = 16
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prefixed = [prefix + t for t in texts] if prefix else texts
        states, masks, pooled = [], [], []
        for start in range(0, len(prefixed), batch_size):
            batch = self.tokenizer(
                prefixed[start : start + batch_size],
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            mask = batch["attention_mask"]
            output = self.encoder(**{k: v.to(self.device) for k, v in batch.items()}).last_hidden_state
            m = mask.to(self.device).unsqueeze(-1)
            mean = (output * m).sum(1) / m.sum(1).clamp_min(1)
            states.append(output.cpu().to(torch.float16))
            masks.append(mask.bool())
            pooled.append(F.normalize(mean, dim=1).cpu())
        return torch.cat(states), torch.cat(masks), torch.cat(pooled)

    def _load_or_build_doc_artifacts(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        art_path = self._resolve_file("doc_artifacts.pt") or (self.artifacts_dir / "doc_artifacts.pt")
        if art_path.is_file():
            data = torch.load(art_path, map_location="cpu", weights_only=False)
            if data.get("doc_ids") == self.doc_ids and data.get("model_name") == self.model_name:
                return data["doc_states"], data["doc_masks"], data["doc_dense"]

        batch_size = 32 if self.device.type == "cuda" else 8
        doc_states, doc_masks, doc_dense = self._encode_texts(
            self.doc_texts, max_length=self.max_d_len, prefix="passage: ", batch_size=batch_size
        )
        torch.save(
            {
                "doc_states": doc_states,
                "doc_masks": doc_masks,
                "doc_dense": doc_dense,
                "doc_ids": self.doc_ids,
                "model_name": self.model_name,
                "max_d_len": self.max_d_len,
            },
            art_path,
        )
        return doc_states, doc_masks, doc_dense

    @torch.inference_mode()
    def search(
        self,
        query: str,
        top_k: int = 10,
        mode: Literal["hybrid_qda", "rrf"] = "hybrid_qda",
    ) -> dict:
        """Retrieve and optionally rerank Perpres clauses for a query."""
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        q_states, q_masks, q_dense = self._encode_texts(
            [query], max_length=self.max_q_len, prefix="query: ", batch_size=1
        )

        # Stage 1: BM25 + Dense RRF
        sparse_scores = self.bm25.get_scores(tokenize(query))
        dense_scores = (self.doc_dense @ q_dense[0]).numpy()
        sparse = np.argsort(-sparse_scores, kind="stable")[: self.top_k_each]
        dense = np.argsort(-dense_scores, kind="stable")[: self.top_k_each]

        fused: dict[int, float] = defaultdict(float)
        for ranking in (sparse, dense):
            for rank, di in enumerate(ranking, start=1):
                fused[int(di)] += 1.0 / (self.rrf_k + rank)

        sorted_cands = sorted(fused.items(), key=lambda x: (-x[1], self.doc_ids[x[0]]))[:20]
        cand_indices = [di for di, _ in sorted_cands]
        rrf_scores = np.array([score for _, score in sorted_cands], dtype=float)

        # Stage 2: QDA + Convex Score Fusion
        effective_mode = mode
        if mode == "hybrid_qda" and self.qda is not None and self.rerank_alpha > 0.0:
            di_tensor = torch.as_tensor(cand_indices, dtype=torch.long)
            d = self.doc_states.index_select(0, di_tensor).to(self.device, dtype=torch.float32)
            dm = self.doc_masks.index_select(0, di_tensor).to(self.device)
            q = q_states.expand(len(cand_indices), -1, -1).to(self.device, dtype=torch.float32)
            qm = q_masks.expand(len(cand_indices), -1).to(self.device)
            qda_scores = self.qda(q, qm, d, dm).cpu().numpy().astype(float)

            qda_std = qda_scores.std()
            qda_norm = (qda_scores - qda_scores.mean()) / (qda_std + 1e-8) if qda_std > 0 else np.zeros_like(qda_scores)
            rrf_std = rrf_scores.std()
            rrf_norm = (rrf_scores - rrf_scores.mean()) / (rrf_std + 1e-8) if rrf_std > 0 else np.zeros_like(rrf_scores)
            final_scores = self.rerank_alpha * qda_norm + (1.0 - self.rerank_alpha) * rrf_norm
        else:
            if mode == "hybrid_qda" and self.qda is None:
                effective_mode = "rrf_fallback"
            final_scores = rrf_scores

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        latency_ms = round((time.perf_counter() - start) * 1000.0, 2)

        order = sorted(
            zip(cand_indices, final_scores.tolist()),
            key=lambda z: (-z[1], self.doc_ids[z[0]]),
        )[:top_k]

        results = []
        for rank, (di, score) in enumerate(order, start=1):
            row = self.corpus.iloc[di]
            results.append(
                {
                    "rank": rank,
                    "doc_id": self.doc_ids[di],
                    "nomor_peraturan": str(getattr(row, "nomor_peraturan", "N/A")),
                    "sub_segmen": str(getattr(row, "sub_segmen", "N/A")),
                    "text": self.doc_texts[di],
                    "score": round(float(score), 4),
                }
            )

        return {
            "query": query,
            "mode": effective_mode,
            "qda_loaded": self.qda_loaded,
            "rerank_alpha": self.rerank_alpha if effective_mode == "hybrid_qda" else 0.0,
            "latency_ms": latency_ms,
            "results": results,
        }
