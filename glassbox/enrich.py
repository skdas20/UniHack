"""The optional hosted-model layer: proposals, never assertions.

This module is the *only* place a language model touches the output, and it is
deliberately hemmed in:

* **It never runs by default.** The core pipeline is complete without it. If
  ``NVIDIA_API_KEY`` is absent, :func:`build_proposer` returns ``None`` and the
  run is unaffected -- which is what makes the prototype safe to hand to a
  judge on a machine with no secrets.
* **It only fills blanks.** Anything the rule engine extracted from the input
  wins. The model is never allowed to overwrite a value that has a character
  span behind it.
* **Its output is validated before acceptance.** A proposed attribute value
  that is not in the slot's controlled vocabulary is rejected, not coerced.
  A proposed measurement that fails its slot's plausibility bounds is rejected.
* **It is labelled.** Every accepted value carries ``Source.LLM``, which the
  confidence engine weights at 0.50 against 0.95 for a regex extraction, and
  which routes the row to human review. The provenance sidecar names the model
  and the prompt that produced it.

The reason for that asymmetry is the finding in docs/DERIVED_RULES.md: the gold
rows' ``120 V``, ``15 A`` and ``47 dBA`` are nowhere in their inputs. They came
from the manufacturer's site. A model that knows the Frigidaire PDSH4816AF is a
120 V appliance is genuinely useful for closing that gap -- and genuinely
dangerous if its guesses are indistinguishable from facts. So we take the
usefulness and keep the distinction.

Provider: NVIDIA NIM, which is OpenAI-compatible, so one adapter covers it and
any other compatible endpoint via ``GLASSBOX_LLM_BASE_URL``.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from .attributes import Contract, Kind, Slot
from .provenance import Cell, Provenance, Source

DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "meta/llama-3.3-70b-instruct"

#: Headers the proposer is allowed to write. Anything outside this set is
#: dropped even if the model returns it -- a model cannot invent a part number
#: or a price into the delivery sheet through this path.
PROPOSABLE = frozenset(
    {
        "MARKETING_DESCRIPTION",
        "Product Name",
        "TRADE_NAME",
        "Application",
        "Includes",
        "With",
        "Warranty",
    }
)

#: Cap on how many rows the layer will call out for in one run, so an
#: accidental full-catalogue run cannot burn a quota.
DEFAULT_MAX_CALLS = 250


@dataclass
class ProposerConfig:
    api_key: str = ""
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    temperature: float = 0.0
    timeout_s: float = 30.0
    max_calls: int = DEFAULT_MAX_CALLS
    #: Only call out for rows whose confidence is below this. Well-resolved
    #: rows do not need help and calling for them wastes quota.
    call_below_confidence: float = 1.01
    cache_path: Path | None = None

    @classmethod
    def from_env(cls) -> "ProposerConfig":
        return cls(
            api_key=os.environ.get("NVIDIA_API_KEY", "").strip()
            or os.environ.get("GLASSBOX_LLM_API_KEY", "").strip(),
            base_url=os.environ.get("GLASSBOX_LLM_BASE_URL", DEFAULT_BASE_URL).strip(),
            model=os.environ.get("GLASSBOX_LLM_MODEL", DEFAULT_MODEL).strip(),
            max_calls=int(os.environ.get("GLASSBOX_LLM_MAX_CALLS", DEFAULT_MAX_CALLS)),
            cache_path=Path(
                os.environ.get("GLASSBOX_LLM_CACHE", "outputs/llm_cache.jsonl")
            ),
        )


@dataclass
class ProposerStats:
    calls: int = 0
    cache_hits: int = 0
    accepted: int = 0
    rejected_not_in_lov: int = 0
    rejected_out_of_bounds: int = 0
    rejected_not_proposable: int = 0
    errors: int = 0
    latency_s: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "calls": self.calls,
            "cache_hits": self.cache_hits,
            "values_accepted": self.accepted,
            "rejected_not_in_lov": self.rejected_not_in_lov,
            "rejected_out_of_bounds": self.rejected_out_of_bounds,
            "rejected_not_proposable": self.rejected_not_proposable,
            "errors": self.errors,
            "mean_latency_s": round(self.latency_s / max(self.calls, 1), 3),
        }


SYSTEM_PROMPT = """You are a product data specialist for an industrial \
distributor's catalogue. You are given a manufacturer part number, a brand, a \
product category, and a terse distributor description.

Your job is to fill ONLY the fields listed in the request, using what you \
actually know about that specific part number or product line.

Absolute rules:
1. If you do not know a value for a specific field, return null for it. A null \
is a correct answer. A plausible guess is a wrong answer.
2. For fields with an "allowed" list, return one of those exact strings or null. \
Never return a value outside the list, and never invent a new one.
3. Never return a part number, SKU, price, UPC, EAN, GTIN or country of origin.
4. Units must be written as a bare number in "value" with the unit in "uom", \
using the approved abbreviation given in the request.
5. Return strict JSON only, with no prose, no markdown fences and no commentary.
"""


def _build_request(context: Any, blanks: list[Slot]) -> dict[str, Any]:
    """Describe the row and the blank slots, with each slot's constraints."""
    fields = []
    for slot in blanks:
        entry: dict[str, Any] = {"name": slot.label}
        if slot.lov:
            entry["allowed"] = list(slot.lov)
        if slot.kind is Kind.MEASURE:
            entry["kind"] = "measurement"
            if slot.family:
                entry["measurement_family"] = slot.family
            if slot.lo is not None or slot.hi is not None:
                entry["plausible_range"] = [slot.lo, slot.hi]
        elif slot.kind is Kind.COUNT:
            entry["kind"] = "integer count"
        else:
            entry["kind"] = "text"
        fields.append(entry)

    category = (
        context.classification.category.classpath
        if context.classification.ok
        else "unknown"
    )
    return {
        "manufacturer_part_number": context.row.mpn,
        "brand": context.brand.brand_plain,
        "manufacturer": context.brand.manufacturer,
        "category": category,
        "product_name": context.product_name,
        "distributor_description": context.body,
        "fields_to_fill": fields,
        "response_format": {
            "attributes": {
                "<field name>": {"value": "<string or null>", "uom": "<string or null>"}
            },
            "marketing_description": "<one or two sentences, or null>",
        },
    }


def _parse_json(text: str) -> dict[str, Any]:
    """Tolerate a fenced or chatty response without trusting its content."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return {}
    return {}


class Proposer:
    """Fills blank contract slots from a hosted model, under validation."""

    def __init__(self, config: ProposerConfig, client: Any) -> None:
        self.config = config
        self.client = client
        self.stats = ProposerStats()
        self._cache: dict[str, dict[str, Any]] = {}
        self._load_cache()

    # ---------- cache ----------

    def _load_cache(self) -> None:
        path = self.config.cache_path
        if not path or not path.exists():
            return
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                self._cache[record["key"]] = record["response"]
        except Exception:
            self._cache = {}

    def _save_cache_entry(self, key: str, response: dict[str, Any]) -> None:
        path = self.config.cache_path
        if not path:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"key": key, "response": response}) + "\n")
        except Exception:
            pass

    # ---------- the call ----------

    def _ask(self, payload: dict[str, Any]) -> dict[str, Any]:
        key = json.dumps(payload, sort_keys=True)
        if key in self._cache:
            self.stats.cache_hits += 1
            return self._cache[key]
        if self.stats.calls >= self.config.max_calls:
            return {}

        started = time.perf_counter()
        try:
            completion = self.client.chat.completions.create(
                model=self.config.model,
                temperature=self.config.temperature,
                timeout=self.config.timeout_s,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
            )
            self.stats.calls += 1
            self.stats.latency_s += time.perf_counter() - started
            content = (completion.choices[0].message.content or "").strip()
        except Exception:
            self.stats.errors += 1
            return {}

        parsed = _parse_json(content)
        self._cache[key] = parsed
        self._save_cache_entry(key, parsed)
        return parsed

    # ---------- validation ----------

    def _accept_attribute(
        self, slot: Slot, value: str, uom: str
    ) -> tuple[str, str] | None:
        """Validate one proposed attribute value against the slot's contract."""
        value = (value or "").strip()
        if not value or value.lower() in {"null", "none", "unknown", "n/a"}:
            return None

        if slot.lov and value not in slot.lov:
            self.stats.rejected_not_in_lov += 1
            return None

        if slot.kind in {Kind.MEASURE, Kind.COUNT}:
            from . import units as U

            numeric = U.to_decimal(value)
            if numeric is None:
                self.stats.rejected_out_of_bounds += 1
                return None
            if slot.lo is not None and numeric < slot.lo:
                self.stats.rejected_out_of_bounds += 1
                return None
            if slot.hi is not None and numeric > slot.hi:
                self.stats.rejected_out_of_bounds += 1
                return None
            if slot.kind is Kind.MEASURE and slot.family:
                canonical = U.canonicalise(uom or "")
                unit = U.lookup(canonical)
                if unit is None or unit.family != slot.family:
                    self.stats.rejected_out_of_bounds += 1
                    return None
                uom = unit.canonical
        return value, (uom or "").strip()

    # ---------- the proposer interface ----------

    def __call__(self, context: Any) -> dict[str, Cell]:
        blanks = [
            f.slot
            for f in context.filled.values()
            if not f and f.slot.kind not in {Kind.ADDITIONAL, Kind.MODEL}
        ]
        if not blanks:
            return {}

        payload = _build_request(context, blanks)
        response = self._ask(payload)
        if not response:
            return {}

        out: dict[str, Cell] = {}
        by_label = {s.label: s for s in blanks}
        proposed = response.get("attributes") or {}
        if isinstance(proposed, dict):
            for label, entry in proposed.items():
                slot = by_label.get(label)
                if slot is None:
                    continue
                if isinstance(entry, dict):
                    value, uom = entry.get("value"), entry.get("uom")
                else:
                    value, uom = entry, ""
                accepted = self._accept_attribute(slot, str(value or ""), str(uom or ""))
                if accepted is None:
                    continue
                value, uom = accepted
                self.stats.accepted += 1
                # Write back into the filled map so the description channels
                # can use it -- still tagged as a proposal.
                from .extract import Filled

                cell = Cell(
                    value,
                    Provenance(
                        Source.LLM,
                        rule=f"proposed:{slot.label}",
                        confidence=0.5,
                        detail=(
                            f"proposed by {self.config.model} from the part number, "
                            f"validated against this slot's constraints; NOT read "
                            f"from the input text"
                        ),
                    ),
                )
                context.filled[slot.label] = Filled(slot, value, uom, cell)

        marketing = response.get("marketing_description")
        if isinstance(marketing, str) and marketing.strip():
            out["MARKETING_DESCRIPTION"] = Cell(
                marketing.strip(),
                Provenance(
                    Source.LLM,
                    rule="proposed:MARKETING_DESCRIPTION",
                    confidence=0.5,
                    detail=f"generated by {self.config.model}; not sourced from the input",
                ),
            )
            self.stats.accepted += 1

        return {k: v for k, v in out.items() if k in PROPOSABLE}


def build_proposer(config: ProposerConfig | None = None) -> Proposer | None:
    """Construct the proposer, or return None when it cannot run.

    Returning ``None`` rather than raising is the point: a missing key must
    degrade the run to the deterministic core, not break it.
    """
    config = config or ProposerConfig.from_env()
    if not config.api_key:
        return None
    try:
        from openai import OpenAI
    except ImportError:
        return None
    try:
        client = OpenAI(api_key=config.api_key, base_url=config.base_url)
    except Exception:
        return None
    return Proposer(config, client)


def probe(config: ProposerConfig | None = None) -> dict[str, Any]:
    """One tiny call, to verify credentials and model availability."""
    config = config or ProposerConfig.from_env()
    if not config.api_key:
        return {"ok": False, "reason": "no API key in NVIDIA_API_KEY"}
    try:
        from openai import OpenAI

        client = OpenAI(api_key=config.api_key, base_url=config.base_url)
        started = time.perf_counter()
        completion = client.chat.completions.create(
            model=config.model,
            temperature=0.0,
            timeout=config.timeout_s,
            messages=[
                {"role": "system", "content": "Reply with the single word: ready"},
                {"role": "user", "content": "ping"},
            ],
        )
        return {
            "ok": True,
            "model": config.model,
            "base_url": config.base_url,
            "latency_s": round(time.perf_counter() - started, 3),
            "reply": (completion.choices[0].message.content or "").strip()[:40],
        }
    except Exception as exc:
        return {"ok": False, "reason": repr(exc), "model": config.model}
