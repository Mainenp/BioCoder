from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from typing import Any

import httpx
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field, field_validator

from app.config import Settings
from app.rag.store import KnowledgeStore
from biocoder.security.permissions import ToolPermission
from biocoder.security.validation import validate_tool_text
from biocoder.tools.registry import ToolRegistry
from biocoder.tools.schema import RetryPolicy, ToolMetadata


def _compact_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


class LocalKnowledgeInput(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=10)

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        return validate_tool_text(value)


class WebSearchInput(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    max_results: int = Field(default=5, ge=1, le=10)

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        return validate_tool_text(value)


class ClinicalTrialsInput(BaseModel):
    condition_or_drug: str = Field(min_length=1, max_length=2000)
    max_results: int = Field(default=5, ge=1, le=10)

    @field_validator("condition_or_drug")
    @classmethod
    def validate_query(cls, value: str) -> str:
        return validate_tool_text(value)


def build_research_tool_registry(store: KnowledgeStore, settings: Settings) -> ToolRegistry:
    @tool(args_schema=LocalKnowledgeInput)
    def search_local_knowledge(query: str, top_k: int = 5) -> str:
        """Search the private local pharmaceutical knowledge base. Use for internal documents, uploaded papers, mechanisms, trial summaries, or drug notes."""
        try:
            return _compact_json({"results": store.search(query, top_k)})
        except Exception as exc:  # tool errors should be visible to the agent
            return _compact_json({"error": f"Local retrieval failed: {exc}", "results": []})

    @tool(args_schema=WebSearchInput)
    def search_pubmed(query: str, max_results: int = 5) -> str:
        """Search PubMed biomedical literature and return article titles, abstracts, dates, PMIDs, and citation URLs."""
        if not settings.enable_web_tools:
            return _compact_json({"error": "Web tools are disabled", "results": []})
        count = max(1, min(max_results, 10))
        common = {"tool": "bioagent", "email": settings.ncbi_email}
        if settings.ncbi_api_key:
            common["api_key"] = settings.ncbi_api_key
        try:
            with httpx.Client(timeout=settings.request_timeout_seconds) as client:
                search = client.get(
                    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                    params={"db": "pubmed", "term": query, "retmode": "json", "retmax": count, **common},
                )
                search.raise_for_status()
                ids = search.json().get("esearchresult", {}).get("idlist", [])
                if not ids:
                    return _compact_json({"results": []})
                details = client.get(
                    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                    params={"db": "pubmed", "id": ",".join(ids), "retmode": "xml", **common},
                )
                details.raise_for_status()
            root = ET.fromstring(details.content)
            results = []
            for record in root.findall(".//PubmedArticle"):
                pmid = record.findtext(".//PMID", default="")
                article = record.find(".//Article")
                if article is None:
                    continue
                title_node = article.find("ArticleTitle")
                title = "".join(title_node.itertext()) if title_node is not None else f"PubMed {pmid}"
                abstract_parts = []
                for node in article.findall(".//AbstractText"):
                    text = "".join(node.itertext()).strip()
                    label = node.attrib.get("Label")
                    abstract_parts.append(f"{label}: {text}" if label and text else text)
                authors = []
                for author in article.findall(".//Author")[:5]:
                    name = " ".join(
                        part for part in (author.findtext("ForeName"), author.findtext("LastName")) if part
                    )
                    if name:
                        authors.append(name)
                journal = article.findtext(".//Journal/Title", default="")
                year = article.findtext(".//PubDate/Year") or article.findtext(".//PubDate/MedlineDate", default="")
                results.append(
                    {
                        "title": title,
                        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                        "source_type": "pubmed",
                        "snippet": " ".join(abstract_parts)[:1600] or f"{year} · {journal}",
                        "metadata": {"pmid": pmid, "authors": authors, "journal": journal, "date": year},
                    }
                )
            return _compact_json({"results": results})
        except Exception as exc:
            return _compact_json({"error": f"PubMed request failed: {exc}", "results": []})

    @tool(args_schema=WebSearchInput)
    def search_openfda_drugs(query: str, max_results: int = 5) -> str:
        """Search official openFDA drug labels for indications, warnings, adverse reactions, dosage, and pharmacology."""
        if not settings.enable_web_tools:
            return _compact_json({"error": "Web tools are disabled", "results": []})
        count = max(1, min(max_results, 10))
        escaped = query.replace('"', "")
        try:
            with httpx.Client(timeout=settings.request_timeout_seconds) as client:
                response = client.get(
                    "https://api.fda.gov/drug/label.json",
                    params={"search": f'openfda.generic_name:"{escaped}"', "limit": count},
                )
                response.raise_for_status()
                rows = response.json().get("results", [])
            results = []
            for row in rows:
                fda = row.get("openfda", {})
                generic = (fda.get("generic_name") or [query])[0]
                set_id = row.get("set_id")
                results.append(
                    {
                        "title": generic,
                        "url": "https://open.fda.gov/apis/drug/label/" if not set_id else f"https://dailymed.nlm.nih.gov/dailymed/search.cfm?query={set_id}",
                        "source_type": "openfda",
                        "snippet": " ".join((row.get("indications_and_usage") or row.get("description") or [""])[:1])[:1000],
                        "metadata": {
                            "brand_names": fda.get("brand_name", []),
                            "manufacturer": fda.get("manufacturer_name", []),
                            "warnings": (row.get("warnings") or [""])[0][:600],
                        },
                    }
                )
            return _compact_json({"results": results})
        except Exception as exc:
            return _compact_json({"error": f"openFDA request failed: {exc}", "results": []})

    @tool(args_schema=ClinicalTrialsInput)
    def search_clinical_trials(condition_or_drug: str, max_results: int = 5) -> str:
        """Search ClinicalTrials.gov for current and completed studies by condition, intervention, sponsor, or drug name."""
        if not settings.enable_web_tools:
            return _compact_json({"error": "Web tools are disabled", "results": []})
        count = max(1, min(max_results, 10))
        try:
            with httpx.Client(timeout=settings.request_timeout_seconds) as client:
                response = client.get(
                    "https://clinicaltrials.gov/api/v2/studies",
                    params={"query.term": condition_or_drug, "pageSize": count, "format": "json"},
                )
                response.raise_for_status()
                studies = response.json().get("studies", [])
            results = []
            for study in studies:
                protocol = study.get("protocolSection", {})
                ident = protocol.get("identificationModule", {})
                description = protocol.get("descriptionModule", {})
                status = protocol.get("statusModule", {})
                design = protocol.get("designModule", {})
                nct = ident.get("nctId", "")
                results.append(
                    {
                        "title": ident.get("briefTitle", nct),
                        "url": f"https://clinicaltrials.gov/study/{nct}",
                        "source_type": "clinical_trial",
                        "snippet": description.get("briefSummary", "")[:1000],
                        "metadata": {
                            "nct_id": nct,
                            "status": status.get("overallStatus"),
                            "phases": design.get("phases", []),
                            "enrollment": design.get("enrollmentInfo", {}).get("count"),
                        },
                    }
                )
            return _compact_json({"results": results})
        except Exception as exc:
            return _compact_json({"error": f"ClinicalTrials.gov request failed: {exc}", "results": []})

    registry = ToolRegistry()
    retry = RetryPolicy(max_attempts=2, backoff_seconds=0.25, retryable_errors=["timeout", "429", "5xx"])
    for candidate in [search_local_knowledge, search_pubmed, search_openfda_drugs, search_clinical_trials]:
        registry.register(
            candidate,
            ToolMetadata(
                name=candidate.name,
                description=candidate.description,
                parameters=candidate.args_schema.model_json_schema() if candidate.args_schema else candidate.args,
                permission=ToolPermission.READ_ONLY,
                timeout_seconds=settings.request_timeout_seconds,
                retry_policy=retry,
                side_effect=False,
            ),
        )
    return registry


def build_research_tools(store: KnowledgeStore, settings: Settings) -> list[BaseTool]:
    return build_research_tool_registry(store, settings).tools()


def tool_names(tools: list[BaseTool]) -> list[str]:
    return [candidate.name for candidate in tools]
