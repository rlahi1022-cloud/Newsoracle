# services/ensemble.py
from typing import Dict, Any, List
from logger import get_logger
from config import (RULE_WEIGHT, SEMANTIC_WEIGHT, CLASSIFIER_WEIGHT, AGENCY_WEIGHT,
    OFFICIAL_THRESHOLD, INTERNAL_RELIABILITY_WEIGHT, EXTERNAL_RELIABILITY_WEIGHT)
from services.reliability_scorer import compute_internal_reliability

logger = get_logger(__name__)

def _safe_get(d, key, default=0.0):
    try:
        v = d.get(key, default)
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default

def compute_official_score(scores):
    rule=_safe_get(scores,"rule_score"); sem=_safe_get(scores,"semantic_score")
    clf=_safe_get(scores,"classifier_score"); agn=_safe_get(scores,"agency_score")
    total = rule*RULE_WEIGHT + sem*SEMANTIC_WEIGHT + clf*CLASSIFIER_WEIGHT + agn*AGENCY_WEIGHT
    return round(max(0.0, min(1.0, total)), 4)

def compute_external_reliability(cross_info):
    cs = int(cross_info.get("cluster_size",1) or 1)
    us = int(cross_info.get("unique_sources",1) or 1)
    hod = bool(cross_info.get("has_official_domain", False))
    avg = float(cross_info.get("avg_similarity",0.0) or 0.0)
    doms = cross_info.get("official_domains",[]) or []
    cross_score = min(cs/10.0, 1.0)
    source_score = min(us/8.0, 1.0)
    domain_score = 1.0 if hod else 0.0
    consistency_score = avg if cs > 1 else 0.5
    ext = cross_score*0.35 + source_score*0.25 + domain_score*0.20 + consistency_score*0.20
    ext = round(max(0.0, min(1.0, ext)), 4)
    return {
        "external_reliability": ext,
        "cross_reporting": {"score": round(cross_score,4), "cluster_size": cs, "unique_sources": us},
        "official_domain": {"score": round(domain_score,4), "has_official_domain": hod, "domains": doms[:5]},
        "content_consistency": {"score": round(consistency_score,4), "avg_similarity": round(avg,4), "is_single_report": cs<=1},
    }

def compute_reliability(title, content, originallink, cross_info):
    ib = compute_internal_reliability(title, content, originallink)
    eb = compute_external_reliability(cross_info)
    i = float(ib.get("internal_reliability",0.0))
    e = float(eb.get("external_reliability",0.0))
    final = i*INTERNAL_RELIABILITY_WEIGHT + e*EXTERNAL_RELIABILITY_WEIGHT
    final = round(max(0.0, min(1.0, final)), 4)
    return {
        "reliability_score": final,
        "internal_reliability": round(i,4),
        "external_reliability": round(e,4),
        "weights": {"internal": INTERNAL_RELIABILITY_WEIGHT, "external": EXTERNAL_RELIABILITY_WEIGHT},
        "reliability_breakdown": {
            "source_accountability": ib["source_accountability"],
            "verifiability": ib["verifiability"],
            "neutrality": ib["neutrality"],
            "cross_reporting": eb["cross_reporting"],
            "official_domain": eb["official_domain"],
            "content_consistency": eb["content_consistency"],
        },
    }

def build_final_result(article, scores, cross_info):
    try:
        off = compute_official_score(scores)
        rel = compute_reliability(article.get("title",""), article.get("content",""),
            article.get("originallink",""), cross_info or {})
        label = 1 if off >= OFFICIAL_THRESHOLD else 0
        result = {
            "title": article.get("title",""), "source": article.get("source",""),
            "originallink": article.get("originallink",""),
            "rule_score": round(_safe_get(scores,"rule_score"),4),
            "semantic_score": round(_safe_get(scores,"semantic_score"),4),
            "classifier_score": round(_safe_get(scores,"classifier_score"),4),
            "agency_score": round(_safe_get(scores,"agency_score"),4),
            "official_score": off, "predicted_label": label,
        }
        result.update(rel)
        return result
    except Exception as ex:
        logger.exception(f"build_final_result failed: {ex}")
        return {"title": article.get("title",""), "source": article.get("source",""),
            "originallink": article.get("originallink",""), "official_score": 0.0,
            "reliability_score": 0.0, "predicted_label": 0, "reliability_breakdown": {}}