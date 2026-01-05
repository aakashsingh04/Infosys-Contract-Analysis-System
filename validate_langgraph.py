"""Validation script for Planning Module, LangGraph coordination, and Milestone 3 features."""
import json
import tempfile
from typing import List
from planning_module import PlanningModule
from agents import AgentOrchestrator
from contract_analyzer import ContractAnalyzer


def validate_planning(contract_text: str) -> dict:
    """Validate PlanningModule plan generation."""
    planner = PlanningModule(use_free_model=True)
    plan = planner.generate_agent_plan(contract_text, metadata={"file_name": "dummy.txt", "file_type": "txt"})
    assert "domain" in plan
    assert "agents" in plan and all(k in plan["agents"] for k in ["compliance", "finance", "legal", "operations"])
    assert isinstance(plan.get("analysis_sequence", []), list)
    return plan


def validate_langgraph(contract_text: str, plan: dict, roles: List[str]) -> dict:
    """Validate LangGraph orchestration across agents."""
    orchestrator = AgentOrchestrator(use_free_model=True)
    results = orchestrator.analyze_contract(contract_text, planning_info=plan, agent_roles=roles)
    assert "analyses" in results
    return results


def main():
    contract_text = (
        "This Software Services Agreement outlines API access, service levels (SLAs), "
        "payment terms with milestone-based billing, data protection obligations (GDPR), "
        "and operational delivery timelines."
    )
    print("== Validating Planning Module ==")
    plan = validate_planning(contract_text)
    print(json.dumps({
        "domain": plan.get("domain"),
        "sequence": plan.get("analysis_sequence"),
        "primary_agents": plan.get("coordination_strategy", {}).get("primary_agents", [])
    }, indent=2))

    roles = list(plan.get("agents", {}).keys())
    print("\n== Validating LangGraph Coordination ==")
    results = validate_langgraph(contract_text, plan, roles)
    analyses = results.get("analyses", {})
    summary = {k: (v.get("role", k)) for k, v in analyses.items()}
    print(json.dumps({"agents_executed": summary, "completed_agents": results.get("completed_agents", [])}, indent=2))

    print("\n== Milestone 3: Pipelines, Parallel Extraction, Multi-turn, Storage ==")
    analyzer = ContractAnalyzer(use_free_model=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8") as tmpf:
        tmpf.write(contract_text)
        tmp_path = tmpf.name
    doc_id = analyzer.upload_document(tmp_path)
    extracted = analyzer.extract_clauses_parallel(doc_id, domains=["compliance", "finance"], k=3)
    print(json.dumps({"extracted_counts": {k: len(v) for k, v in extracted.items()}}, indent=2))
    comp_pipe = analyzer.compliance_risk_pipeline(doc_id, k=3)
    fin_pipe = analyzer.financial_risk_pipeline(doc_id, k=3)
    print(json.dumps({
        "compliance_pipeline": {"clauses": len(comp_pipe["clauses"]), "analysis_keys": list(comp_pipe["analyses"].keys())},
        "financial_pipeline": {"clauses": len(fin_pipe["clauses"]), "analysis_keys": list(fin_pipe["analyses"].keys())}
    }, indent=2))
    multi = analyzer.simulate_multi_turn(doc_id)
    print(json.dumps({
        "multi_turn": {
            "round1_compliance_len": len(multi["round1_compliance"]["analysis"]),
            "round1_finance_len": len(multi["round1_finance"]["analysis"]),
            "round2_compliance_len": len(multi["round2_compliance"]["analysis"])
        }
    }, indent=2))
    print("\n== Validation Complete ==")

if __name__ == "__main__":
    main()

