'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Build the structured ``hva_res_dict`` consumed by the Run Deck status_matrix
builder from the report.json each hva run prints.

One group per run (``hip``, ``verbs``), one node record per node. A node fails
when any test fails the gate or no usable report came back. Its drill-down
lists every test that did not pass; passing tests are only counted, since a
node runs dozens of them. A known failure that did not fail the gate shows as
``na`` with its ticket so it stays visible without turning the cell red.

Shape (consumed by cvs/lib/report/rundeck/dataset_builders/status_matrix.py)::

    {
      "_meta":  {"cluster": ..., "version_label": "hva", "suite": ...},
      "groups": {"<run>": {"nodes": {"<label>": <node record>}}},
    }
'''

from cvs.lib import hva_lib

_STATUSES = ("pass", "fail", "na")


def make_meta(cluster_dict, suite_name):
    '''Assemble the _meta block for the deck run card.'''
    cluster = (cluster_dict or {}).get("cluster_name") or (cluster_dict or {}).get("name") or "—"
    return {
        "cluster": cluster,
        "version": "—",
        "version_label": "hva",
        "suite": suite_name or "hva_cvs",
        "generated_at": "",
    }


def build_node_record(status, items=None, items_summary=""):
    '''Assemble one node record for a run.'''
    normalized = str(status or "na").lower()
    if normalized not in _STATUSES:
        normalized = "na"
    return {
        "status": normalized,
        "items_summary": items_summary or "",
        "items": items or [],
        "errors_json_href": "",
        "log_tarball_href": "",
    }


def record_group(res_dict, group, node_records, meta=None):
    '''Merge one run's per-node records. A re-run of the same group overwrites.'''
    if meta and not res_dict.get("_meta"):
        res_dict["_meta"] = dict(meta)
    groups = res_dict.setdefault("groups", {})
    group_entry = groups.setdefault(str(group), {"nodes": {}})
    group_entry["nodes"].update(node_records)
    return res_dict


def _item(test):
    status = test.get("status")
    if test.get("gate_fail"):
        item_status = "fail"
    elif status == "pass":
        item_status = "pass"
    else:
        item_status = "na"
    return {"name": str(test.get("id", "?")), "status": item_status, "message": hva_lib.describe_test(test)}


def summarize_counts(counts, known_count=0):
    '''``58 pass, 5 fail (5 known), 4 skip, 5 na`` from report.json summary counts.'''
    failed = counts.get("failed", 0)
    fail_text = f"{failed} fail"
    if known_count:
        fail_text += f" ({known_count} known)"
    return (
        f"{counts.get('passed', 0)} pass, {fail_text}, "
        f"{counts.get('skipped', 0)} skip, {counts.get('not_applicable', 0)} na"
    )


def node_record_from_run(parsed, peer=None):
    '''Node record for one parsed hva run (see hva_lib.parse_run_output).

    Args:
      parsed: parse_run_output result.
      peer: the other node of a two-node Verbs pair, named in the summary.
    '''
    result = hva_lib.verdict(parsed)
    prefix = f"with {peer}: " if peer else ""
    if parsed.get("report") is None:
        message = result["error"]
        return build_node_record(
            "fail", [{"name": "report.json", "status": "fail", "message": message}], prefix + message
        )

    tests = parsed["report"].get("tests") or []
    items = [_item(t) for t in tests if t.get("status") != "pass" or t.get("gate_fail")]
    summary = prefix + summarize_counts(result["counts"], len(result["known_failures"]))
    if result["error"]:
        items.append({"name": "hva", "status": "fail", "message": result["error"]})
        summary = f"{summary}; {result['error']}"
    if not result["passed"]:
        status = "fail"
    elif any(t.get("status") == "pass" for t in tests):
        status = "pass"
    else:
        status = "na"
    return build_node_record(status, items, summary)


def record_run(res_dict, group, node_runs, meta=None):
    '''Record one hva run across nodes.

    Args:
      node_runs: {node label: (parsed run, peer or None)}.
    '''
    node_records = {}
    version = ""
    for node, (parsed, peer) in (node_runs or {}).items():
        node_records[str(node)] = node_record_from_run(parsed, peer)
        if not version:
            version = hva_lib.verdict(parsed)["version"]
    recorded = record_group(res_dict, group, node_records, meta=meta)
    run_meta = recorded.get("_meta")
    if version and isinstance(run_meta, dict) and run_meta.get("version") in (None, "", "—"):
        run_meta["version"] = version
    return recorded
