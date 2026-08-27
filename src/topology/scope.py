"""Topology scope for primary ranking and enforcement analyses.

These analyses use five named topologies. Mesh remains available for breadth
and stress tests but is excluded from primary denominators.
"""

PRIMARY_TOPOLOGIES = ("sequential", "centralized", "decentralized",
                      "hierarchical", "hybrid")

# Kept explicit so primary aggregates can reject stray mesh rows.
EXCLUDED_FROM_PRIMARY = ("mesh",)

# Breadth/stress-test scope only.
PILOT_TOPOLOGIES = ("mesh",)


def is_primary(topology):
    return topology in PRIMARY_TOPOLOGIES


def assert_primary_scope(topologies, where="primary aggregate"):
    """Refuse to let an excluded topology enter a primary denominator."""
    bad = sorted({t for t in topologies if t in EXCLUDED_FROM_PRIMARY})
    if bad:
        raise AssertionError(
            "%s would include excluded topolog%s %s. Mesh is removed from "
            "primary denominators; filter with PRIMARY_TOPOLOGIES before "
            "aggregating." % (where, "y" if len(bad) == 1 else "ies",
                              ", ".join(bad)))
    return True


def filter_primary(rows, key=lambda r: r.get("topology")):
    return [r for r in rows if key(r) in PRIMARY_TOPOLOGIES]
