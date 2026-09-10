from __future__ import annotations

from collections import defaultdict, deque

from .models import AuditStatus, ClaimStatus, ResearchState


class ProofGraph:
    def __init__(self, state: ResearchState):
        self.state = state
        self.parents: dict[str, set[str]] = defaultdict(set)
        self.children: dict[str, set[str]] = defaultdict(set)
        for edge in state.dependency_edges:
            if edge.relation != "requires":
                continue
            self.parents[edge.child_claim_id].add(edge.parent_claim_id)
            self.children[edge.parent_claim_id].add(edge.child_claim_id)

    def ancestors(self, claim_id: str) -> set[str]:
        seen: set[str] = set()
        stack = list(self.parents.get(claim_id, ()))
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            stack.extend(self.parents.get(node, ()))
        return seen

    def descendants(self, claim_id: str) -> set[str]:
        seen: set[str] = set()
        stack = list(self.children.get(claim_id, ()))
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            stack.extend(self.children.get(node, ()))
        return seen

    def candidate_roots(self) -> list[str]:
        ids = {c.id for c in self.state.claims}
        parent_ids = set(self.children)
        roots = ids - parent_ids
        return sorted(
            roots,
            key=lambda cid: (
                self.state.claim(cid).claim_type == "theorem",
                self.state.proof_completeness(cid),
                len(self.ancestors(cid)),
            ),
            reverse=True,
        )

    def critical_unresolved(self, root_claim_id: str) -> list[tuple[str, int]]:
        unresolved = set(self.state.unresolved_dependencies(root_claim_id))
        impact: list[tuple[str, int]] = []
        relevant = self.ancestors(root_claim_id) | {root_claim_id}
        for cid in unresolved:
            affected = len(self.descendants(cid) & relevant)
            impact.append((cid, affected))
        return sorted(impact, key=lambda x: x[1], reverse=True)

    def has_cycle(self) -> bool:
        nodes = {c.id for c in self.state.claims}
        indegree = {n: len(self.parents.get(n, ())) for n in nodes}
        queue = deque(n for n, d in indegree.items() if d == 0)
        seen = 0
        while queue:
            node = queue.popleft()
            seen += 1
            for child in self.children.get(node, ()):
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)
        return seen != len(nodes)

    def proof_order(self, root_claim_id: str) -> list[str]:
        """Dependencies-first order for the subgraph needed by a root claim."""
        relevant = self.ancestors(root_claim_id) | {root_claim_id}
        indegree = {n: len(self.parents.get(n, set()) & relevant) for n in relevant}
        queue = deque(sorted(n for n, d in indegree.items() if d == 0))
        out: list[str] = []
        while queue:
            node = queue.popleft()
            out.append(node)
            for child in sorted(self.children.get(node, set()) & relevant):
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)
        if len(out) != len(relevant):
            raise ValueError("proof dependency graph contains a cycle")
        return out

    def is_dependency_closed(self, root_claim_id: str) -> bool:
        root = self.state.claim(root_claim_id)
        if root.status not in {ClaimStatus.VERIFIED, ClaimStatus.FORMALIZED}:
            return False
        return not self.state.unresolved_dependencies(root_claim_id)

    def ready_for_formalization(self, root_claim_id: str) -> bool:
        if self.has_cycle() or not self.is_dependency_closed(root_claim_id):
            return False
        audit = self.state.latest_audit(root_claim_id)
        return audit is None or audit.status != AuditStatus.BLOCK
