"""本地引文图；有向边 A -> B 表示 A 引用 B。该图仅包含提供的供应商记录，
并不声称是完整的全球引文网络。在遍历时不会进行任何 API 调用。
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from typing import Literal, Protocol

import networkx as nx

from services.paper import Paper, RelatedPaper

Direction = Literal["forward", "backward"]
Era = Literal["classic", "latest"]


class PaperProvider(Protocol):
    """用于种子加载的最小可注入服务合同。"""

    def get_paper(self, paper_id: str) -> Paper | None:
        """返回标准化的论文详细信息，或在服务失败时返回 None。"""
        ...

    def forward_citations(self, paper_id: str) -> list[Paper]:
        """返回引用所提供论文的作品。"""
        ...


class CitationGraph:
    """具有确定性本地遍历的去重定向引用图。"""

    def __init__(self) -> None:
        """创建一个空图而不初始化任何外部服务。"""
        self._graph = nx.DiGraph()
        self.seed_id: str | None = None

    @classmethod
    def from_papers(cls, papers: Iterable[Paper]) -> CitationGraph:
        """从标准化的参考文献/引文构建，保留边的方向。"""
        graph = cls()
        for paper in papers:
            graph.add_paper(paper)
        return graph

    @classmethod
    def from_seed(cls, paper_id: str, *, client: PaperProvider) -> CitationGraph:
        """通过服务获取一个种子；如果失败则返回一个空的图。只加载与种子返回的关系。
        使用 add_paper() 来丰富本地网络以进行多跳探索。
        重试由提供的服务负责。seed_id 是规范的 API ID，而不是 DOI 别名。
        """
        if not paper_id.strip():
            raise ValueError("paper_id must not be blank")
        graph = cls()
        paper = client.get_paper(paper_id.strip())
        if paper is not None:
            forward = getattr(client, "forward_citations", None)
            citations = (
                tuple(RelatedPaper(item.paper_id, item.title, item.year)
                      for item in forward(paper.paper_id))
                if callable(forward)
                else paper.citations
            )
            if hasattr(paper, "model_copy"):
                paper = paper.model_copy(update={"citations": citations})
            graph.add_paper(paper)
            graph.seed_id = paper.paper_id
        return graph

    def add_paper(self, paper: Paper) -> None:
        """合并一条记录及其关联；忽略无效的ID和自引用。
        缺失的元数据永远不会覆盖已知的元数据。完整的论文记录优先于摘要的相关论文记录。
        """
        if not paper.paper_id.strip():
            raise ValueError("paper_id must not be blank")
        self._merge(RelatedPaper(paper.paper_id, paper.title, paper.year), full=True)
        for related in paper.references:
            if not related.paper_id.strip() or related.paper_id == paper.paper_id:
                continue
            self._merge(related)
            self._graph.add_edge(paper.paper_id, related.paper_id)
        for related in paper.citations:
            if not related.paper_id.strip() or related.paper_id == paper.paper_id:
                continue
            self._merge(related)
            self._graph.add_edge(related.paper_id, paper.paper_id)

    def _merge(self, paper: RelatedPaper, *, full: bool = False) -> None:
        if paper.paper_id not in self._graph:
            self._graph.add_node(paper.paper_id, title=None, year=None, full=False)
        data = self._graph.nodes[paper.paper_id]
        for name, value in (("title", paper.title), ("year", paper.year)):
            if value is not None and (full or data[name] is None):
                data[name] = value
        data["full"] = data["full"] or full

    def number_of_nodes(self) -> int:
        """返回不同规范论文ID的数量。"""
        return self._graph.number_of_nodes()

    def number_of_edges(self) -> int:
        """返回不同有向引用关系的数量。"""
        return self._graph.number_of_edges()

    def citation_chains(
        self, paper_id: str, *, direction: Direction, hops: int = 1,
    ) -> list[tuple[str, ...]]:
        """为每个可到达的非种子节点返回一条确定性的最短路径。每个元组以种子节点开始。
        向后元组沿存储的引用箭头跟随；向前元组沿箭头反向跟随。
        BFS 对每个节点只访问一次，因此循环和分支不会枚举指数路径。
        """
        if direction not in ("forward", "backward"):
            raise ValueError("direction must be forward or backward")
        if isinstance(hops, bool) or not isinstance(hops, int) or hops < 0:
            raise ValueError("hops must be a non-negative integer")
        if paper_id not in self._graph or hops == 0:
            return []
        # NetworkX 有向图的前驱/后继反映存储的箭头：
        # https://networkx.org/documentation/stable/reference/classes/digraph.html
        neighbors = (
            self._graph.predecessors if direction == "forward"
            else self._graph.successors
        )
        visited = {paper_id}
        queue = deque([(paper_id,)])
        paths: list[tuple[str, ...]] = []
        while queue:
            path = queue.popleft()
            if len(path) - 1 >= hops:
                continue
            for neighbor in sorted(neighbors(path[-1])):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                extended = (*path, neighbor)
                paths.append(extended)
                queue.append(extended)
        return paths

    def forward_citations(
        self, paper_id: str, *, hops: int = 1, min_year: int | None = None,
        max_year: int | None = None, era: Era | None = None,
    ) -> list[RelatedPaper]:
        """返回引用种子论文的论文（或当跳数 > 1 时引用其论文的论文）。"""
        return self._query(paper_id, "forward", hops, min_year, max_year, era)

    def backward_citations(
        self, paper_id: str, *, hops: int = 1, min_year: int | None = None,
        max_year: int | None = None, era: Era | None = None,
    ) -> list[RelatedPaper]:
        """返回由种子引用的论文，可选择跨多个跳数。"""
        return self._query(paper_id, "backward", hops, min_year, max_year, era)

    def walk_backward(self, paper_id: str, hops: int = 1) -> list[RelatedPaper]:
        """返回广度优先引用遍历，不包括种子。"""
        return self.backward_citations(paper_id, hops=hops)

    def _query(
        self, paper_id: str, direction: Direction, hops: int,
        min_year: int | None, max_year: int | None, era: Era | None,
    ) -> list[RelatedPaper]:
        if era not in (None, "classic", "latest"):
            raise ValueError("era must be classic, latest or None")
        for year in (min_year, max_year):
            if year is not None and (type(year) is not int or year < 1):
                raise ValueError("year bounds must be positive integers")
        if min_year is not None and max_year is not None and min_year > max_year:
            raise ValueError("min_year must not exceed max_year")
        if era == "classic" and max_year is None:
            raise ValueError("classic requires an explicit max_year cutoff")
        if era == "latest" and min_year is None:
            raise ValueError("latest requires an explicit min_year cutoff")
        paths = self.citation_chains(paper_id, direction=direction, hops=hops)
        results: list[RelatedPaper] = []
        for path in paths:
            node = self._graph.nodes[path[-1]]
            year = node["year"]
            # 仅筛选结果，而不是引用路径上的中间节点。
            if (min_year is not None or max_year is not None) and year is None:
                continue
            if min_year is not None and year < min_year:
                continue
            if max_year is not None and year > max_year:
                continue
            results.append(RelatedPaper(path[-1], node["title"], year))
        if era:
            results.sort(key=lambda p: (
                p.year if era == "classic" else -p.year, p.paper_id,
            ))
        return results
