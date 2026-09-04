import heapq
import itertools
import math


def shortest_path(nodes, edges, lengths, source, target):
    """Return the length and edge sequence of a shortest directed path."""
    outgoing = {node: [] for node in nodes}

    for edge in edges:
        outgoing[edge[0]].append(edge)

    distance = {node: math.inf for node in nodes}
    previous = {}
    counter = itertools.count()
    distance[source] = 0.0
    pending = [(0.0, next(counter), source)]

    while pending:
        current_distance, _, node = heapq.heappop(pending)

        if current_distance > distance[node] + 1e-12:
            continue
        if node == target:
            break

        for edge in outgoing[node]:
            successor = edge[1]
            candidate = current_distance + lengths[edge]

            if candidate < distance[successor] - 1e-12:
                distance[successor] = candidate
                previous[successor] = (node, edge)
                heapq.heappush(
                    pending,
                    (candidate, next(counter), successor),
                )

    if not math.isfinite(distance[target]):
        return math.inf, []

    path = []
    node = target
    while node != source:
        node, edge = previous[node]
        path.append(edge)
    path.reverse()

    return distance[target], path


def directed_min_cut(nodes, edges, capacities, source, target, tolerance=1e-9):
    """Return a minimum directed cut using an Edmonds--Karp residual graph."""
    adjacent = {node: set() for node in nodes}
    residual = {node: {} for node in nodes}

    for tail, head in edges:
        capacity = max(0.0, float(capacities[tail, head]))
        adjacent[tail].add(head)
        adjacent[head].add(tail)
        residual[tail][head] = residual[tail].get(head, 0.0) + capacity
        residual[head].setdefault(tail, 0.0)

    while True:
        previous = {source: None}
        pending = [source]
        position = 0

        while position < len(pending) and target not in previous:
            node = pending[position]
            position += 1

            for successor in sorted(adjacent[node], key=str):
                if successor in previous:
                    continue
                if residual[node].get(successor, 0.0) <= tolerance:
                    continue

                previous[successor] = node
                pending.append(successor)

        if target not in previous:
            break

        increment = math.inf
        node = target
        while node != source:
            predecessor = previous[node]
            increment = min(increment, residual[predecessor][node])
            node = predecessor

        node = target
        while node != source:
            predecessor = previous[node]
            residual[predecessor][node] -= increment
            residual[node][predecessor] = (
                residual[node].get(predecessor, 0.0) + increment
            )
            node = predecessor

    reachable = {source}
    pending = [source]

    while pending:
        node = pending.pop()

        for successor in sorted(adjacent[node], key=str):
            if successor in reachable:
                continue
            if residual[node].get(successor, 0.0) <= tolerance:
                continue

            reachable.add(successor)
            pending.append(successor)

    cut_edges = [
        edge
        for edge in edges
        if edge[0] in reachable and edge[1] not in reachable
    ]
    cut_value = sum(capacities[edge] for edge in cut_edges)

    return float(cut_value), reachable, cut_edges
