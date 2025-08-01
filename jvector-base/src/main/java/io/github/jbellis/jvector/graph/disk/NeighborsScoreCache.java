package io.github.jbellis.jvector.graph.disk;

import io.github.jbellis.jvector.graph.*;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

public class NeighborsScoreCache {
    private final Map<Integer, Map<Integer, NodeArray>> perLevelNeighborsScoreCache;

    public NeighborsScoreCache(OnHeapGraphIndex graphIndex) throws IOException {
        try (OnHeapGraphIndex.FrozenView view = graphIndex.getFrozenView()) {
            final Map<Integer, Map<Integer, NodeArray>> perLevelNeighborsScoreCache = new HashMap<>(graphIndex.getMaxLevel() + 1);
            for (int level = 0; level <= graphIndex.getMaxLevel(); level++) {
                final Map<Integer, NodeArray> levelNeighborsScores = new HashMap<>(graphIndex.size(level) + 1);
                final NodesIterator nodesIterator = graphIndex.getNodes(level);
                while (nodesIterator.hasNext()) {
                    final int nodeId = nodesIterator.nextInt();

                    ConcurrentNeighborMap.NeighborIterator neighborIterator = (ConcurrentNeighborMap.NeighborIterator) view.getNeighborsIterator(level, nodeId);
                    final NodeArray neighbours = neighborIterator.merge(new NodeArray(neighborIterator.size()));
                    levelNeighborsScores.put(nodeId, neighbours);
                }

                perLevelNeighborsScoreCache.put(level, levelNeighborsScores);
            }

            this.perLevelNeighborsScoreCache = perLevelNeighborsScoreCache;
        }
    }

    public NodeArray getNeighborsScore(int node, int level) {
        return perLevelNeighborsScoreCache.get(level).get(node);
    }

    public Map<Integer, NodeArray> getNeighborsScoresInLevel(int level) {
        return perLevelNeighborsScoreCache.get(level);
    }
}
