#include "bfs.h"

#include <cstdlib>
#include <omp.h>

#include "../common/graph.h"

#ifdef VERBOSE
#include "../common/CycleTimer.h"
#include <stdio.h>
#endif // VERBOSE

constexpr int ROOT_NODE_ID = 0;
constexpr int NOT_VISITED_MARKER = -1;

void vertex_set_clear(VertexSet *list)
{
    list->count = 0;
}

void vertex_set_init(VertexSet *list, int count)
{
    list->max_vertices = count;
    list->vertices = new int[list->max_vertices];
    vertex_set_clear(list);
}

void vertex_set_destroy(VertexSet *list)
{
    delete[] list->vertices;
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step(Graph g, VertexSet *frontier, VertexSet *new_frontier, int *distances)
{
    #pragma omp parallel
    {
        // Each thread maintains a local buffer for new vertices
        const int LOCAL_BUFFER_SIZE = 1024;
        int local_vertices[LOCAL_BUFFER_SIZE];
        int local_count = 0;

        #pragma omp for schedule(dynamic, 1024)
        for (int i = 0; i < frontier->count; i++)
        {
            int node = frontier->vertices[i];

            int start_edge = g->outgoing_starts[node];
            int end_edge = (node == g->num_nodes - 1) ? g->num_edges : g->outgoing_starts[node + 1];

            // attempt to add all neighbors to the new frontier
            for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
            {
                int outgoing = g->outgoing_edges[neighbor];

                // Avoid compare_and_swap if we know it will fail
                if (distances[outgoing] == NOT_VISITED_MARKER)
                {
                    if (__sync_bool_compare_and_swap(&distances[outgoing], NOT_VISITED_MARKER, distances[node] + 1))
                    {
                        local_vertices[local_count++] = outgoing;

                        // Flush local buffer if full
                        if (local_count == LOCAL_BUFFER_SIZE)
                        {
                            #pragma omp critical
                            {
                                for (int j = 0; j < local_count; j++)
                                {
                                    new_frontier->vertices[new_frontier->count++] = local_vertices[j];
                                }
                            }
                            local_count = 0;
                        }
                    }
                }
            }
        }

        // Merge remaining local buffers into global new_frontier
        if (local_count > 0)
        {
            #pragma omp critical
            {
                for (int i = 0; i < local_count; i++)
                {
                    new_frontier->vertices[new_frontier->count++] = local_vertices[i];
                }
            }
        }
    }
}

// Take one step of "bottom-up" BFS.  For each unvisited vertex,
// check all incoming edges to see if any neighbor is on the frontier.
// If so, add this vertex to the new_frontier.
void bottom_up_step(Graph g, VertexSet *frontier, VertexSet *new_frontier, int *distances, int current_distance){
    #pragma omp parallel
    {
        // Each thread maintains a local buffer for new vertices
        const int LOCAL_BUFFER_SIZE = 1024;
        int local_vertices[LOCAL_BUFFER_SIZE];
        int local_count = 0;

        #pragma omp for schedule(dynamic, 1024)
        for (int node = 0; node < g->num_nodes; node++)
        {
            // Skip already visited nodes
            if (distances[node] != NOT_VISITED_MARKER)
                continue;

            int start_edge = g->incoming_starts[node];
            int end_edge = (node == g->num_nodes - 1) ? g->num_edges : g->incoming_starts[node + 1];

            // Check all incoming neighbors to see if any are in the frontier
            for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
            {
                int incoming = g->incoming_edges[neighbor];

                // If neighbor is in the frontier (was visited in the current level), add this node to new_frontier
                if (distances[incoming] == current_distance)
                {
                    distances[node] = current_distance + 1;
                    local_vertices[local_count++] = node;

                    // Flush local buffer if full
                    if (local_count == LOCAL_BUFFER_SIZE)
                    {
                        #pragma omp critical
                        {
                            for (int j = 0; j < local_count; j++)
                            {
                                new_frontier->vertices[new_frontier->count++] = local_vertices[j];
                            }
                        }
                        local_count = 0;
                    }
                    break; // No need to check other neighbors
                }
            }
        }

        // Merge remaining local buffers into global new_frontier
        if (local_count > 0)
        {
            #pragma omp critical
            {
                for (int i = 0; i < local_count; i++)
                {
                    new_frontier->vertices[new_frontier->count++] = local_vertices[i];
                }
            }
        }
    }
}



// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution *sol)
{

    VertexSet list1;
    VertexSet list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    VertexSet *frontier = &list1;
    VertexSet *new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0)
    {

#ifdef VERBOSE
        double start_time = CycleTimer::current_seconds();
#endif

        vertex_set_clear(new_frontier);

        top_down_step(graph, frontier, new_frontier, sol->distances);

#ifdef VERBOSE
        double end_time = CycleTimer::current_seconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        VertexSet *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }

    // free memory
    vertex_set_destroy(&list1);
    vertex_set_destroy(&list2);
}

void bfs_bottom_up(Graph graph, solution *sol)
{
    VertexSet list1;
    VertexSet list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    VertexSet *frontier = &list1;
    VertexSet *new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    int current_distance = 0;

    while (frontier->count != 0)
    {
#ifdef VERBOSE
        double start_time = CycleTimer::current_seconds();
#endif

        vertex_set_clear(new_frontier);

        bottom_up_step(graph, frontier, new_frontier, sol->distances, current_distance);

#ifdef VERBOSE
        double end_time = CycleTimer::current_seconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        VertexSet *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;

        current_distance++;
    }

    // free memory
    vertex_set_destroy(&list1);
    vertex_set_destroy(&list2);
}

void bfs_hybrid(Graph graph, solution *sol)
{
    // For PP students:
    //
    // You will need to implement the "hybrid" BFS here as
    // described in the handout.
}
