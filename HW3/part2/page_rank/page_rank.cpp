#include "page_rank.h"

#include <cmath>
#include <cstdlib>
#include <omp.h>

#include "../common/graph.h"

// page_rank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void page_rank(Graph g, double *solution, double damping, double convergence)
{

    // initialize vertex weights to uniform probability. Double
    // precision scores are used to avoid underflow for large graphs

    int nnodes = num_nodes(g);
    double equal_prob = 1.0 / nnodes;
    for (int i = 0; i < nnodes; ++i)
    {
        solution[i] = equal_prob;
    }

    /*
       For PP students: Implement the page rank algorithm here.  You
       are expected to parallelize the algorithm using openMP.  Your
       solution may need to allocate (and free) temporary arrays.

       Basic page rank pseudocode is provided below to get you started:

       // initialization: see example code above
       score_old[vi] = 1/nnodes;

       while (!converged) {

         // compute score_new[vi] for all nodes vi:
         score_new[vi] = sum over all nodes vj reachable from incoming edges
                            { score_old[vj] / number of edges leaving vj  }
         score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / nnodes;

         score_new[vi] += sum over all nodes v in graph with no outgoing edges
                            { damping * score_old[v] / nnodes }

         // compute how much per-node scores have changed
         // quit once algorithm has converged

         global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
         converged = (global_diff < convergence)
       }

     */
     // 配置兩個陣列用於 double buffering
     double *score_old = new double[nnodes];
     double *score_new = new double[nnodes];

     // 初始化 score_old
     #pragma omp parallel for
     for(int i=0;i<nnodes;i++){
        score_old[i] = solution[i];
     }

     bool converged = false;
     while(!converged){
        // 先計算 no outgoing edges 的貢獻 (dead end)
        double no_outgoing_contrib = 0.0;
        #pragma omp parallel for reduction(+:no_outgoing_contrib)
        for(int vi=0;vi<nnodes;vi++){
          if(outgoing_size(g,vi)==0){
            no_outgoing_contrib += damping * score_old[vi] / nnodes;
          }
        }

        // 合併多個步驟到一個平行區域，減少 parallel overhead
        double global_diff = 0.0;
        #pragma omp parallel for reduction(+:global_diff)
        for(int vi=0;vi<nnodes;vi++){
          // 計算新分數：sum over all incoming edges
          double sum = 0.0;
          const Vertex* start = incoming_begin(g, vi);
          const Vertex* end = incoming_end(g, vi);
          for(const Vertex* neighbor = start; neighbor != end; neighbor++){
            int vj = *neighbor;
            sum += score_old[vj] / outgoing_size(g, vj);
          }

          // 應用 damping factor 和 dead end 貢獻
          score_new[vi] = (damping * sum) + (1.0 - damping) / nnodes + no_outgoing_contrib;

          // 同時計算 diff
          global_diff += fabs(score_new[vi] - score_old[vi]);
        }

        converged = (global_diff < convergence);

        // 交換指標而不是複製數據，避免額外的記憶體操作
        double *temp = score_old;
        score_old = score_new;
        score_new = temp;
     }

     // 將最終結果複製回 solution
     #pragma omp parallel for
     for(int i=0;i<nnodes;i++){
        solution[i] = score_old[i];
     }

     delete[] score_old;
     delete[] score_new;
}
