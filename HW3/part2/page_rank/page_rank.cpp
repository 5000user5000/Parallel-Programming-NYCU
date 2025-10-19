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
     bool converged = false;
     while(!converged){
        double *score_new = new double[nnodes];

        // 計算新分數
        // sum over all incoming edges
        for(int vi=0;vi<nnodes;vi++){
          score_new[vi] = 0.0;  // 初始化
          const Vertex* start = incoming_begin(g, vi);
          const Vertex* end = incoming_end(g, vi);
          for(const Vertex* neighbor = start; neighbor != end; neighbor++){
            int vj = *neighbor;  // 解引用获取邻居节点编号
            score_new[vi] += solution[vj] / outgoing_size(g, vj);
          }
        }

        for(int vi=0;vi<nnodes;vi++){
            score_new[vi] = (damping * score_new[vi]) + (1.0 - damping) / nnodes;
        }

        // 處理 dead end (no outgoing)
        double bias = 0.0;
        for(int vi=0;vi<nnodes;vi++){
          if(outgoing_size(g,vi)==0){
            bias += damping*solution[vi] / nnodes;
          }
        }
        // 將 bias 累加上去
        for(int vi=0;vi<nnodes;vi++){
          score_new[vi] += bias;
        }

        // 計算 global_diff
        double global_diff = 0.0;
        for(int vi=0;vi<nnodes;vi++){
          global_diff += fabs(score_new[vi]-solution[vi]);
        }

        converged = (global_diff < convergence);

        // update score
        for(int i=0;i<nnodes;i++){
          solution[i] = score_new[i];
        }

        delete[] score_new;
     }
}
