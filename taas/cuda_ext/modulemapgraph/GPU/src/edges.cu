/***********************************************************************
 * Module Map Graph library
 * Track reconstruction for HEP
 * copyright © 2024,2025,2026 COLLARD Christophe
 * Laboratoire des 2 Infinis - Toulouse - France (L2IT)
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 **********************************************************************/

#include "MMG/edges.cuh"

// delete after deleting sort_old
// get target vertices of the source layer that are connected  to the source vertices of the target layer                    // -----o--o--o--o--     Target Layer - Target vertices    outer
//---------------------------------------------------------------------------------------------------------------------      //      |  |  |  |             Target layer                  |
template <class T>                                                                                                           // -----o--o--o--o--     Target Layer - Source vertices      |
__global__ void connect_edges (T *mask,                                                                                      //                                                           | ATLAS detector
                               int nb_source_edges, int *SourceLayer_target_vertices,                                        // --o--o--o--o--o--o--  Source Layer - Target vertices      |
                               int nb_target_edges, int *TargetLayer_source_vertices){                                       //   |  |  |  |  |  |           Source layer                  |
//---------------------------------------------------------------------------------------------------------------------      // --o--o--o--o--o--o--  Source Layer - Source vertices    inner
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i >= nb_source_edges || j >= nb_target_edges) return;

  if (SourceLayer_target_vertices[i] == TargetLayer_source_vertices[j])
    mask[i] = true;

}






__global__ void print (bool* vector, int size)
{
  for (int i=0; i<size; i++)
    printf ("%d ", vector[i]);
    printf ("\n");
}

__global__ void print (int* vector, int size)
{
  for (int i=0; i<size; i++)
    printf ("%d ", vector[i]);
    printf ("\n");
}

__global__ void print (float* vector, int size)
{
  printf ("print vector from edges.cu \n");
  for (int i=0; i<size; i++)
    printf ("%f ", vector[i]);
    printf ("\n");
}

__global__ void print_edges (int *source, int *target, int * global_edges_position, int *global_edges_tag, int nb_edges)
{
  for (int i=0; i<nb_edges; i++)
    printf ("@%d global position = %d, source = %d, target = %d, edge tag = %d \n", i, global_edges_position[i], source[i], target[i], global_edges_tag [global_edges_position[i]]);
}

template <class T>
__global__ void print_edge (int *source, int *target, T *weight, int *tags, int value, int nb_edges)
{
  for (int i=0; i<nb_edges; i++){
    if (source[i] == value)
      printf ("i=%d, source = \033[0;34m%d\033[0m, target = %d, weight = %f, tag = %d \n", i, source[i], target[i], weight[i], tags[i]);
    if (target[i] == value)
      printf ("i=%d, source = %d, target = \033[0;34m%d\033[0m, weight = %f, tag = %d \n", i, source[i], target[i], weight[i], tags[i]);
  }
}

template <class T>
__global__ void print_edge_source (int *source, int *target, T *weight, int *tags, int value, int nb_edges)
{
  for (int i=0; i<nb_edges; i++)
    if (source[i] == value)
      printf ("i=%d, source = \033[0;34m%d\033[0m, target = %d, weight = %f, tag = %d \n", i, source[i], target[i], weight[i], tags[i]);
}

template <class T>
__global__ void print_edge_target (int *source, int *target, T *weight, int *tags, int value, int nb_edges)
{
  for (int i=0; i<nb_edges; i++)
    if (target[i] == value)
      printf ("i=%d, source = %d, target = \033[0;34m%d\033[0m, weight = %f, tag = %d \n", i, source[i], target[i], weight[i], tags[i]);
}

template <class T>
__global__ void print_edge_tag (int *source, int *target, T *weight, int *tags, int value, int nb_edges)
{
  for (int i=0; i<nb_edges; i++)
    if (tags[i] == value)
      printf ("i=%d, source = \033[0;34m%d\033[0m, target = %d, weight = %f, tag = %d \n", i, source[i], target[i], weight[i], tags[i]);
}

__global__ void find_edge (int *source, int *target, int indice, int size){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size) return;

//  if (source[i] == indice){
//    printf ("\033[32m%d ", source[i]);
//    printf ("\033[31m%d ", target[i]);
//    printf ("\033[0;30m\n");
//  }

  if (target[i] == indice)
    printf ("\033[31m%d \033[32m%d \033[0;30m\n", source[i], target[i]);
}


//=====GPU functions for edges=========================================================================================


//---------------------------------------------------------------------------------------------------------------------
__global__ void tag (int *layer,
                     int tag, int size){
//---------------------------------------------------------------------------------------------------------------------
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size) return;

  layer[i] = tag + i + 1;
}

//---------------------------------------------------------------------------------------------------------------------
__global__ void tag (int *layer_tag,
                     int layer_nb_edges, const int *layer_target,
                     const int *next_layer_tag, int next_layer_nb_edges, const int *next_layer_source){
//---------------------------------------------------------------------------------------------------------------------
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i >= layer_nb_edges || j >= next_layer_nb_edges) return;

  if (layer_target[i] == next_layer_source[j] && next_layer_tag[j])
    layer_tag[i] = next_layer_tag [j];
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
__global__ void duplicate_vertices (int *tag, int *shared_hit,
                                    const int *vertex, const T* weight, int nb_vertices){
//---------------------------------------------------------------------------------------------------------------------
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= nb_vertices || j >= nb_vertices || i >= j) return;

// if tag[i] != tag[j]) return;

  if (vertex[i] != vertex[j]) return;
  if (weight[i] * weight[j] > 0.81  &&  tag[i] != tag[j]){
    shared_hit[i] = tag[j];
    shared_hit[j] = tag[i];
    return;
  }

  int k = (weight[i] < weight[j]) ? i : j;
  tag[k] = 0;
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
__global__ void find_duplicate_vertices (bool* kill_mask,
                                         const bool *mask, const int *vertex, const T* weight, int nb_vertices){
//---------------------------------------------------------------------------------------------------------------------
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i >= nb_vertices || j >= nb_vertices || i >= j) return;
  if (vertex[i] != vertex[j]) return;

  int k;
  if (mask[i] != mask[j])
    k = i*mask[j] + j*mask[i]; // edge to kill
  else
    k = (weight[i] < weight[j]) ? i : j; // tag the edge with the lowest score (to kill)

  kill_mask[k] = true; // tag the edge to kill
}

// get target vertices of the source layer that are connected  to the source vertices of the target layer                    // -----o--o--o--o--     Target Layer - Target vertices    outer
//---------------------------------------------------------------------------------------------------------------------      //      |  |  |  |             Target layer                  |
template <class T>                                                                                                           // -----o--o--o--o--     Target Layer - Source vertices      |
__global__ void connect_edges_tags (T *mask,                                                                                 //                                                           | ATLAS detector
                                    int nb_source_edges, const int *SourceLayer_target_vertices,                             // --o--o--o--o--o--o--  Source Layer - Target vertices      |
                                    int nb_target_edges, const int *TargetLayer_source_vertices){                            //   |  |  |  |  |  |           Source layer                 |
//---------------------------------------------------------------------------------------------------------------------      // --o--o--o--o--o--o--  Source Layer - Source vertices    inner
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i >= nb_source_edges || j >= nb_target_edges) return;

  if (SourceLayer_target_vertices[i] == TargetLayer_source_vertices[j])
    mask[i] = true;

}

//---------------------------------------------------------------------------------------------------------------------
// connect edges -> update tag and weight
template <class T>
__global__ void connect_edges (
                            int layer_nb_edges, int *layer_target, T *layer_weight, int *layer_tag,
                            int next_layer_nb_edges, int *next_layer_source, T *next_layer_weight, int* next_layer_tag,
                            int *shared_hit){
//---------------------------------------------------------------------------------------------------------------------
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= layer_nb_edges || j >= next_layer_nb_edges) return;

  if (layer_target[i] == next_layer_source[j] && next_layer_tag[j]) {
//  if (layer_target[i] == next_layer_source[j]) {
//    if (shared_hit[j] != -1) return; //&& (abs (layer_rphislope[i] - next_layer_rphislope[j]) > abs (layer_rphislope[i] - shared_rphislope[j]))) return;
    layer_tag[i] = next_layer_tag [j];
    layer_weight[i] *= next_layer_weight[j];
  }
}

/*
//---------------------------------------------------------------------------------------------------------------------
__global__ void  filter_last_vertices (bool *mask,
                                       int *source, int *target, int nb_edges,
                                       int *last_edges_source, int *last_edges_target, int last_edges_nb_edges){
//---------------------------------------------------------------------------------------------------------------------
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i >= nb_edges || j >= last_edges_nb_edges) return;
  if (source[i] == 205516 && target[i] == 203702) printf ("edge found\n");
  if (mask[j]) return;
  if (source[i] == 205516 && target[i] == 203702) printf ("edge found\n");

  if (source[i] == last_edges_source[j]  &&  target[i] == last_edges_target[j])
    last_edges_target[j] = -1;
  else
    mask[j] = true;
}
*/

//---------------------------------------------------------------------------------------------------------------------
__global__ void  filter_last_vertices (bool *rev_mask,
                                       int *source, int *target, int nb_edges,
                                       int *last_edges_source, int *last_edges_target, int last_edges_nb_edges){
//---------------------------------------------------------------------------------------------------------------------
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= nb_edges) return;
  if (rev_mask[i]) return;

  bool not_last_edge = true;
  for (int j=0; j<last_edges_nb_edges; j++)
    if (source[i] == last_edges_source[j]  &&  target[i] == last_edges_target[j]){
      last_edges_target[j] = -1;
      not_last_edge = false;
    }

  rev_mask[i] = not_last_edge;
}

//---------------------------------------------------------------------------------------------------------------------
__global__ void  filter_last_vertices_2D (bool *last_edge,
                                       int *source, int *target, int nb_edges,
                                       int *last_edges_source, int *last_edges_target, int last_edges_nb_edges){
//---------------------------------------------------------------------------------------------------------------------
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= nb_edges || j >= last_edges_nb_edges) return;

  if (source[i] == last_edges_source[j]  &&  target[i] == last_edges_target[j]){
    last_edge[i] = true;
    last_edges_target[j] = -1;
  }
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
__global__ void score_filter (bool *rev_mask,
                                   T *weight, T score, int nb_edges){
//---------------------------------------------------------------------------------------------------------------------
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nb_edges) return;

  if (weight[i] < score)
    rev_mask[i] = true;
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
__global__ void merge (T *vertex,
                       T *next_layer_vertices, T *new_vertices, int n, int nb_edges){
//---------------------------------------------------------------------------------------------------------------------
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nb_edges) return;

  T *edge_set;
  int k = i;
  if (i < n) edge_set = next_layer_vertices;
  else {
    edge_set = new_vertices;
    k = i-n;
  }

  vertex[i] = edge_set[k];
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
__global__ void merge_by_key (T *vertex,
                              T *next_layer_vertices, T *new_vertices, int *key, int n, int nb_edges){
//---------------------------------------------------------------------------------------------------------------------
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nb_edges) return;

  T *edge_set;
  int k;
  if (i < n) {
    edge_set = next_layer_vertices;
    k = key[i];
  }
  else {
    edge_set = new_vertices;
    k = i-n;
  }

  vertex[i] = edge_set[k];
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
__global__ void edge_features (T *dR, T *dz, T *r_phi_slope, int *M1_SP, int *M2_SP,
                               int* vertices_sum, T *R, T *z, T *phi_slope, data_type2 pi, int nb_edges){
//---------------------------------------------------------------------------------------------------------------------
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nb_edges) return;

  int SP1 = M1_SP[i];
  int SP2 = M2_SP[i];
  T r_sum = 0.5 * (R[SP1] + R[SP2]);
  dR[i] = R[SP2] - R[SP1];
  dz[i] = z[SP2] - z[SP1];
  M1_SP[i] = vertices_sum[SP1];
  M2_SP[i] = vertices_sum[SP2];

  if (dR)
    r_phi_slope[i] = r_sum * phi_slope[i];
  else
    r_phi_slope[i] = phi_slope[i];
}

//---------------------------------------------------------------------------------------------------------------------
__global__ void convert2ACORN (int *ACORN_edge_vertex,
                               int *edge_vertex, int *ACORN_positions, int nb_edges){
//---------------------------------------------------------------------------------------------------------------------
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nb_edges) return;

  ACORN_edge_vertex[i] = ACORN_positions [edge_vertex[i]];
}


//=====CPU functions for edges=========================================================================================


//---------------------------------------------------------------------------------------------------------------------
template <class T>
CUDA_edges<T>::CUDA_edges (int n){
//---------------------------------------------------------------------------------------------------------------------
  _nb_edges = n;
  _tag = 0;
  _ACORN = _ACTS = _decorations = false;
  cudaMalloc (&_edge_source, _nb_edges * sizeof(int));
  cudaMalloc (&_edge_target, _nb_edges * sizeof(int));
  cudaMalloc (&_edge_weight, _nb_edges * sizeof(T));
  cudaMalloc (&_edge_tag, _nb_edges * sizeof(int));
  cudaMalloc (&_shared_hit, _nb_edges * sizeof(int));
  cudaMemsetAsync (_edge_tag, 0, _nb_edges * sizeof(int));
  cudaMemsetAsync (_shared_hit, -1, _nb_edges * sizeof(int));
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
CUDA_edges<T>::CUDA_edges (CUDA_edges<T>* edges, CUDA_mask& mask, int blocks){
//---------------------------------------------------------------------------------------------------------------------
  _ACORN = _ACTS = _decorations = false;
  build (edges, mask, blocks);
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
CUDA_edges<T>::CUDA_edges (int nb_edges, int *edge_source, int *edge_target, T *edge_weight){
//---------------------------------------------------------------------------------------------------------------------
  _nb_edges = nb_edges;
  _tag = 0;
  _ACORN = _ACTS = _decorations = false;
  cudaMemcpyAsync (_edge_source, edge_source, _nb_edges * sizeof (int), cudaMemcpyDeviceToDevice);
  cudaMemcpyAsync (_edge_target, edge_target, _nb_edges * sizeof (int), cudaMemcpyDeviceToDevice);
  cudaMemcpyAsync (_edge_weight, edge_weight, _nb_edges * sizeof (T), cudaMemcpyDeviceToDevice);
  cudaMemsetAsync (_edge_tag, 0, _nb_edges * sizeof(int));
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
CUDA_edges<T>::CUDA_edges (int nb_edges, int *edge_source, int *edge_target, T *edge_weight, bool ACTS){
//---------------------------------------------------------------------------------------------------------------------
  _nb_edges = nb_edges;
  _tag = 0;
  _ACORN = _decorations = false;
  _ACTS = true;
  _edge_source = edge_source;
  _edge_target = edge_target;
  _edge_weight = edge_weight;
  cudaMalloc (&_edge_tag, _nb_edges * sizeof(int));
  cudaMemsetAsync (_edge_tag, 0, _nb_edges * sizeof(int));
  cudaMemsetAsync (_shared_hit, -1, _nb_edges * sizeof(int));
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
CUDA_edges<T>::~CUDA_edges (){
//---------------------------------------------------------------------------------------------------------------------
  if (_nb_edges && !_ACTS) {
    if (_tag != -1) {
      cudaFree (_edge_weight);
      cudaFree (_shared_hit);
      cudaFree (_edge_tag);
    }
    if (_ACORN) {
      cudaFree (_ACORN_edge_target);
      cudaFree (_ACORN_edge_source);
    }
    if (_decorations) {
      cudaFree (_r_phi_slope);
      cudaFree (_phi_slope);
      cudaFree (_dphi);
      cudaFree (_deta);
      cudaFree (_dz);
      cudaFree (_dR);
    }
    cudaFree (_edge_target);
    cudaFree (_edge_source);
  }
  _ACORN = _decorations = false;
  _nb_edges = 0;
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
void CUDA_edges<T>::reduce (int* M1, int* M2, T* deta, T* dphi, T* phi_slope, CUDA_mask& mask, int blocks){
//---------------------------------------------------------------------------------------------------------------------
  assert (!_nb_edges);
  _nb_edges = mask.scan (blocks);
  _decorations = true;
  cudaMalloc (&_edge_source, _nb_edges * sizeof(int));
  cudaMalloc (&_edge_target, _nb_edges * sizeof(int));
  cudaMalloc (&_dR, _nb_edges * sizeof(T));
  cudaMalloc (&_dz, _nb_edges * sizeof(T));
  cudaMalloc (&_deta, _nb_edges * sizeof(T));
  cudaMalloc (&_dphi, _nb_edges * sizeof(T));
  cudaMalloc (&_phi_slope, _nb_edges * sizeof(T));
  cudaMalloc (&_r_phi_slope, _nb_edges * sizeof(T));

  int mask_size = mask.size();
  int grid_dim = (mask_size + blocks - 1) / blocks;
  compact_stream<<<grid_dim,blocks>>> (_edge_source, M1, mask.cuda_mask(), mask.cuda_mask_sum(), mask_size);
  compact_stream<<<grid_dim,blocks>>> (_edge_target, M2, mask.cuda_mask(), mask.cuda_mask_sum(), mask_size);
  compact_stream<<<grid_dim,blocks>>> (_deta, deta, mask.cuda_mask(), mask.cuda_mask_sum(), mask_size);
  compact_stream<<<grid_dim,blocks>>> (_dphi, dphi, mask.cuda_mask(), mask.cuda_mask_sum(), mask_size);
  compact_stream<<<grid_dim,blocks>>> (_phi_slope, phi_slope, mask.cuda_mask(), mask.cuda_mask_sum(), mask_size);
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
void CUDA_edges<T>::reduce (int* M1, int* M2, T* deta, T* dphi, T* phi_slope,
                            CUDA_mask& mask, CUDA_hits<T>& hits, int blocks){
//---------------------------------------------------------------------------------------------------------------------
  reduce (M1, M2, deta, dphi, phi_slope, mask, blocks);
  _ACORN = hits.ACORN();
  if (_ACORN) {
    cudaMalloc (&_ACORN_edge_source, _nb_edges * sizeof(int));
    cudaMalloc (&_ACORN_edge_target, _nb_edges * sizeof(int));
    dim3 grid_dim = (_nb_edges + blocks - 1) / blocks;
    convert2ACORN<<<grid_dim,blocks>>> (_ACORN_edge_source, _edge_source, hits.cuda_ACORN_positions(), _nb_edges);
    convert2ACORN<<<grid_dim,blocks>>> (_ACORN_edge_target, _edge_target, hits.cuda_ACORN_positions(), _nb_edges);
  }
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
void CUDA_edges<T>::reduce (CUDA_mask& mask, int blocks){
//---------------------------------------------------------------------------------------------------------------------
  if (!_nb_edges) return;

  CUDA_edges<T> new_edges (this, mask, blocks);
  cudaFree (_edge_tag);
  cudaFree (_edge_weight);
  cudaFree (_edge_target);
  cudaFree (_edge_source);
  _nb_edges = new_edges._nb_edges;
  _tag = 0;
  _edge_source = new_edges._edge_source;
  _edge_target = new_edges._edge_target;
  _edge_weight = new_edges._edge_weight;
  _edge_tag = new_edges._edge_tag;
  new_edges._nb_edges = 0; // to avoid memory deletion of temporary object (reallocated to current object)
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
void CUDA_edges<T>::reduce (CUDA_mask& mask, int *ACORN_positions, int blocks){
//---------------------------------------------------------------------------------------------------------------------
  if (!_nb_edges) return;

  CUDA_edges<T> new_edges (this, mask, ACORN_positions, blocks);
  cudaFree (_edge_tag);
  cudaFree (_edge_weight);
  cudaFree (_edge_target);
  cudaFree (_edge_source);
  _nb_edges = new_edges._nb_edges;
  _tag = 0;
  _edge_source = new_edges._edge_source;
  _edge_target = new_edges._edge_target;
  _edge_weight = new_edges._edge_weight;
  _edge_tag = new_edges._edge_tag;
  new_edges._nb_edges = 0; // to avoid memory deletion of temporary object (reallocated to current object)
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
void CUDA_edges<T>::reverse_reduce (CUDA_mask& rmask, int blocks){
//---------------------------------------------------------------------------------------------------------------------
  if (!_nb_edges) return;

  CUDA_edges<T> new_edges;
  new_edges.reverse_build (this, rmask, blocks);
  cudaFree (_edge_tag);
  cudaFree (_edge_weight);
  cudaFree (_edge_target);
  cudaFree (_edge_source);
  _nb_edges = new_edges._nb_edges;
  _tag = new_edges._tag;//0;
  _edge_source = new_edges._edge_source;
  _edge_target = new_edges._edge_target;
  _edge_weight = new_edges._edge_weight;
  _edge_tag = new_edges._edge_tag;
  new_edges._nb_edges = 0; // to avoid memory deletion of temporary object (reallocated to current object)
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
void CUDA_edges<T>::build (const CUDA_edges<T>* edges, CUDA_mask& mask, int blocks){
//---------------------------------------------------------------------------------------------------------------------
  _nb_edges = mask.scan (blocks);
  if (!_nb_edges) return;

  _tag = (*edges)._tag;
  cudaMalloc (&_edge_source, _nb_edges * sizeof(int));
  cudaMalloc (&_edge_target, _nb_edges * sizeof(int));
  cudaMalloc (&_edge_weight, _nb_edges * sizeof(T));
  cudaMalloc (&_edge_tag, _nb_edges * sizeof(int));
  cudaMalloc (&_shared_hit, _nb_edges * sizeof(int)); // shared hits

  dim3 grid_dim (((*edges)._nb_edges + blocks - 1) / blocks);
  compact_stream<<<grid_dim,blocks>>> (_edge_source,
                                    (*edges)._edge_source, mask.cuda_mask(), mask.cuda_mask_sum(), (*edges)._nb_edges);
  compact_stream<<<grid_dim,blocks>>> (_edge_target,
                                    (*edges)._edge_target, mask.cuda_mask(), mask.cuda_mask_sum(), (*edges)._nb_edges);
  compact_stream<<<grid_dim,blocks>>> (_edge_weight,
                                    (*edges)._edge_weight, mask.cuda_mask(), mask.cuda_mask_sum(), (*edges)._nb_edges);
  compact_stream<<<grid_dim,blocks>>> (_edge_tag,
                                    (*edges)._edge_tag, mask.cuda_mask(), mask.cuda_mask_sum(), (*edges)._nb_edges);
  compact_stream<<<grid_dim,blocks>>> (_shared_hit,
                                    (*edges)._shared_hit, mask.cuda_mask(), mask.cuda_mask_sum(), (*edges)._nb_edges);
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
void CUDA_edges<T>::build (CUDA_edges<T>* edges, CUDA_mask& mask, int *ACORN_positions, int blocks){
//---------------------------------------------------------------------------------------------------------------------
  build (edges, mask, blocks);

  _ACORN = true;
  cudaMalloc (&_ACORN_edge_source, _nb_edges * sizeof(int));
  cudaMalloc (&_ACORN_edge_target, _nb_edges * sizeof(int));

  dim3 grid_dim = (_nb_edges + blocks - 1) / blocks;
  convert2ACORN<<<grid_dim,blocks>>> (_ACORN_edge_source, _edge_source, ACORN_positions, _nb_edges);
  convert2ACORN<<<grid_dim,blocks>>> (_ACORN_edge_target, _edge_target, ACORN_positions, _nb_edges);
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
void CUDA_edges<T>::reverse_build (const CUDA_edges<T>* edges, CUDA_mask& rmask, int blocks){
//---------------------------------------------------------------------------------------------------------------------
  rmask.scan(blocks);
  _nb_edges = rmask.size() - rmask.sum();
  if (!_nb_edges) return;

  _tag = (*edges)._tag;
  cudaMalloc (&_edge_source, _nb_edges * sizeof(int));
  cudaMalloc (&_edge_target, _nb_edges * sizeof(int));
  cudaMalloc (&_edge_weight, _nb_edges * sizeof(T));
  cudaMalloc (&_edge_tag, _nb_edges * sizeof(int));
  cudaMalloc (&_shared_hit, _nb_edges * sizeof(int));
//  cudaMalloc (&_edge_global_position, _nb_edges * sizeof(int));

  dim3 grid (((*edges)._nb_edges + blocks - 1) / blocks);
  compact_stream_reverse_mask<<<grid,blocks>>> (_edge_source,
                                  (*edges)._edge_source, rmask.cuda_mask(), rmask.cuda_mask_sum(), (*edges)._nb_edges);
  compact_stream_reverse_mask<<<grid,blocks>>> (_edge_target,
                                  (*edges)._edge_target, rmask.cuda_mask(), rmask.cuda_mask_sum(), (*edges)._nb_edges);
  compact_stream_reverse_mask<<<grid,blocks>>> (_edge_weight,
                                  (*edges)._edge_weight, rmask.cuda_mask(), rmask.cuda_mask_sum(), (*edges)._nb_edges);
  compact_stream_reverse_mask<<<grid,blocks>>> (_edge_tag,
                                  (*edges)._edge_tag, rmask.cuda_mask(), rmask.cuda_mask_sum(), (*edges)._nb_edges);
  compact_stream_reverse_mask<<<grid,blocks>>> (_shared_hit,
                                  (*edges)._shared_hit, rmask.cuda_mask(), rmask.cuda_mask_sum(), (*edges)._nb_edges);
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
void CUDA_edges<T>::tag_edges (int blocks, int next_layer_tags){
//---------------------------------------------------------------------------------------------------------------------
  _tag = next_layer_tags;

  dim3 grid_dim = (_nb_edges + blocks - 1) / blocks;
  if (_nb_edges)
    tag<<<grid_dim,blocks>>> (_edge_tag, _tag, _nb_edges);
  _tag += _nb_edges;
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
void CUDA_edges<T>::tag_edges (CUDA_edges<T>& next_layer, int blocks){
//---------------------------------------------------------------------------------------------------------------------
  _tag = next_layer._tag;

  int blocks_2D = sqrt (blocks);
  dim3 block_dim_2D (blocks_2D, blocks_2D);
  int layer_size = (_nb_edges + blocks_2D - 1) / blocks_2D;
  int next_layer_size = (next_layer._nb_edges + blocks_2D - 1) / blocks_2D;
  dim3 grid_dim_2D (layer_size, next_layer_size);

  tag<<<grid_dim_2D,block_dim_2D>>> (_edge_tag, _nb_edges, _edge_target,
                                     next_layer._edge_tag, next_layer._nb_edges, next_layer._edge_source);
}

/*
//---------------------------------------------------------------------------------------------------------------------
template <class T>
void CUDA_edges<T>::tag_new_edges (CUDA_edges<T>& last_edges, int blocks, int next_layer_tags){
//---------------------------------------------------------------------------------------------------------------------
  _tag = next_layer_tags;

  dim3 grid_dim = (_nb_edges + blocks - 1) / blocks;
  if (_nb_edges)
    tag<<<grid_dim,blocks>>> (_edge_tag, _tag, _nb_edges);
  _tag += _nb_edges;
}
*/

//---------------------------------------------------------------------------------------------------------------------
template <class T>
void CUDA_edges<T>::connect_layer_edges (CUDA_mask& edges_mask,
                                         const CUDA_edges<T>& next_layer, int blocks){
//---------------------------------------------------------------------------------------------------------------------
  int blocks_2D = sqrt (blocks);
  dim3 block_dim_2D (blocks_2D, blocks_2D);
  int next_layer_nb_edges = next_layer._nb_edges;
  int size_source = (_nb_edges + blocks_2D - 1) / blocks_2D;
  int size_target = (next_layer_nb_edges + blocks_2D - 1) / blocks_2D;
  dim3 grid_dim_2D (size_source, size_target);

  // create edges mask with 0 if layer.target is not in next_layer.source (new egde) and 1 if layer.target is in next_layer.source (existing edge)
  connect_edges_tags<<<grid_dim_2D,block_dim_2D>>> (edges_mask.cuda_mask(),
                                                    _nb_edges, _edge_target, next_layer_nb_edges, next_layer._edge_source);
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
void CUDA_edges<T>::copy_tags (const CUDA_edges<T>& next_layer, int blocks){
//---------------------------------------------------------------------------------------------------------------------
  _tag = next_layer._tag;

  int blocks_2D = sqrt (blocks);
//  int blocks_2D = blocks;
  dim3 block_dim_2D (blocks_2D, blocks_2D);
  int layer_size = (_nb_edges + blocks_2D - 1) / blocks_2D;
  int next_layer_size = (next_layer._nb_edges + blocks_2D - 1) / blocks_2D;
  dim3 grid_dim_2D (layer_size, next_layer_size);

  ::connect_edges<<<grid_dim_2D,block_dim_2D>>> (
                 _nb_edges, _edge_target, _edge_weight, _edge_tag, 
                 next_layer._nb_edges, next_layer._edge_source, next_layer._edge_weight, next_layer._edge_tag,
                 next_layer._shared_hit);
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
void CUDA_edges<T>::cut_new_edges (CUDA_edges<T>& target_edges_layer, int blocks){
//---------------------------------------------------------------------------------------------------------------------
  int blocks_2D = sqrt (blocks);
//  int blocks_2D = blocks;
  dim3 block_dim_2D (blocks_2D, blocks_2D);
  int nb_source_edges = _nb_edges;
  int nb_target_edges = target_edges_layer.nb_edges();
  int size_source = (nb_source_edges + blocks_2D - 1) / blocks_2D;
  int size_target = (nb_target_edges + blocks_2D - 1) / blocks_2D;
  dim3 grid_dim_2D (size_source, size_target);

  CUDA_mask mask (_nb_edges);
  cudaMemsetAsync (mask.cuda_mask(), false, _nb_edges * sizeof (bool));
cudaDeviceSynchronize();
clock_t start = clock ();
connect_edges_tags<<<grid_dim_2D,block_dim_2D>>>(mask.cuda_mask(),
                                      nb_source_edges, _edge_target, nb_target_edges, target_edges_layer._edge_source);
cudaDeviceSynchronize();
clock_t end_clock = clock();
std::cout << red << "cut new edges cpu time : " << (double)(end_clock-start)/CLOCKS_PER_SEC*1000 << " ms" << reset;
  (*this).reduce (mask, blocks);
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
void CUDA_edges<T>::merge_edges (CUDA_edges<T>& next_layer_edges, CUDA_edges<T>& new_edges, int blocks){
//---------------------------------------------------------------------------------------------------------------------
  dim3 grid ((_nb_edges + blocks - 1) / blocks);
  merge<<<grid,blocks>>> (_edge_source,
                         next_layer_edges._edge_source, new_edges._edge_source, next_layer_edges._nb_edges, _nb_edges);
  merge<<<grid,blocks>>> (_edge_target,
                         next_layer_edges._edge_target, new_edges._edge_target, next_layer_edges._nb_edges, _nb_edges);
  merge<<<grid,blocks>>> (_edge_weight,
                         next_layer_edges._edge_weight, new_edges._edge_weight, next_layer_edges._nb_edges, _nb_edges);
  merge<<<grid,blocks>>> (_edge_tag,
                         next_layer_edges._edge_tag, new_edges._edge_tag, next_layer_edges._nb_edges, _nb_edges);
  merge<<<grid,blocks>>> (_shared_hit,
                         next_layer_edges._shared_hit, new_edges._shared_hit, next_layer_edges._nb_edges, _nb_edges);
  _tag = new_edges._tag;
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
void CUDA_edges<T>::merge_edges (CUDA_edges<T>& next_layer_edges, CUDA_edges<T>& new_edges, int *key, int blocks){
//---------------------------------------------------------------------------------------------------------------------
  dim3 grid_dim ((_nb_edges + blocks - 1) / blocks);
  _nb_edges = next_layer_edges._nb_edges + new_edges._nb_edges;

  merge_by_key<<<grid_dim,blocks>>> (_edge_source, next_layer_edges._edge_source, new_edges._edge_source, key, next_layer_edges._nb_edges, _nb_edges);
  merge_by_key<<<grid_dim,blocks>>> (_edge_target, next_layer_edges._edge_target, new_edges._edge_target, key, next_layer_edges._nb_edges, _nb_edges);
  merge_by_key<<<grid_dim,blocks>>> (_edge_weight, next_layer_edges._edge_weight, new_edges._edge_weight, key, next_layer_edges._nb_edges, _nb_edges);
  merge<<<grid_dim,blocks>>> (_edge_tag, next_layer_edges._edge_tag, new_edges._edge_tag, next_layer_edges._nb_edges, _nb_edges); // tags of next_layer_edges are already reordered (done in sort)
  merge_by_key<<<grid_dim,blocks>>> (_shared_hit, next_layer_edges._shared_hit, new_edges._shared_hit, key, next_layer_edges._nb_edges, _nb_edges);

  if (new_edges._nb_edges)
    _tag = new_edges._tag;
  else
    _tag = next_layer_edges._tag;
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
void CUDA_edges<T>::find_duplicate_sources (CUDA_mask& duplicate_sources,
                                             CUDA_mask& known_sources, int blocks){
//---------------------------------------------------------------------------------------------------------------------
  int blocks_2D = sqrt (blocks);
//  int blocks_2D = blocks;
  dim3 block_dim_2D (blocks_2D, blocks_2D);
  int block_dim = (_nb_edges + blocks_2D - 1) / blocks_2D;
  dim3 grid_dim_2D (block_dim, block_dim);

  find_duplicate_vertices<<<grid_dim_2D,block_dim_2D>>> (duplicate_sources.cuda_mask(),
                                                    known_sources.cuda_mask(), _edge_source, _edge_weight, _nb_edges);
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
void CUDA_edges<T>::untag_duplicate_sources (int blocks){
//---------------------------------------------------------------------------------------------------------------------
  int blocks_2D = sqrt (blocks);
//  int blocks_2D = blocks;
  dim3 block_dim_2D (blocks_2D, blocks_2D);
  int block_dim = (_nb_edges + blocks_2D - 1) / blocks_2D;
  dim3 grid_dim_2D (block_dim, block_dim);

  duplicate_vertices<<<grid_dim_2D,block_dim_2D>>> (_edge_tag, _shared_hit,
                                                    _edge_source, _edge_weight, _nb_edges);
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
void CUDA_edges<T>::untag_duplicate_targets (int blocks){
//---------------------------------------------------------------------------------------------------------------------
  int blocks_2D = sqrt (blocks);
//  int blocks_2D = blocks;
  dim3 block_dim_2D (blocks_2D, blocks_2D);
  int block_dim = (_nb_edges + blocks_2D - 1) / blocks_2D;
  dim3 grid_dim_2D (block_dim, block_dim);

  duplicate_vertices<<<grid_dim_2D,block_dim_2D>>> (_edge_tag, _shared_hit,
                                                    _edge_target, _edge_weight, _nb_edges);
}

/*
//---------------------------------------------------------------------------------------------------------------------
template <class T>
void CUDA_edges<T>::filter_last_edges (CUDA_mask& rev_mask, CUDA_edges<T>& last_edges, int blocks){
//---------------------------------------------------------------------------------------------------------------------
  dim3 grid = (last_edges._nb_edges + blocks - 1) / blocks;
  filter_last_vertices<<<grid,blocks>>>(rev_mask.cuda_mask(),
                                        _edge_source, _edge_target, _nb_edges,
                                        last_edges._edge_source, last_edges._edge_target, last_edges._nb_edges);
}
*/

//---------------------------------------------------------------------------------------------------------------------
template <class T>
void CUDA_edges<T>::filter_last_edges (CUDA_mask& rev_mask, CUDA_edges<T>& last_edges, int blocks){
//---------------------------------------------------------------------------------------------------------------------
  int blocks_2D = sqrt (blocks);
  dim3 block_dim_2D (blocks_2D, blocks_2D);
  int block_dim1 = (_nb_edges + blocks_2D - 1) / blocks_2D;
  int block_dim2 = (last_edges._nb_edges + blocks_2D - 1) / blocks_2D;
  dim3 grid_dim_2D (block_dim1, block_dim2);

  dim3 grid = (_nb_edges + blocks - 1) / blocks;
  bool *is_last_edge;
  cudaMalloc (&is_last_edge, _nb_edges * sizeof (bool));
  cudaMemsetAsync (is_last_edge, 0, _nb_edges * sizeof(bool));
  filter_last_vertices_2D<<<grid_dim_2D,block_dim_2D>>> (is_last_edge,
                                              _edge_source, _edge_target, _nb_edges,
                                              last_edges._edge_source, last_edges._edge_target, last_edges._nb_edges);

  reverse_mask_reverse_filter<<<grid,blocks>>> (rev_mask.cuda_mask(), is_last_edge, _nb_edges);
  cudaFree (is_last_edge);
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
void CUDA_edges<T>::filter_low_scores (CUDA_mask& rev_mask, T score, int blocks){
//---------------------------------------------------------------------------------------------------------------------
  int grid = (_nb_edges + blocks - 1) / blocks;
  score_filter<<<grid,blocks>>> (rev_mask.cuda_mask(), _edge_weight, score, _nb_edges);
}

/*
//------------------------------------------------------------------------------------------------------
template <class T>
void CUDA_edges<T>::filter (CUDA_mask& mask, CUDA_edges<T>& edges, CUDA_edges<T>& ref_edges, int blocks)
//------------------------------------------------------------------------------------------------------
{
  int nb_edges = mask.size();
  int blocks = (nb_edges + blocks_2D - 1) / blocks_2D;
  int block_dim2 = (ref_edges._nb_edges + blocks_2D - 1) / blocks_2D;
  dim3 grid_dim_2D (block_dim1, block_dim2);
  CUDA_mask filter_mask (_nb_edges);

  filter_edges<<<grid_dim_2D,block_dim_2D>>> (filter_mask.cuda_mask(), mask.cuda_mask(), _edge_source, _edge_target, _nb_edges, ref_edges._edge_source, ref_edges._edge_target, ref_edges._nb_edges);
  cudaMemcpy (mask.cuda_mask(), filter_mask.cuda_mask(), _nb_edges * sizeof (bool), cudaMemcpyDeviceToDevice);
}
*/

//---------------------------------------------------------------------------------------------------------------------
template <class T>
void CUDA_edges<T>::features (CUDA_hits<T>& hits, int blocks){
//---------------------------------------------------------------------------------------------------------------------
  dim3 grid_dim = (_nb_edges + blocks - 1) / blocks;
  edge_features<<<grid_dim,blocks>>> (_dR, _dz, _r_phi_slope, _edge_source, _edge_target,
                                      hits.cuda_vertices_sum(), hits.cuda_R(), hits.cuda_z(), _phi_slope,
                                      std::numbers::pi_v<T>, _nb_edges);
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
void CUDA_edges<T>::kill (){
//---------------------------------------------------------------------------------------------------------------------
  if (_nb_edges) {
    if (_tag != -1) cudaFree (_edge_weight);
    if (_ACORN) {
      cudaFree (_ACORN_edge_target);
      cudaFree (_ACORN_edge_source);
    }
    if (_decorations) {
      cudaFree (_r_phi_slope);
      cudaFree (_phi_slope);
      cudaFree (_dphi);
      cudaFree (_deta);
      cudaFree (_dz);
      cudaFree (_dR);
    }
    cudaFree (_edge_target);
    cudaFree (_edge_source);
    if (_tag != -1) cudaFree (_edge_tag);
  }
  _ACORN = _decorations = false;
  _nb_edges = 0;
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
void CUDA_edges<T>::DeviceToHost (edges<T>& e){
//---------------------------------------------------------------------------------------------------------------------
  e = edges<T> (_nb_edges);
  assert (_decorations);
  cudaMemcpyAsync (e.edge_source().data(), _edge_source, _nb_edges * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpyAsync (e.edge_target().data(), _edge_target, _nb_edges * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpyAsync (e.dR().data(), _dR, _nb_edges * sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpyAsync (e.dz().data(), _dz, _nb_edges * sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpyAsync (e.deta().data(), _deta, _nb_edges * sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpyAsync (e.dphi().data(), _dphi, _nb_edges * sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpyAsync (e.phi_slope().data(), _phi_slope, _nb_edges * sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpyAsync (e.r_phi_slope().data(), _r_phi_slope, _nb_edges * sizeof(T), cudaMemcpyDeviceToHost);
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
int* CUDA_edges<T>::ACORN_source (){
//---------------------------------------------------------------------------------------------------------------------
  assert (_ACORN);
  return _ACORN_edge_source;
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
int* CUDA_edges<T>::ACORN_target (){
//---------------------------------------------------------------------------------------------------------------------
  assert (_ACORN);
  return _ACORN_edge_target;
}

//---------------------------------------------------------------------------------------------------------------------
template <class Tf> std::ostream& operator << (std::ostream& s, const CUDA_edges<Tf>& edges){
//---------------------------------------------------------------------------------------------------------------------
  assert (edges._nb_edges);

  print_vector<<<1,1>>> (edges._edge_source, edges._nb_edges);

  return s;
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
void CUDA_edges<T>::save (const std::string& path, const std::string& filename){
//---------------------------------------------------------------------------------------------------------------------
  std::string full_filename = path + "/" + filename;
  std::ofstream outfile;
  outfile.open(full_filename.c_str(), std::ios_base::app);

  std::vector<int> source(_nb_edges), target(_nb_edges);
  std::vector<T> weight (_nb_edges);
  cudaMemcpyAsync (source.data(), _edge_source, _nb_edges * sizeof (int), cudaMemcpyDeviceToHost);
  cudaMemcpyAsync (target.data(), _edge_target, _nb_edges * sizeof (int), cudaMemcpyDeviceToHost);
  cudaMemcpyAsync (weight.data(), _edge_weight, _nb_edges * sizeof (T), cudaMemcpyDeviceToHost);

  for (int i=0; i<_nb_edges; i++)
    outfile << source[i] << "," << target[i] << "," << weight[i] << std::endl;

  outfile.close();
}

//delete this
//---------------------------------------------------------------------------------------------------------------------
template <class T>
void CUDA_edges<T>::connect_edges (CUDA_edges<T>& next_layer, int blocks){
//---------------------------------------------------------------------------------------------------------------------
  _tag = next_layer._tag;

  int blocks_2D = sqrt (blocks);
//  int blocks_2D = blocks;
  dim3 block_dim_2D (blocks_2D, blocks_2D);
  int layer_size = (_nb_edges + blocks_2D - 1) / blocks_2D;
  int next_layer_size = (next_layer._nb_edges + blocks_2D - 1) / blocks_2D;
  dim3 grid_dim_2D (layer_size, next_layer_size);

  ::connect_edges<<<grid_dim_2D,block_dim_2D>>> (
                 _nb_edges, _edge_target, _edge_weight, _edge_tag, 
                 next_layer._nb_edges, next_layer._edge_source, next_layer._edge_weight, next_layer._edge_tag);
}
