/***********************************************************************
 * Module Map Graph library
 * Track reconstruction for HEP
 * copyright © 2024,2025 COLLARD Christophe
 * Laboratoire des 2 Infinis - Toulouse - France (L2IT)
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 **********************************************************************/

#include "MMG/nodes.cuh"

//=====GPU functions for nodes===============================================


//=====CPU functions for nodes===============================================

//---------------------------
template <class T>
CUDA_nodes<T>::~CUDA_nodes ()
//---------------------------
{
  if (_nb_nodes) {
    cudaFree (_cuda_phi);
    cudaFree (_cuda_eta);
    cudaFree (_cuda_z);
    cudaFree (_cuda_R);
    cudaFree (_cuda_hit_id);
  }
}

//--------------------------------------------------------------
template <class T>
void CUDA_nodes<T>::reduce (CUDA_hits<T>& cuda_hits, int blocks)
//--------------------------------------------------------------
{
  assert (!_nb_nodes);
  _nb_nodes = cuda_hits.cuda_vertices_scan();
  int nb_hits = cuda_hits.size();

  cudaMalloc (&_cuda_hit_id, _nb_nodes * sizeof(uint64_t));
  cudaMalloc (&_cuda_R, _nb_nodes * sizeof(T));
  cudaMalloc (&_cuda_z, _nb_nodes * sizeof(T));
  cudaMalloc (&_cuda_eta, _nb_nodes * sizeof(T));
  cudaMalloc (&_cuda_phi, _nb_nodes * sizeof(T));
  cudaMalloc (&_cuda_ACORN_positions, _nb_nodes * sizeof(int));

  int grid_dim = ((nb_hits + blocks - 1) / blocks);
  compact_stream<<<grid_dim,blocks>>> (_cuda_hit_id, cuda_hits.cuda_hit_id(), cuda_hits.cuda_vertices(), cuda_hits.cuda_vertices_sum(), nb_hits);
  compact_stream<<<grid_dim,blocks>>> (_cuda_R, cuda_hits.cuda_R(), cuda_hits.cuda_vertices(), cuda_hits.cuda_vertices_sum(), nb_hits);
  compact_stream<<<grid_dim,blocks>>> (_cuda_z, cuda_hits.cuda_z(), cuda_hits.cuda_vertices(), cuda_hits.cuda_vertices_sum(), nb_hits);
  compact_stream<<<grid_dim,blocks>>> (_cuda_eta, cuda_hits.cuda_eta(), cuda_hits.cuda_vertices(), cuda_hits.cuda_vertices_sum(), nb_hits);
  compact_stream<<<grid_dim,blocks>>> (_cuda_phi, cuda_hits.cuda_phi(), cuda_hits.cuda_vertices(), cuda_hits.cuda_vertices_sum(), nb_hits);
  if (cuda_hits.ACORN())
    compact_stream<<<grid_dim,blocks>>> (_cuda_ACORN_positions, cuda_hits.cuda_ACORN_positions(), cuda_hits.cuda_vertices(), cuda_hits.cuda_vertices_sum(), nb_hits);
}

//-------------------------
template <class T>
void CUDA_nodes<T>::kill ()
//-------------------------
{
  if (_nb_nodes) {
    cudaFree (_cuda_ACORN_positions);
    cudaFree (_cuda_phi);
    cudaFree (_cuda_eta);
    cudaFree (_cuda_z);
    cudaFree (_cuda_R);
    cudaFree (_cuda_hit_id);
  }
  _nb_nodes = 0;
}

//--------------------------------------------
template <class T>
void CUDA_nodes<T>::DeviceToHost (nodes<T>& n)
//--------------------------------------------
{
  n = nodes<T> (_nb_nodes);
  cudaMemcpy (n.hitID().data(), _cuda_hit_id, _nb_nodes * sizeof(uint64_t), cudaMemcpyDeviceToHost);
  cudaMemcpy (n.R().data(), _cuda_R, _nb_nodes * sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpy (n.z().data(), _cuda_z, _nb_nodes * sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpy (n.eta().data(), _cuda_eta, _nb_nodes * sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpy (n.phi().data(), _cuda_phi, _nb_nodes * sizeof(T), cudaMemcpyDeviceToHost);
}
