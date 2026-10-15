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

#include "MMG/module_map_triplet.cuh"

//=====GPU functions for cuda module map triplet=======================================================================

//=====Private methods for cuda module map triplet=====================================================================

//---------------------------------------------------------------------------------------------------------------------
template <class T>
void CUDA_module_map_triplet<T>::allocate_triplets ()
//---------------------------------------------------------------------------------------------------------------------
{
  cudaMalloc (&_cuda_diff_dydx_min, _nb_triplets * sizeof(T));
  cudaMalloc (&_cuda_diff_dydx_max, _nb_triplets * sizeof(T));
  cudaMalloc (&_cuda_diff_dzdr_min, _nb_triplets * sizeof(T));
  cudaMalloc (&_cuda_diff_dzdr_max, _nb_triplets * sizeof(T));
  cudaMalloc (&_cuda_MD12_map, _nb_triplets * sizeof(int));
  cudaMalloc (&_cuda_MD23_map, _nb_triplets * sizeof(int));
}


//=====Public methods for cuda module map triplet======================================================================


//---------------------------------------------------------------------------------------------------------------------
template <class T>
template<class Tf>
CUDA_module_map_triplet<T>::CUDA_module_map_triplet (module_map_triplet<Tf>& MMT)
//---------------------------------------------------------------------------------------------------------------------
{
  init (MMT);
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
CUDA_module_map_triplet<T>::~CUDA_module_map_triplet ()
//---------------------------------------------------------------------------------------------------------------------
{
  cudaFree (_cuda_MD23_map);
  cudaFree (_cuda_MD12_map);
  cudaFree (_cuda_diff_dzdr_max);
  cudaFree (_cuda_diff_dzdr_min);
  cudaFree (_cuda_diff_dydx_max);
  cudaFree (_cuda_diff_dydx_min);
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
template <class Tf>
void CUDA_module_map_triplet<T>::init (module_map_triplet<Tf>& MMT)
//---------------------------------------------------------------------------------------------------------------------
{
  _nb_triplets = MMT.map_triplet().size();

  _MD12.build_doublets (MMT, "12");
  _MD23.build_doublets (MMT, "23");

  for (std::_Rb_tree_iterator<std::pair<const std::vector<uint64_t>, module_triplet<Tf>>> it3 = MMT.map_triplet().begin(); it3 != MMT.map_triplet().end(); it3++) {
    _MD12_map.push_back (MMT.map_pairs().find (std::vector<uint64_t>{it3->first[0], it3->first[1]}) -> second);
    _MD23_map.push_back (MMT.map_pairs().find (std::vector<uint64_t>{it3->first[1], it3->first[2]}) -> second);
    _diff_dydx_min.push_back ((T) it3->second.diff_dydx_min());
    _diff_dydx_max.push_back ((T) it3->second.diff_dydx_max());
    _diff_dzdr_min.push_back ((T) it3->second.diff_dzdr_min());
    _diff_dzdr_max.push_back ((T) it3->second.diff_dzdr_max());
    _occurence.push_back (it3->second.occurence());
  }
}
//---------------------------------------------------------------------------------------------------------------------
template <class T>
void CUDA_module_map_triplet<T>::HostToDevice ()
//---------------------------------------------------------------------------------------------------------------------
{
  allocate_triplets();
  // send module map doublets data to GPU
  _MD12.HostToDevice();
  _MD23.HostToDevice();
  cudaMemcpyAsync (_cuda_MD12_map, _MD12_map.data(), _nb_triplets * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpyAsync (_cuda_MD23_map, _MD23_map.data(), _nb_triplets * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpyAsync (_cuda_diff_dydx_min, _diff_dydx_min.data(), _nb_triplets * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpyAsync (_cuda_diff_dydx_max, _diff_dydx_max.data(), _nb_triplets * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpyAsync (_cuda_diff_dzdr_min, _diff_dzdr_min.data(), _nb_triplets * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpyAsync (_cuda_diff_dzdr_max, _diff_dzdr_max.data(), _nb_triplets * sizeof(T), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize ();
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
void CUDA_module_map_triplet<T>::HostToDevice (cudaStream_t& stream)
//---------------------------------------------------------------------------------------------------------------------
{
  allocate_triplets();
  // send module map doublets data to GPU
  _MD12.HostToDevice();
  _MD23.HostToDevice();
  cudaMemcpyAsync (_cuda_MD12_map, _MD12_map.data(), _nb_triplets * sizeof(int), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync (_cuda_MD23_map, _MD23_map.data(), _nb_triplets * sizeof(int), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync (_cuda_diff_dydx_min, _diff_dydx_min.data(), _nb_triplets*sizeof(T), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync (_cuda_diff_dydx_max, _diff_dydx_max.data(), _nb_triplets*sizeof(T), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync (_cuda_diff_dzdr_min, _diff_dzdr_min.data(), _nb_triplets*sizeof(T), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync (_cuda_diff_dzdr_max, _diff_dzdr_max.data(), _nb_triplets*sizeof(T), cudaMemcpyHostToDevice, stream);
  cudaDeviceSynchronize ();
}
