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

#include "MMG/module_map_doublet.cuh"


//=====GPU functions for cuda module map doublet=======================================================================

//=====Private methods for cuda module map doublet=====================================================================

//---------------------------------------------------------------------------------------------------------------------
template <class T>
void CUDA_module_map_doublet<T>::allocate_doublets (){
//---------------------------------------------------------------------------------------------------------------------
  _memory_allocation = true;

  cudaMalloc (&_cuda_module1, _nb_doublets * sizeof(int));
  cudaMalloc (&_cuda_module2, _nb_doublets * sizeof(int));
  if (_modulesID.size())
    cudaMalloc (&_cuda_modulesID, _modulesID.size() * sizeof(uint64_t));
  cudaMalloc (&_cuda_z0_min, _nb_doublets * sizeof(T));
  cudaMalloc (&_cuda_z0_max, _nb_doublets * sizeof(T));
  cudaMalloc (&_cuda_dphi_min, _nb_doublets * sizeof(T));
  cudaMalloc (&_cuda_dphi_max, _nb_doublets * sizeof(T));
  cudaMalloc (&_cuda_phi_slope_min, _nb_doublets * sizeof(T));
  cudaMalloc (&_cuda_phi_slope_max, _nb_doublets * sizeof(T));
  cudaMalloc (&_cuda_deta_min, _nb_doublets * sizeof(T));
  cudaMalloc (&_cuda_deta_max, _nb_doublets * sizeof(T));
}

//=====Public methods for cuda module map doublet======================================================================


//---------------------------------------------------------------------------------------------------------------------
template <class T>
template <class Tf>
CUDA_module_map_doublet<T>::CUDA_module_map_doublet (module_map_triplet<Tf>& MMT){
//---------------------------------------------------------------------------------------------------------------------
  _memory_allocation = false;
  init (MMT);
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
CUDA_module_map_doublet<T>::~CUDA_module_map_doublet (){
//---------------------------------------------------------------------------------------------------------------------
  if (_memory_allocation) {
    cudaFree (_cuda_deta_max);
    cudaFree (_cuda_deta_min);
    cudaFree (_cuda_phi_slope_max);
    cudaFree (_cuda_phi_slope_min);
    cudaFree (_cuda_dphi_max);
    cudaFree (_cuda_dphi_min);
    cudaFree (_cuda_z0_max);
    cudaFree (_cuda_z0_min);
    if (_module_map.size())
      cudaFree (_cuda_modulesID);
    cudaFree (_cuda_module2);
    cudaFree (_cuda_module1);
    _memory_allocation = false;
  }
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
template <class Tf>
void CUDA_module_map_doublet<T>::init (module_map_triplet<Tf>& MMT){
//---------------------------------------------------------------------------------------------------------------------
  _nb_doublets = MMT.map_doublet().size();
  _module_map = MMT.module_map();

  for (std::_Rb_tree_iterator<std::pair<const uint64_t, int>> it = _module_map.begin(); it != _module_map.end(); it++)
    _modulesID.push_back (it->first);

  for (std::pair<std::vector<uint64_t>, int> pairs : MMT.map_pairs ()) {
    _module1.push_back (_module_map.find (pairs.first[0]) -> second);
    _module2.push_back (_module_map.find (pairs.first[1]) -> second);
  }

  for (std::_Rb_tree_iterator<std::pair<const std::vector<uint64_t>, module_doublet<Tf>>> it2 = MMT.map_doublet().begin(); it2 != MMT.map_doublet().end(); it2++) {
    _z0_min.push_back ((T) it2->second.z0_min());
    _z0_max.push_back ((T) it2->second.z0_max());
    _dphi_min.push_back ((T) it2->second.dphi_min());
    _dphi_max.push_back ((T) it2->second.dphi_max());
    _phi_slope_min.push_back ((T) it2->second.phi_slope_min());
    _phi_slope_max.push_back ((T) it2->second.phi_slope_max());
    _deta_min.push_back ((T) it2->second.deta_min());
    _deta_max.push_back ((T) it2->second.deta_max());
  }
}

/*
//---------------------------------------------------------------------------------------------------------------------
template <class T>
void CUDA_module_map_doublet<T>::filter_doublets (const CUDA_mask& hits_mask, int blocks){
//---------------------------------------------------------------------------------------------------------------------
  dim3 grid = (_nb_doublets + blocks - 1) / blocks;
//  filter<<<grid,blocks>>> (_active_doublets.cuda_mask(),
//                           hits_mask.cuda_mask(), _cuda_module1, _cuda_module2, _nb_doublets);
}
*/

//---------------------------------------------------------------------------------------------------------------------
template <class T>
template <class Tf>
void CUDA_module_map_doublet<T>::build_doublets (module_map_triplet<Tf>& MMT, std::string modules){
//---------------------------------------------------------------------------------------------------------------------
  _nb_doublets = MMT.size();
  _module_map = MMT.module_map();
  module_doublet<Tf> *it2;

  for (std::_Rb_tree_iterator<std::pair<const std::vector<uint64_t>, module_triplet<Tf>>> it3 = MMT.map_triplet().begin(); it3 != MMT.map_triplet().end(); it3++) {
    if (modules == "12") {
      _module1.push_back (_module_map.find (it3->first[0]) -> second);
      _module2.push_back (_module_map.find (it3->first[1]) -> second);
      it2 = &it3->second.modules12();
    }

    if (modules == "23") {
      _module1.push_back (_module_map.find (it3->first[1]) -> second);
      _module2.push_back (_module_map.find (it3->first[2]) -> second);
      it2 = &it3->second.modules23();
    }

    _z0_min.push_back ((T) it2->z0_min());
    _z0_max.push_back ((T) it2->z0_max());
    _dphi_min.push_back ((T) it2->dphi_min());
    _dphi_max.push_back ((T) it2->dphi_max());
    _phi_slope_min.push_back ((T) it2->phi_slope_min());
    _phi_slope_max.push_back ((T) it2->phi_slope_max());
    _deta_min.push_back ((T) it2->deta_min());
    _deta_max.push_back ((T) it2->deta_max());
  }
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
void CUDA_module_map_doublet<T>::HostToDevice (){
//---------------------------------------------------------------------------------------------------------------------
  allocate_doublets();
  // send module map doublets data to GPU
  if (_modulesID.size())
    cudaMemcpyAsync (_cuda_modulesID, _modulesID.data(), _modulesID.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);
  cudaMemcpyAsync (_cuda_module1, _module1.data(), _nb_doublets * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpyAsync (_cuda_module2, _module2.data(), _nb_doublets * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpyAsync (_cuda_z0_min, _z0_min.data(), _nb_doublets * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpyAsync (_cuda_z0_max, _z0_max.data(), _nb_doublets * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpyAsync (_cuda_dphi_min, _dphi_min.data(), _nb_doublets * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpyAsync (_cuda_dphi_max, _dphi_max.data(), _nb_doublets * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpyAsync (_cuda_phi_slope_min, _phi_slope_min.data(), _nb_doublets * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpyAsync (_cuda_phi_slope_max, _phi_slope_max.data(), _nb_doublets * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpyAsync (_cuda_deta_min, _deta_min.data(), _nb_doublets * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpyAsync (_cuda_deta_max, _deta_max.data(), _nb_doublets * sizeof(T), cudaMemcpyHostToDevice);
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
void CUDA_module_map_doublet<T>::HostToDevice (cudaStream_t& stream){
//---------------------------------------------------------------------------------------------------------------------
  allocate_doublets();
  // send module map doublets data to GPU
  if (_modulesID.size())
    cudaMemcpyAsync (_cuda_modulesID, _modulesID.data(), _modulesID.size() * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync (_cuda_module1, _module1.data(), _nb_doublets * sizeof(int), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync (_cuda_module2, _module2.data(), _nb_doublets * sizeof(int), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync (_cuda_z0_min, _z0_min.data(), _nb_doublets * sizeof(T), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync (_cuda_z0_max, _z0_max.data(), _nb_doublets * sizeof(T), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync (_cuda_dphi_min, _dphi_min.data(), _nb_doublets * sizeof(T), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync (_cuda_dphi_max, _dphi_max.data(), _nb_doublets * sizeof(T), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync (_cuda_phi_slope_min, _phi_slope_min.data(), _nb_doublets * sizeof(T), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync (_cuda_phi_slope_max, _phi_slope_max.data(), _nb_doublets * sizeof(T), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync (_cuda_deta_min, _deta_min.data(), _nb_doublets * sizeof(T), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync (_cuda_deta_max, _deta_max.data(), _nb_doublets * sizeof(T), cudaMemcpyHostToDevice, stream);
}
