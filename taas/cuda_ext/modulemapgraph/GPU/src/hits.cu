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

#include "MMG/hits.cuh"

//=====GPU functions for CUDA hits=====================================================================================

//-----GPU kernels for count_hits_by_module----------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------------------------
__global__ void module_nb_hits (int *module_nb_hits, bool *limit,
                                uint64_t *hit_module, int nb_hits){
//---------------------------------------------------------------------------------------------------------------------
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nb_hits) return;

  if (hit_module[i] != hit_module[i+1] || i == nb_hits-1){
    module_nb_hits[i] = i+1;
    limit[i] = true;
  }
}

//---------------------------------------------------------------------------------------------------------------------
__global__ void expand_nb_hits (int *module_nb_hits, bool *module_flags,
                                int *compressed_module_nb_hits, uint64_t *hit_moduleID, uint64_t *moduleID,
                                int nb_compressed_modules){
//---------------------------------------------------------------------------------------------------------------------
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nb_compressed_modules) return;

  int nhits = compressed_module_nb_hits[i];
  int k = i;
  uint64_t ID = hit_moduleID[i];
  while (ID != moduleID[k]) k++;
  module_nb_hits[k+1] = nhits;
  module_flags[k] = true;
}

//-----GPU kernels for classify_hits_on_modules------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------------------------
__global__ void is_module_map_hit (bool *mask, int *hit_indices,
                                   const uint64_t *MM_ID, int MMsize, const uint64_t *hit_ID, int nb_hits){
//---------------------------------------------------------------------------------------------------------------------
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= nb_hits) return;

  int j;
  findFirstWithBisect (0, MMsize, hit_ID[i], j, MM_ID);
  mask[i] = true;
  hit_indices[i] = j;

/*
  for (int j=0; j<MMsize; j++)
    if (MM_ID[j] == hit_ID[i]) {
      mask[i] = true;
      hit_indices[i] = j;
      assert (j==k);
      return;
    }
*/
}

//---------------------------------------------------------------------------------------------------------------------
__global__ void is_module_map_hit (int *hit_indices,
                                        const uint64_t *MM_ID, int MMsize, const uint64_t *hit_ID, int nb_hits){
//---------------------------------------------------------------------------------------------------------------------
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nb_hits) return;

  int j;
  findFirstWithBisect (0, MMsize, hit_ID[i], j, MM_ID);
  hit_indices[i] = j;
}

//---------------------------------------------------------------------------------------------------------------------
__global__ void which_module_map (int *hit_indices,
                                  uint64_t *MM_ID, int MMsize, uint64_t *hit_ID, int nb_hits){
//---------------------------------------------------------------------------------------------------------------------
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= nb_hits || j >= MMsize) return;

  if (MM_ID[j] == hit_ID[i])
    hit_indices[i] = j;
}

//---------------------------------------------------------------------------------------------------------------------
__global__ void kill_duplicate_hitID (bool *mask, int *nb_duplicates,
                                      const int* nb_hits, const uint64_t *hit_ID, int nb_modules){
//---------------------------------------------------------------------------------------------------------------------
  // general procedure if hits are not arranged by hit ID
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nb_modules) return;

  int limit = nb_hits[i+1];
  int nb=0;

  for (int j=nb_hits[i]; j<limit; j++)
    for (int k=j+1; k<limit; k++)
      if (hit_ID[k] == hit_ID[j]){
        nb += (1-mask[k]);
        mask[k] = true;
      }

  nb_duplicates[i] = nb;
}

//---------------------------------------------------------------------------------------------------------------------
__global__ void kill_duplicate_hitID (bool *mask,
                                      uint64_t *hit_ID, int nb_hits){
//---------------------------------------------------------------------------------------------------------------------
  // general procedure if hits are not arranged by hit ID
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = i + 1 + blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= nb_hits || j >= nb_hits) return;

  if (hit_ID[i] == hit_ID[j])
    mask[j] = false;
}

//---------------------------------------------------------------------------------------------------------------------
__global__ void kill_duplicate_classified_hitID (bool *mask,
                                                 uint64_t *hit_ID, int nb_hits){
//---------------------------------------------------------------------------------------------------------------------
  // assume hits are classified by hitID (as generated by the dumper)
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nb_hits-1) return;
  if (!mask[i] || !mask[i+1] || hit_ID[i+1] != hit_ID[i]) return;

  mask[i+1] = false;
}

//---------------------------------------------------------------------------------------------------------------------
__global__ void module_nb_hits (int *module_nb_hits,
                                int *module, int nb_hits){
//---------------------------------------------------------------------------------------------------------------------
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nb_hits) return;

  if (module[i] != module[i+1] || i == nb_hits-1)
    module_nb_hits[module[i]+1] = i+1;
}

//---------------------------------------------------------------------------------------------------------------------
__global__ void modules_hits (int *module_nb_hits, //bool *active_module,
                              int *module, int nb_hits){
//---------------------------------------------------------------------------------------------------------------------
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nb_hits) return;

  if (module[i] != module[i+1] || i == nb_hits-1){
    module_nb_hits[module[i]+1] = i+1;
//    active_module[module[i]] = true;
  }
}

//---------------------------------------------------------------------------------------------------------------------
__global__ void fill_empty_module_indices (int *module_nb_hits,
                                           int nb_modules){
//---------------------------------------------------------------------------------------------------------------------
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nb_modules) return;

  int value = 0;
  if (!module_nb_hits[i+1]) {
    for (int n=i; n>0 && !value; n--) // look for previous not 0 element (for loop due to parallel fill)
      value = module_nb_hits[n]; // starts at i=1 (keep value 0 at i=0)
    module_nb_hits[i+1] = value;
  }
}

/*
//-------------------------------------------------------------------------------------------------
template <class T>
__global__ void hits_constants (T* cuda_R, T* cuda_eta, T* cuda_phi, T* x, T* y, T* z, int nb_hits)
//-------------------------------------------------------------------------------------------------
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nb_hits) return;

  data_type2 xi = x[i];
  data_type2 yi = y[i];
  data_type2 zi = z[i];
//  data_type2 R = std::sqrt (pow (xi, 2) + pow (yi, 2));
  data_type2 R2 = pow (xi, 2) + pow (yi, 2);
  cuda_R[i] = std::sqrt (R2);
//  data_type2 r3_inv = std::sqrt (1 + pow (zi / (xi+yi), 2));

//  data_type2 r3 = std::sqrt (pow(R, 2) + pow(zi,2));
  data_type2 r3 = std::sqrt (R2 + pow(zi, 2)) ;
  data_type2 theta = (data_type2) 0.5 * acos (zi / r3);
  cuda_eta[i] = -log (tan (theta));

  cuda_phi[i] = atan2 (yi, xi);
}
*/

//---------------------------------------------------------------------------------------------------------------------
template <class T>
__global__ void hits_constants (T* cuda_R, T* cuda_eta, T* cuda_phi,
                                T* x, T* y, T* z, int nb_hits){
//---------------------------------------------------------------------------------------------------------------------
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nb_hits) return;

  T xi = x[i];
  T yi = y[i];
  T zi = z[i];
  T R2 = pow (xi, 2) + pow (yi, 2);
  cuda_R[i] = std::sqrt (R2);

  T r3 = std::sqrt (R2 + pow(zi, 2)) ;
//  cuda_eta[i] = -log (tan (0.5 * acos (zi / r3)));
  T theta = zi/r3;
  cuda_eta[i] = 0.5 * log ((1 + theta) / (1 - theta));

  cuda_phi[i] = atan2 (yi, xi);
}


//=====Private methods for hits========================================================================================

//=====Public methods for hits=========================================================================================


//---------------------------------------------------------------------------------------------------------------------
template <class T>
CUDA_hits<T>::CUDA_hits (){
//---------------------------------------------------------------------------------------------------------------------
  _size = _position = _max_size = 0;
  _gpu_mem_allocated = _ACORN = false;
  _cuda_hit_id = _cuda_module_id = nullptr;
  _cuda_x = _cuda_y = _cuda_z = nullptr;
  _cuda_R = _cuda_eta = _cuda_phi = nullptr;
  _cuda_hit_indice = _ACORN_positions = nullptr;
  _extra_features = false;
}

/*
//---------------------------------------------------------------------------------------------------------------------
template <class T>
CUDA_hits<T>::CUDA_hits (CUDA_module_map_doublet<T>& module_map, const uint64_t* hitID, const uint64_t* moduleID,
                         const T* x, const T* y, const T* z, int size, int blocks, int max_blocks){
//---------------------------------------------------------------------------------------------------------------------
  _size = _position = _max_size = 0;
  _gpu_mem_allocated = _ACORN = false;
  classify_hits_on_modules (module_map, hitID, moduleID, x, y, z, size, blocks, max_blocks);
}
*/

//---------------------------------------------------------------------------------------------------------------------
template <class T>
CUDA_hits<T>::CUDA_hits (CUDA_module_map_doublet<T>& module_map, const uint64_t* hitID, const uint64_t* moduleID,
                        const T* x, const T* y, const T* z, int size, int blocks, int max_blocks,
                        const std::string *hardware, const int *barrel_endcap,
                        const int *layer_disk, const int *eta_module, const int *phi_module){
//---------------------------------------------------------------------------------------------------------------------
  _size = _position = _max_size = 0;
  _gpu_mem_allocated = _ACORN = false;
  _cuda_hit_id = _cuda_module_id = nullptr;
  _cuda_x = _cuda_y = _cuda_z = nullptr;
  _cuda_R = _cuda_eta = _cuda_phi = nullptr;
  _cuda_hit_indice = _ACORN_positions = nullptr;
  _extra_features = (hardware != nullptr);
  classify_hits_on_modules (module_map, hitID, moduleID, x, y, z, size, blocks, max_blocks);
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
CUDA_hits<T>::~CUDA_hits (){
//---------------------------------------------------------------------------------------------------------------------
  if (_size && _gpu_mem_allocated) {
    if (_ACORN)
      cudaFree (_ACORN_positions);
    if (_cuda_hit_indice) cudaFree (_cuda_hit_indice);
    if (_cuda_phi) cudaFree (_cuda_phi);
    if (_cuda_eta) cudaFree (_cuda_eta);
    if (_cuda_R) cudaFree (_cuda_R);
    if (_cuda_z) cudaFree (_cuda_z);
    if (_cuda_y) cudaFree (_cuda_y);
    if (_cuda_x) cudaFree (_cuda_x);
    if (_cuda_module_id) cudaFree (_cuda_module_id);
    if (_cuda_hit_id) cudaFree (_cuda_hit_id);
  }

  if (_extra_features){
    cudaFree (_phi_module);
    cudaFree (_eta_module);
    cudaFree (_layer_disk);
    cudaFree (_barrel_endcap);
    cudaFree (_hardware);
    cudaFree (_module_z);
    cudaFree (_module_y);
    cudaFree (_module_x);
    _extra_features = false;
  }

  _size = _position = 0;
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
void CUDA_hits<T>::count_hits_by_module (CUDA_module_map_doublet<T>& module_map,
                                            const uint64_t* hitID, const uint64_t* moduleID,
                                            const T* x, const T* y, const T* z, int size, int blocks, int max_blocks){
//---------------------------------------------------------------------------------------------------------------------
  clock_t start, end;
  clock_t start_cp2gpu, end_cp2gpu;
  clock_t start_hits_by_module, end_hits_by_module;
  clock_t start_kill_duplicate_hits, end_kill_duplicate_hits;
  clock_t start_features_allocation, end_features_allocation;

#ifdef MMG_ENABLE_TIMING
  cudaDeviceSynchronize();
  start = start_cp2gpu = clock();
  std::chrono::system_clock::time_point start_real = std::chrono::system_clock::now();
#endif

  _position = 0;
  _blocks = blocks;
  _size = size;

  //-------------------------------- get direct access from gpu in wrapped code

  uint64_t *cuda_module_id;

  cudaMalloc (&_cuda_hit_id, _size * sizeof(uint64_t));
  cudaMalloc (&cuda_module_id, _size * sizeof(uint64_t));
  cudaMalloc (&_cuda_x, _size * sizeof(T));
  cudaMalloc (&_cuda_y, _size * sizeof(T));
  cudaMalloc (&_cuda_z, _size * sizeof(T));

  cudaMemcpyAsync (_cuda_hit_id, hitID, size * sizeof(uint64_t), cudaMemcpyHostToDevice);
  cudaMemcpyAsync (cuda_module_id, moduleID, size * sizeof(uint64_t), cudaMemcpyHostToDevice);
  cudaMemcpyAsync (_cuda_x, x, _size * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpyAsync (_cuda_y, y, _size * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpyAsync (_cuda_z, z, _size * sizeof(T), cudaMemcpyHostToDevice);

// ------------------end

  int nb_modules = module_map.module_map().size();
  cudaMalloc (&_cuda_hit_indice, (nb_modules+1) * sizeof(int));
  cudaMemsetAsync (_cuda_hit_indice, 0, (nb_modules+1) * sizeof(int));

  _ACORN = true;
  cudaMalloc (&_ACORN_positions, _size * sizeof(int));
  dim3 grid_dim = (_size + _blocks - 1) / _blocks;
  init_vector<<<grid_dim,_blocks>>> (_ACORN_positions, size);

#ifdef MMG_ENABLE_TIMING
  cudaDeviceSynchronize();
  end_cp2gpu = start_hits_by_module = clock();
#endif


  //------------------------------------------------------------------------------------
  // get nb of hits in MM (some hits may not be part of the MM or can be duplicate hits)
  //------------------------------------------------------------------------------------

  //-----------------------------------------------------------------
  // build vector with nb of hits by module (no matter the module ID)
  //-----------------------------------------------------------------

  int *cuda_hit_indice;
  cudaMalloc (&cuda_hit_indice, _size * sizeof(int));
  cudaMemsetAsync (cuda_hit_indice, 0, _size * sizeof(int));
  CUDA_mask hits_indice_mask (_size);

  module_nb_hits<<<grid_dim,_blocks>>> (cuda_hit_indice, hits_indice_mask.cuda_mask(),
                                        cuda_module_id, _size);

  int n = hits_indice_mask.scan (_blocks);
  compact_stream (&cuda_hit_indice, hits_indice_mask, n, _blocks);
  compact_stream (&cuda_module_id, hits_indice_mask, n, _blocks);

  dim3 grid = (n + _blocks - 1) / _blocks;
  CUDA_mask module_list (nb_modules);
  expand_nb_hits<<<grid,_blocks>>> (_cuda_hit_indice, module_list.cuda_mask(),
                                    cuda_hit_indice, cuda_module_id, module_map.cuda_modules_ID(), n);

  // extract inactive modules
  int nb_active_modules = module_list.scan(_blocks);
std::cout << "nb active modules = " << nb_active_modules << reset;
  cudaFree (cuda_hit_indice);
  cudaFree (cuda_module_id);

#ifdef MMG_ENABLE_TIMING
  cudaDeviceSynchronize();
  end_hits_by_module = clock();
  std::cout << blue << "cpu time for hits by module: " << (double)(end_hits_by_module - start_hits_by_module) / CLOCKS_PER_SEC << reset;
#endif
  std::cout << "nb modules = " << n << " / " << nb_modules << reset;

  // -------------------------
  // look for inactive modules
  // -------------------------

  grid_dim = (nb_modules + _blocks - 1) / _blocks;
  fill_empty_module_indices<<<grid_dim,_blocks>>> (_cuda_hit_indice, nb_modules);

#ifdef MMG_ENABLE_TIMING
  cudaDeviceSynchronize();
  start_kill_duplicate_hits = clock();
#endif

  //---------------------------------
  // kill duplicate hit IDs by module (low combinatory)
  //---------------------------------

int old_size = _size;

  CUDA_mask cuda_duplicate_hits (_size);
  int *cuda_nb_hits;
  cudaMalloc (&cuda_nb_hits, nb_modules * sizeof(int));
  kill_duplicate_hitID<<<grid_dim,_blocks>>> (cuda_duplicate_hits.cuda_mask(), cuda_nb_hits,
                                              _cuda_hit_indice, _cuda_hit_id, nb_modules);

  int *cuda_nb_hits_sum;
  cudaMalloc (&cuda_nb_hits_sum, (nb_modules+1) * sizeof(int));
  int nb_duplicate_hits = scan (cuda_nb_hits_sum, cuda_nb_hits, _blocks, nb_modules);

  if (nb_duplicate_hits){
    std::cout << red << "duplicate hits found" << reset;
    _size -= cuda_duplicate_hits.scan (_blocks);
    vector_subtraction<<<grid_dim,_blocks>>> (_cuda_hit_indice, cuda_nb_hits_sum, nb_modules+1);
    compact_stream_reverse_mask (&_cuda_hit_id, cuda_duplicate_hits, _size, _blocks);
    compact_stream_reverse_mask (&_cuda_x, cuda_duplicate_hits, _size, _blocks);
    compact_stream_reverse_mask (&_cuda_y, cuda_duplicate_hits, _size, _blocks);
    compact_stream_reverse_mask (&_cuda_z, cuda_duplicate_hits, _size, _blocks);

    // ACORN part to get original hit table-----------------------------------------------
    compact_stream_reverse_mask (&_ACORN_positions, cuda_duplicate_hits, _size, _blocks);
    // -----------------------------------------------------------------------------------
  }

  cudaFree (cuda_nb_hits_sum);
  cudaFree (cuda_nb_hits);

#ifdef MMG_ENABLE_TIMING
  cudaDeviceSynchronize();
  end_kill_duplicate_hits = start_features_allocation = clock();
#endif

  _gpu_mem_allocated = _ACORN = true;
  cudaMalloc (&_cuda_R, _size * sizeof(T));
  cudaMalloc (&_cuda_eta, _size * sizeof(T));
  cudaMalloc (&_cuda_phi, _size * sizeof(T));
  _cuda_vertices.init (_size);


#ifdef MMG_ENABLE_TIMING
  cudaDeviceSynchronize();
  std::chrono::system_clock::time_point end_real = std::chrono::system_clock::now();
  end = end_features_allocation = clock ();

  std::cout << magenta << "cpu time for " << size << " hits in event: " << (double)(end - start) / CLOCKS_PER_SEC * 1000 << " ms" << reset;
  std::cout << red << "cpu real time: " <<  std::chrono::duration_cast<std::chrono::microseconds>(end_real - start_real).count() * 0.001 << " ms" << reset;
  std::cout << cyan << "cpu time to copy data to GPU: " << (double)(end_cp2gpu - start_cp2gpu) / CLOCKS_PER_SEC << reset;
  std::cout << cyan << "cpu time for hits by module: " << (double)(end_hits_by_module - start_hits_by_module) / CLOCKS_PER_SEC << reset;
  std::cout << cyan << "cpu time to kill duplicate hits: " << (double)(end_kill_duplicate_hits - start_kill_duplicate_hits) / CLOCKS_PER_SEC << reset;
  std::cout << cyan << "cpu time to allocate memory for features: " << (double)(end_features_allocation - start_features_allocation) / CLOCKS_PER_SEC << reset;
  std::cout << "initial nb elts = " << size << std::endl;
  std::cout << "nb of elts in MM = " << _size << std::endl;
#endif
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
void CUDA_hits<T>::classify_hits_on_modules(CUDA_module_map_doublet<T>& module_map,
                                            const uint64_t* hitID, const uint64_t* moduleID,
                                            const T* x, const T* y, const T* z, int size, int blocks, int max_blocks){
//---------------------------------------------------------------------------------------------------------------------

#ifdef MMG_ENABLE_TIMING
  clock_t start, end;
  clock_t start_cp2gpu, end_cp2gpu;
  clock_t start_hit2module, end_hit2module;
  clock_t start_kill_duplicates, end_kill_duplicates;
  clock_t start_reduction, end_reduction;
  clock_t start_qs, end_qs;
  clock_t start_hits_by_module, end_hits_by_module;
  clock_t start_kill_duplicate_hits, end_kill_duplicate_hits;
#endif

#ifdef MMG_ENABLE_TIMING
  cudaDeviceSynchronize();
  start = start_cp2gpu = clock();
  std::chrono::system_clock::time_point start_real = std::chrono::system_clock::now();
#endif

  _position = 0;
  _blocks = blocks;

  //-------------------------------- get direct access from gpu in wrapped code

  uint64_t *cuda_module_id;

  cudaMalloc (&_cuda_hit_id, size * sizeof(uint64_t));
  cudaMalloc (&cuda_module_id, size * sizeof(uint64_t));
  cudaMalloc (&_cuda_x, size * sizeof(T));
  cudaMalloc (&_cuda_y, size * sizeof(T));
  cudaMalloc (&_cuda_z, size * sizeof(T));

  cudaMemcpyAsync (_cuda_hit_id, hitID, size * sizeof(uint64_t), cudaMemcpyHostToDevice);
  cudaMemcpyAsync (cuda_module_id, moduleID, size * sizeof(uint64_t), cudaMemcpyHostToDevice);
  cudaMemcpyAsync (_cuda_x, x, size * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpyAsync (_cuda_y, y, size * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpyAsync (_cuda_z, z, size * sizeof(T), cudaMemcpyHostToDevice);

#ifdef MMG_ENABLE_TIMING
  cudaDeviceSynchronize();
  end_cp2gpu = start_hit2module = clock();
#endif

// ------------------end

  //----------------------
  // assign hits to module
  //----------------------

  int nb_modules = module_map.module_map().size();

  int *cuda_MM_position;
  cudaMalloc (&cuda_MM_position, size * sizeof(int));

  dim3 grid = (((size + blocks - 1) / blocks));

  // ACORN part to get original hit table-----------------------------------------------
  int *positions;
  cudaMalloc (&positions, size * sizeof(int));
  init_vector<<<grid,_blocks>>> (positions, size);
  // -----------------------------------------------------------------------------------

#ifdef MMG_ENABLE_DEBUG
  CUDA_mask cuda_hits_in_MM (size);
  is_module_map_hit<<<grid,blocks>>> (cuda_hits_in_MM.cuda_mask(), cuda_MM_position,
                                      module_map.cuda_modules_ID(), nb_modules, cuda_module_id, size);
  _size = cuda_hits_in_MM.scan (_blocks);

  //-------------------------------------------------------
  // delete hits that are not on a module or duplicate hits
  //-------------------------------------------------------
  if (_size != size){
    compact_stream (&_cuda_hit_id, cuda_hits_in_MM, _size, _blocks);
    compact_stream (&_cuda_x, cuda_hits_in_MM, _size, _blocks);
    compact_stream (&_cuda_y, cuda_hits_in_MM, _size, _blocks);
    compact_stream (&_cuda_z, cuda_hits_in_MM, _size, _blocks);
    compact_stream (&cuda_MM_position, cuda_hits_in_MM, _size, _blocks);
    compact_stream (&positions, cuda_hits_in_MM, _size, _blocks);
    grid = (_size + _blocks - 1) / _blocks;
  }
#else
  is_module_map_hit<<<grid,blocks>>> (cuda_MM_position,
                                      module_map.cuda_modules_ID(), nb_modules, cuda_module_id, size);
  _size = size;
#endif

  cudaFree (cuda_module_id);

  //------------------------------------------------------------------------------------
  // get nb of hits in MM (some hits may not be part of the MM or can be duplicate hits)
  //------------------------------------------------------------------------------------

  //--------------------
  // sort hits by module
  //--------------------

  int *cuda_sorted_hit_module;
  cudaMalloc (&cuda_sorted_hit_module, _size * sizeof(int));
  thrust_sort_nokeysort (cuda_sorted_hit_module, cuda_MM_position, _size, _blocks);

  _ACORN = true;
  cudaMalloc (&_ACORN_positions, _size * sizeof(int));

  sort (&_cuda_hit_id, cuda_sorted_hit_module, _size, _blocks);
  sort (&_cuda_x, cuda_sorted_hit_module, _size, _blocks);
  sort (&_cuda_y, cuda_sorted_hit_module, _size, _blocks);
  sort (&_cuda_z, cuda_sorted_hit_module, _size, _blocks);
  sort<<<grid,_blocks>>> (_ACORN_positions, positions, cuda_sorted_hit_module, _size);

  cudaFree (positions);

#ifdef MMG_ENABLE_TIMING
  cudaDeviceSynchronize();
  end_qs = start_hits_by_module = clock();
#endif

  //-----------------------------------------------------------------
  // build vector with nb of hits by module (no matter the module ID)
  //-----------------------------------------------------------------

  int *cuda_sorted_MM_position;
  cudaMalloc (&cuda_sorted_MM_position, _size * sizeof(int));
  sort<<<grid,_blocks>>> (cuda_sorted_MM_position, cuda_MM_position, cuda_sorted_hit_module, _size);
  cudaMalloc (&_cuda_hit_indice, (nb_modules+1) * sizeof(int));
  cudaMemsetAsync (_cuda_hit_indice, 0, (nb_modules+1) * sizeof(int));
//  CUDA_mask modules_mask (nb_modules);
  modules_hits<<<grid,_blocks>>> (_cuda_hit_indice, //modules_mask.cuda_mask(),
                                  cuda_sorted_MM_position, _size);
//  int nb_active_modules = modules_mask.scan(_blocks);

  cudaFree (cuda_sorted_MM_position);
  cudaFree (cuda_sorted_hit_module);
  cudaFree (cuda_MM_position);
  cudaDeviceSynchronize();

#ifdef MMG_ENABLE_TIMING
  cudaDeviceSynchronize();
  end_hits_by_module = clock();
#endif

  // -------------------------
  // look for inactive modules
  // fill _cuda_hit_indice with previous non zero value
  // -------------------------

  grid = (nb_modules + _blocks - 1) / _blocks;
  fill_empty_module_indices<<<grid,_blocks>>> (_cuda_hit_indice, nb_modules);

#ifdef MMG_ENABLE_TIMING
  cudaDeviceSynchronize();
  end_hit2module = start_kill_duplicates = clock();
#endif

#ifdef MMG_ENABLE_DEBUG
  //---------------------------------
  // kill duplicate hit IDs by module (low combinatory)
  //---------------------------------
  CUDA_mask cuda_duplicate_hits (_size);
  int *cuda_nb_hits;
  cudaMalloc (&cuda_nb_hits, nb_modules * sizeof(int));
  kill_duplicate_hitID<<<grid,_blocks>>> (cuda_duplicate_hits.cuda_mask(), cuda_nb_hits,
                                          _cuda_hit_indice, _cuda_hit_id, nb_modules);

  int *cuda_nb_hits_sum;
  cudaMalloc (&cuda_nb_hits_sum, (nb_modules+1) * sizeof(int));
  int nb_duplicate_hits = scan (cuda_nb_hits_sum, cuda_nb_hits, _blocks, nb_modules);

  if (nb_duplicate_hits){
    _size -= cuda_duplicate_hits.scan (_blocks);
    vector_subtraction<<<grid,_blocks>>> (_cuda_hit_indice, cuda_nb_hits_sum, nb_modules+1);
    compact_stream_reverse_mask (&_cuda_hit_id, cuda_duplicate_hits, _size, _blocks);
    compact_stream_reverse_mask (&_cuda_x, cuda_duplicate_hits, _size, _blocks);
    compact_stream_reverse_mask (&_cuda_y, cuda_duplicate_hits, _size, _blocks);
    compact_stream_reverse_mask (&_cuda_z, cuda_duplicate_hits, _size, _blocks);
    // reduce modules_mask.cuda_mask() and modules_mask.cuda_mask_sum()

    // ACORN part to get original hit table-----------------------------------------------
    compact_stream_reverse_mask (&_ACORN_positions, cuda_duplicate_hits, _size, _blocks);
    // -----------------------------------------------------------------------------------
  }

  cudaFree (cuda_nb_hits_sum);
  cudaFree (cuda_nb_hits);
#endif

#ifdef MMG_ENABLE_TIMING
  cudaDeviceSynchronize();
  end_kill_duplicate_hits = clock();
#endif

  //------------------
  // deallocate memory
  //------------------

  _gpu_mem_allocated = _ACORN = true;
  cudaMalloc (&_cuda_R, _size * sizeof(T));
  cudaMalloc (&_cuda_eta, _size * sizeof(T));
  cudaMalloc (&_cuda_phi, _size * sizeof(T));
  _cuda_vertices.init (_size);

#ifdef MMG_ENABLE_TIMING
  cudaDeviceSynchronize();
  std::chrono::system_clock::time_point end_real = std::chrono::system_clock::now();
  end = clock ();

  std::cout << magenta << "cpu time for " << size << " hits in event: " << (double)(end - start) / CLOCKS_PER_SEC * 1000 << " ms" << reset;
  std::cout << yellow << "cpu real time: " <<  std::chrono::duration_cast<std::chrono::microseconds>(end_real - start_real).count() * 0.001 << " ms" << reset;
  std::cout << cyan << "cpu time to copy data to GPU: " << (double)(end_cp2gpu - start_cp2gpu) / CLOCKS_PER_SEC << reset;
  std::cout << cyan << "cpu time to assign hits to modules: " << (double)(end_hit2module - start_hit2module) / CLOCKS_PER_SEC << reset;
  std::cout << cyan << "cpu time to kill duplicate hits: " << (double)(end_kill_duplicates - start_kill_duplicates) / CLOCKS_PER_SEC << reset;
  std::cout << cyan << "cpu time for reduction: " << (double)(end_reduction - start_reduction) / CLOCKS_PER_SEC << reset;
  std::cout << cyan << "cpu time for quick sort: " << (double)(end_qs - start_qs) / CLOCKS_PER_SEC << reset;
  std::cout << cyan << "cpu time for hits by module: " << (double)(end_hits_by_module - start_hits_by_module) / CLOCKS_PER_SEC << reset;
  std::cout << cyan << "cpu time to kill duplicate hits: " << (double)(end_kill_duplicate_hits - start_kill_duplicate_hits) / CLOCKS_PER_SEC << reset;
  std::cout << "initial nb elts = " << size << std::endl;
  std::cout << "nb of elts in MM = " << _size << std::endl;
#endif
}
















//---------------------------------------------------------------------------------------------------------------------
template <class T>
void CUDA_hits<T>::classify_hits_on_modules(CUDA_module_map_doublet<T>& module_map,
                                            int nb_modules, int size, int max_blocks){
//---------------------------------------------------------------------------------------------------------------------
#ifdef MMG_ENABLE_TIMING
  clock_t start, end;
  clock_t start_hit2module, end_hit2module;
  clock_t start_scan, end_scan;
  clock_t start_reduction, end_reduction;
  clock_t start_qs, end_qs;
  clock_t start_hits_by_module, end_hits_by_module;
  clock_t start_kill_duplicate_hits, end_kill_duplicate_hits;
#endif

  _position = 0;

#ifdef MMG_ENABLE_TIMING
  cudaDeviceSynchronize();
  start = start_hit2module = clock();
  std::chrono::system_clock::time_point start_real = std::chrono::system_clock::now();
#endif

  //----------------------
  // assign hits to module
  //----------------------

  CUDA_mask cuda_hits_in_MM (size);

  int *cuda_MM_position;
  cudaMalloc (&cuda_MM_position, size * sizeof(int));

  dim3 grid = (((size + _blocks - 1) / _blocks));
  is_module_map_hit<<<grid,_blocks>>> (cuda_hits_in_MM.cuda_mask(), cuda_MM_position,
                                      module_map.cuda_modules_ID(), nb_modules, _cuda_module_id, size);

//  cudaFree (_cuda_module_id);

#ifdef MMG_ENABLE_TIMING
  cudaDeviceSynchronize();
  end_hit2module = start_scan = clock();
#endif

  dim3 grid_dim ((size + _blocks - 1) / _blocks); // delete this line if the previous ones are uncommented

  //------------------------------------------------------------------------------------
  // get nb of hits in MM (some hits may not be part of the MM or can be duplicate hits)
  //------------------------------------------------------------------------------------

  _size = cuda_hits_in_MM.scan (_blocks);

#ifdef MMG_ENABLE_TIMING
  cudaDeviceSynchronize();
  end_scan = start_reduction = clock();
#endif

  // ACORN part to get original hit table-----------------------------------------------
  int *positions;
  cudaMalloc (&positions, size * sizeof(int));
  init_vector<<<grid_dim,_blocks>>> (positions, size);
  // -----------------------------------------------------------------------------------

  //-------------------------------------------------------
  // delete hits that are not on a module or duplicate hits
  //-------------------------------------------------------

  if (_size != size){
    compact_stream (&_cuda_hit_id, cuda_hits_in_MM, _size, _blocks);
    compact_stream (&_cuda_x, cuda_hits_in_MM, _size, _blocks);
    compact_stream (&_cuda_y, cuda_hits_in_MM, _size, _blocks);
    compact_stream (&_cuda_z, cuda_hits_in_MM, _size, _blocks);
    compact_stream (&cuda_MM_position, cuda_hits_in_MM, _size, _blocks);
    compact_stream (&positions, cuda_hits_in_MM, _size, _blocks);

    if (_extra_features){
//    compact_stream (&_hardware, cuda_hits_in_MM, _size, _blocks);
    compact_stream (&_barrel_endcap, cuda_hits_in_MM, _size, _blocks);
    compact_stream (&_layer_disk, cuda_hits_in_MM, _size, _blocks);
    compact_stream (&_eta_module, cuda_hits_in_MM, _size, _blocks);
    compact_stream (&_phi_module, cuda_hits_in_MM, _size, _blocks);
    }

    grid_dim = (_size + _blocks - 1) / _blocks;
  }

#ifdef MMG_ENABLE_TIMING
  cudaDeviceSynchronize();
  end_reduction = start_qs = clock();
#endif

  //--------------------
  // sort hits by module
  //--------------------

  int *cuda_sorted_hit_module;
  cudaMalloc (&cuda_sorted_hit_module, _size * sizeof(int));
  thrust_sort_nokeysort (cuda_sorted_hit_module, cuda_MM_position, _size, _blocks);

  _ACORN = true;
  cudaMalloc (&_ACORN_positions, _size * sizeof(int));

#ifdef MMG_ENABLE_DEBUG
  sort (&_cuda_hit_id, cuda_sorted_hit_module, _size, _blocks);
  sort (&_cuda_x, cuda_sorted_hit_module, _size, _blocks);
  sort (&_cuda_y, cuda_sorted_hit_module, _size, _blocks);
  sort (&_cuda_z, cuda_sorted_hit_module, _size, _blocks);

  if (_extra_features){
//    sort (&_hardware, cuda_sorted_hit_module, _size, _blocks);
    sort (&_barrel_endcap, cuda_sorted_hit_module, _size, _blocks);
    sort (&_layer_disk, cuda_sorted_hit_module, _size, _blocks);
    sort (&_eta_module, cuda_sorted_hit_module, _size, _blocks);
    sort (&_phi_module, cuda_sorted_hit_module, _size, _blocks);
  }
#endif

  sort<<<grid_dim,_blocks>>> (_ACORN_positions, positions, cuda_sorted_hit_module, _size);
  cudaFree (positions);

#ifdef MMG_ENABLE_TIMING
  cudaDeviceSynchronize();
  end_qs = start_hits_by_module = clock();
#endif

  //-----------------------------------------------------------------
  // build vector with nb of hits by module (no matter the module ID)
  //-----------------------------------------------------------------

#ifdef MMG_ENABLE_DEBUG
  int *cuda_sorted_MM_position;
  cudaMalloc (&cuda_sorted_MM_position, _size * sizeof(int));
  sort<<<grid_dim,_blocks>>> (cuda_sorted_MM_position, cuda_MM_position, cuda_sorted_hit_module, _size);
  cudaMemcpyAsync (cuda_MM_position, cuda_sorted_MM_position, _size * sizeof(int), cudaMemcpyDeviceToDevice);
  cudaFree (cuda_sorted_MM_position);
#endif


  cudaMalloc (&_cuda_hit_indice, (nb_modules+1) * sizeof(int));
  cudaMemsetAsync (_cuda_hit_indice, 0, (nb_modules+1) * sizeof(int));
//  CUDA_mask modules_mask (nb_modules);
//  modules_hits<<<grid_dim,_blocks>>> (_cuda_hit_indice, //modules_mask.cuda_mask(),
//                                      cuda_sorted_MM_position, _size);
//  int nb_active_modules = modules_mask.scan(_blocks);
  modules_hits<<<grid_dim,_blocks>>> (_cuda_hit_indice,
                                      cuda_MM_position, _size);

  cudaFree (cuda_sorted_hit_module);
  cudaFree (cuda_MM_position);
  cudaDeviceSynchronize();

#ifdef MMG_ENABLE_TIMING
  cudaDeviceSynchronize();
  end_hits_by_module = clock();
#endif

  // -------------------------
  // look for inactive modules
  // -------------------------

  grid_dim = (nb_modules + _blocks - 1) / _blocks;
  fill_empty_module_indices<<<grid_dim,_blocks>>> (_cuda_hit_indice, nb_modules);

#ifdef MMG_ENABLE_TIMING
  cudaDeviceSynchronize();
  start_kill_duplicate_hits = clock();
#endif

  //---------------------------------
  // kill duplicate hit IDs by module (low combinatory)
  //---------------------------------
  CUDA_mask cuda_duplicate_hits (_size);
  int *cuda_nb_hits;
  cudaMalloc (&cuda_nb_hits, nb_modules * sizeof(int));
  kill_duplicate_hitID<<<grid_dim,_blocks>>> (cuda_duplicate_hits.cuda_mask(), cuda_nb_hits,
                                              _cuda_hit_indice, _cuda_hit_id, nb_modules);

  int *cuda_nb_hits_sum;
  cudaMalloc (&cuda_nb_hits_sum, (nb_modules+1) * sizeof(int));
  int nb_duplicate_hits = scan (cuda_nb_hits_sum, cuda_nb_hits, _blocks, nb_modules);

  if (nb_duplicate_hits){
    _size -= cuda_duplicate_hits.scan (_blocks);
    vector_subtraction<<<grid_dim,_blocks>>> (_cuda_hit_indice, cuda_nb_hits_sum, nb_modules+1);
    compact_stream_reverse_mask (&_cuda_hit_id, cuda_duplicate_hits, _size, _blocks);
    compact_stream_reverse_mask (&_cuda_x, cuda_duplicate_hits, _size, _blocks);
    compact_stream_reverse_mask (&_cuda_y, cuda_duplicate_hits, _size, _blocks);
    compact_stream_reverse_mask (&_cuda_z, cuda_duplicate_hits, _size, _blocks);
    // reduce modules_mask.cuda_mask() and modules_mask.cuda_mask_sum()

    // ACORN part to get original hit table-----------------------------------------------
    compact_stream_reverse_mask (&_ACORN_positions, cuda_duplicate_hits, _size, _blocks);
    // -----------------------------------------------------------------------------------
  }

  cudaFree (cuda_nb_hits_sum);
  cudaFree (cuda_nb_hits);

#ifdef MMG_ENABLE_TIMING
  cudaDeviceSynchronize();
  end_kill_duplicate_hits = clock();
#endif

  _gpu_mem_allocated = _ACORN = true;
  cudaMalloc (&_cuda_R, _size * sizeof(T));
  cudaMalloc (&_cuda_eta, _size * sizeof(T));
  cudaMalloc (&_cuda_phi, _size * sizeof(T));
  _cuda_vertices.init (_size);

#ifdef MMG_ENABLE_TIMING
  cudaDeviceSynchronize();
  std::chrono::system_clock::time_point end_real = std::chrono::system_clock::now();
  end = clock ();

  std::cout << magenta << "cpu time for " << size << " hits in event: " << (double)(end - start) / CLOCKS_PER_SEC * 1000 << " ms" << reset;
  std::cout << yellow << "cpu real time: " <<  std::chrono::duration_cast<std::chrono::microseconds>(end_real - start_real).count() * 0.001 << " ms" << reset;
  std::cout << cyan << "cpu time to assign hits to modules: " << (double)(end_hit2module - start_hit2module) / CLOCKS_PER_SEC << reset;
  std::cout << cyan << "cpu time for scan: " << (double)(end_scan - start_scan) / CLOCKS_PER_SEC << reset;
  std::cout << cyan << "cpu time for reduction: " << (double)(end_reduction - start_reduction) / CLOCKS_PER_SEC << reset;
  std::cout << cyan << "cpu time for quick sort: " << (double)(end_qs - start_qs) / CLOCKS_PER_SEC << reset;
  std::cout << cyan << "cpu time for hits by module: " << (double)(end_hits_by_module - start_hits_by_module) / CLOCKS_PER_SEC << reset;
  std::cout << cyan << "cpu time to kill duplicate hits: " << (double)(end_kill_duplicate_hits - start_kill_duplicate_hits) / CLOCKS_PER_SEC << reset;
  std::cout << "initial nb elts = " << size << std::endl;
  std::cout << "nb of elts in MM = " << _size << std::endl;
#endif
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
void CUDA_hits<T>::add_event (const std::string& event_id, TTree_hits<T>& TThits,
                              CUDA_module_map_doublet<T>& module_map, int blocks, int max_blocks){
//---------------------------------------------------------------------------------------------------------------------
  assert (!_size);
#ifdef MMG_ENABLE_TIMING
  clock_t start, end;
#endif

#ifdef MMG_ENABLE_TIMING
  cudaDeviceSynchronize();
  start = clock();
  std::chrono::system_clock::time_point start_real = std::chrono::system_clock::now();
#endif

  _size = TThits.size();
  _extra_features = TThits.extra_features();
  int nb_modules = module_map.module_map().size();
  _blocks = blocks;

  cudaMalloc (&_cuda_hit_id, _size * sizeof(uint64_t));
  cudaMalloc (&_cuda_module_id, _size * sizeof(uint64_t));
  cudaMalloc (&_cuda_x, _size * sizeof(T));
  cudaMalloc (&_cuda_y, _size * sizeof(T));
  cudaMalloc (&_cuda_z, _size * sizeof(T));

  cudaMemcpyAsync (_cuda_hit_id, TThits.event_hit_ID(), _size * sizeof(uint64_t), cudaMemcpyHostToDevice);
  cudaMemcpyAsync (_cuda_module_id, TThits.event_module_ID(), _size * sizeof(uint64_t), cudaMemcpyHostToDevice);
  cudaMemcpyAsync (_cuda_x, TThits.event_x(), _size * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpyAsync (_cuda_y, TThits.event_y(), _size * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpyAsync (_cuda_z, TThits.event_z(), _size * sizeof(T), cudaMemcpyHostToDevice);

  if (_extra_features){
    cudaMalloc (&_module_x, _size * sizeof(T));
    cudaMalloc (&_module_y, _size * sizeof(T));
    cudaMalloc (&_module_z, _size * sizeof(T));
    cudaMalloc (&_hardware, _size * sizeof(std::string));
    cudaMalloc (&_barrel_endcap, _size * sizeof(int));
    cudaMalloc (&_layer_disk, _size * sizeof(int));
    cudaMalloc (&_eta_module, _size * sizeof(int));
    cudaMalloc (&_phi_module, _size * sizeof(int));

    cudaMemcpyAsync (_module_x, TThits.event_x(), _size * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpyAsync (_module_y, TThits.event_y(), _size * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpyAsync (_module_z, TThits.event_z(), _size * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpyAsync (_hardware, TThits.event_hardware(), _size * sizeof(std::string), cudaMemcpyHostToDevice);
    cudaMemcpyAsync (_barrel_endcap, TThits.event_barrel_endcap(), _size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyAsync (_layer_disk, TThits.event_layer_disk(), _size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyAsync (_eta_module, TThits.event_eta_module(), _size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyAsync (_phi_module, TThits.event_phi_module(), _size * sizeof(int), cudaMemcpyHostToDevice);
  }

//  count_hits_by_module (module_map, TThits.event_hit_ID(), TThits.event_module_ID(),
//                            TThits.event_x(), TThits.event_y(), TThits.event_z(), TThits.size(), blocks, max_blocks);
//  classify_hits_on_modules (module_map, TThits.event_hit_ID(), TThits.event_module_ID(),
//                            TThits.event_x(), TThits.event_y(), TThits.event_z(), TThits.size(), blocks, max_blocks);

  classify_hits_on_modules (module_map, nb_modules, _size, max_blocks);

#ifdef MMG_ENABLE_TIMING
  cudaDeviceSynchronize();
  end = clock();
  std::chrono::system_clock::time_point end_real = std::chrono::system_clock::now();
  std::cout << red << "pre-processing hits: cpu time = " << (double)(end - start) / CLOCKS_PER_SEC << std::endl;
#endif
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
void CUDA_hits<T>::decorations (){
//---------------------------------------------------------------------------------------------------------------------
  dim3 grid_dim ((_size + _blocks - 1) / _blocks);
  hits_constants<<<grid_dim,_blocks>>> (_cuda_R, _cuda_eta, _cuda_phi, _cuda_x, _cuda_y, _cuda_z, _size);
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
void CUDA_hits<T>::decorations (int device, cudaStream_t& stream){
//---------------------------------------------------------------------------------------------------------------------
  dim3 grid_dim ((_size + _blocks - 1) / _blocks);
  hits_constants<<<grid_dim,_blocks,device,stream>>> (_cuda_R, _cuda_eta, _cuda_phi, _cuda_x, _cuda_y, _cuda_z, _size);
}
