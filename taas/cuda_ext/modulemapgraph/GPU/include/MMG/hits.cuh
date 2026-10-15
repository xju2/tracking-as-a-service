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

#ifndef __cplusplus
#error Must use C++ for the type CUDA hits
#endif

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <cub/cub.cuh>

#include <MMG/colors>
#include <MMG/parameters>
#include <iostream>
#include <fstream>
#include <assert.h>
#include <utility>
#include <set>
//#include <algorithm>
//#include <execution>
#include <vector>
#include <TFile.h>
#include <boost/type_index.hpp>
#include <MMG/TTree_hits>
#include <MMG/CUDA_module_map_doublet>
#include <MMG/CUDA_geometric_cuts>
#include <MMG/CUDA_vector_tools>
#include <MMG/CUDA_mask>
#include <MMG/CUDA_quick_sort>
#include <MMG/CUDA_prescan>
#include <MMG/CUDA_stream_compaction>


//================================
template <class T> class CUDA_hits
//================================
{
  private:
    //---------
    // CPU data
    //---------
    int _position;
    int _size;
    int _blocks;
    int _max_size;
    std::string _event_ID;
    bool _extra_features;

    //---------
    // GPU data
    //---------
    bool _gpu_mem_allocated;
    uint64_t *_cuda_hit_id, *_cuda_module_id;
    T *_cuda_x, *_cuda_y, *_cuda_z;
    std::string *_hardware; // PIXEL=false STRIP=true
    int *_barrel_endcap, *_layer_disk;
    int *_eta_module, *_phi_module;
    T *_cluster1_x, *_cluster1_y, *_cluster1_z, *_cluster2_x, *_cluster2_y, *_cluster2_z;
    T *_cuda_R, *_cuda_eta, *_cuda_phi;
    T *_module_x, *_module_y, *_module_z;
    int *_cuda_hit_indice; // _cuda_hit_indice = indices for pts: _indice_first+1 -> indice_last
    CUDA_mask _cuda_vertices;
    bool _ACORN; // true = uses ACORN dataset
    int* _ACORN_positions;

  public:
    CUDA_hits ();
//    CUDA_hits (CUDA_module_map_doublet<T>&, const uint64_t*, const uint64_t*, const T*, const T*, const T*,
//               int, int, int);
    CUDA_hits (CUDA_module_map_doublet<T>&,
              const uint64_t*, const uint64_t*, const T*, const T*, const T*, int, int, int,
              const std::string *hardware=nullptr, const int *barrel_endcap=nullptr, const int *layer_disk=nullptr,
              const int *eta_module=nullptr, const int *phi_module=nullptr);

//    CUDA_hits (const TTree_hits<T>&, const std::multimap<uint64_t,int>&);

    ~CUDA_hits ();

//    inline void build_R () {std::transform (std::execution::par_unseq, std::begin(_x), std::end(_x), std::begin(_y), std::begin(_R), [](data_type2 xi, data_type2 yi) {return sqrt (xi * xi + yi * yi);});}

    void count_hits_by_module (CUDA_module_map_doublet<T>&,
                               const uint64_t*, const uint64_t*, const T*, const T*, const T*, int, int, int);
    void fast_classify_hits_on_modules (CUDA_module_map_doublet<T>&,
                                   const uint64_t*, const uint64_t*, const T*, const T*, const T*, int, int, int);
    void classify_hits_on_modules (CUDA_module_map_doublet<T>&,
                                   const uint64_t*, const uint64_t*, const T*, const T*, const T*, int, int, int);
    void classify_hits_on_modules (CUDA_module_map_doublet<T>&, int, int, int);
    void add_event (const std::string&, TTree_hits<T>&, CUDA_module_map_doublet<T>&, int, int);
    void decorations ();
    void decorations (int, cudaStream_t&);
    inline std::string event_id () {return _event_ID;} // change value [0] if multiple id

    inline int size () {return _size;}
    inline bool ACORN () const {return _ACORN;}
    inline uint64_t* cuda_hit_id () {return _cuda_hit_id;}
    inline T* cuda_x () {return _cuda_x;}
    inline T* cuda_y () {return _cuda_y;}
    inline T* cuda_z () {return _cuda_z;}
    inline T* cuda_R () {return _cuda_R;}
    inline T* cuda_eta () {return _cuda_eta;}
    inline T* cuda_phi () {return _cuda_phi;}
    inline T* module_x () {return _module_x;}
    inline T* module_y () {return _module_y;}
    inline T* module_z () {return _module_z;}
    inline int* cuda_hit_indice () {return _cuda_hit_indice;}
    inline bool* cuda_vertices () {return _cuda_vertices.cuda_mask();}
    inline int* cuda_vertices_sum () {return _cuda_vertices.cuda_mask_sum();}
    inline int cuda_vertices_scan () {return _cuda_vertices.scan (_blocks);}
    inline int* cuda_ACORN_positions () {return _ACORN_positions;}
};
