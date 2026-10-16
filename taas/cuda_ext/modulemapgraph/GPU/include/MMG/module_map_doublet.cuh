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
#error Must use C++ for the type CUDA module map doublet
#endif

#include <time.h>
#include <string>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <map>
#include <TFile.h>
#include <TTree.h>
#include <boost/program_options.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/regex.hpp>
#include <MMG/hits>
#include <MMG/TTree_hits>
#include <MMG/TTree_particles>
#include <MMG/hits_infos>
#include <MMG/geometric_cuts>
#include <MMG/module_doublet>
#include <MMG/module_triplet>
#include <MMG/module_map_triplet>
#include <MMG/edge>
#include <iterator>
#include <MMG/CUDA_vector_tools>
#include <MMG/CUDA_prescan>
#include <MMG/CUDA_stream_compaction>

//==============================================
template <class T> class CUDA_module_map_doublet
//==============================================
{
  private:
    //---------
    // CPU data
    //---------
    bool _memory_allocation;
    int _nb_doublets;

    //geometric cuts for doublets
    std::vector<T> _z0_min, _z0_max, _dphi_min, _dphi_max, _phi_slope_min, _phi_slope_max, _deta_min, _deta_max;

    std::vector<int> _module1, _module2;
    std::multimap<uint64_t, int> _module_map;
    std::vector<uint64_t> _modulesID;
    unsigned* map_addresses;

    void allocate_doublets ();

    //---------
    // GPU data
    //---------
    T *_cuda_z0_min, *_cuda_z0_max, *_cuda_dphi_min, *_cuda_dphi_max, *_cuda_phi_slope_min, *_cuda_phi_slope_max, *_cuda_deta_min, *_cuda_deta_max;
    int *_cuda_module1, *_cuda_module2;
    uint64_t *_cuda_modulesID;

  public:
    inline CUDA_module_map_doublet () {_nb_doublets=0; _memory_allocation = false;}
    template <class Tf> CUDA_module_map_doublet (module_map_triplet<Tf>&);
    ~CUDA_module_map_doublet ();

    template <class Tf> void init (module_map_triplet<Tf>&);
//    void filter_doublets (const CUDA_mask&, int);
    template <class Tf> void build_doublets (module_map_triplet<Tf>&, std::string);
    inline int size () const {return _nb_doublets;}
    inline const std::multimap <uint64_t, int>& module_map () const {return _module_map;}
    inline std::vector<int> module1 () {return _module1;}
    inline std::vector<int> module2 () {return _module2;}

    inline int* cuda_module1 () {return _cuda_module1;}
    inline int* cuda_module2 () {return _cuda_module2;}
    inline uint64_t* cuda_modules_ID () const {return _cuda_modulesID;}
    inline T* cuda_z0_min () {return _cuda_z0_min;}
    inline T* cuda_z0_max () {return _cuda_z0_max;}
    inline T* cuda_dphi_min () {return _cuda_dphi_min;}
    inline T* cuda_dphi_max () {return _cuda_dphi_max;}
    inline T* cuda_phi_slope_min () {return _cuda_phi_slope_min;}
    inline T* cuda_phi_slope_max () {return _cuda_phi_slope_max;}
    inline T* cuda_deta_min () {return _cuda_deta_min;}
    inline T* cuda_deta_max () {return _cuda_deta_max;}
    void HostToDevice ();
    void HostToDevice (cudaStream_t&);
};
