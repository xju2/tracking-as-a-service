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
#error Must use C++ for the type CUDA module map triplet
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

//==============================================
template <class T> class CUDA_module_map_triplet
//==============================================
{
  private:
    //---------
    // CPU data
    //---------
    int _nb_triplets;

    //geometric cuts for doublets
    std::vector<int> _MD12_map, _MD23_map;

    // geometric cuts for triplets
    CUDA_module_map_doublet<T> _MD12, _MD23;
    std::vector<T> _diff_dydx_min, _diff_dydx_max, _diff_dzdr_min, _diff_dzdr_max;
    std::vector<unsigned int> _occurence;

    void allocate_triplets ();

    //---------
    // GPU data
    //---------
    T *_cuda_diff_dydx_min, *_cuda_diff_dydx_max, *_cuda_diff_dzdr_min, *_cuda_diff_dzdr_max;
    int *_cuda_MD12_map, *_cuda_MD23_map;

  public:
    inline CUDA_module_map_triplet () {_nb_triplets=0;}
    template <class Tf> CUDA_module_map_triplet (module_map_triplet<Tf>&);
    ~CUDA_module_map_triplet ();

    template <class Tf> void init (module_map_triplet<Tf>&);
//    inline int nb_doublets () const {return _nb_doublets;}
//    inline int nb_triplets () const {return _nb_triplets;}
    inline int size () const {return _nb_triplets;}
    inline int* cuda_module12_1 () {return _MD12.cuda_module1();}
    inline int* cuda_module12_2 () {return _MD12.cuda_module2();}
    inline int* cuda_module23_1 () {return _MD23.cuda_module1();}
    inline int* cuda_module23_2 () {return _MD23.cuda_module2();}
    inline int* cuda_module12_map () {return _cuda_MD12_map;}
    inline int* cuda_module23_map () {return _cuda_MD23_map;}
    inline T* cuda_diff_dydx_min () {return _cuda_diff_dydx_min;}
    inline T* cuda_diff_dydx_max () {return _cuda_diff_dydx_max;}
    inline T* cuda_diff_dzdr_min () {return _cuda_diff_dzdr_min;}
    inline T* cuda_diff_dzdr_max () {return _cuda_diff_dzdr_max;}
    inline CUDA_module_map_doublet<T>& module12 () {return _MD12;}
    inline CUDA_module_map_doublet<T>& module23 () {return _MD23;}
    void HostToDevice ();
    void HostToDevice (cudaStream_t&);
};
