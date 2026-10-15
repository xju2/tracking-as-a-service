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
#error Must use C++ for the type CUDA nodes
#endif

#include <string>
#include <iostream>
#include <MMG/nodes>
#include <MMG/CUDA_prescan>
#include <MMG/CUDA_stream_compaction>
#include <MMG/CUDA_hits>

//=================================
template <class T> class CUDA_nodes
//=================================
{
  private:
    int _nb_nodes;
    uint64_t *_cuda_hit_id;
    T *_cuda_R, *_cuda_z, *_cuda_eta, *_cuda_phi;
    int *_cuda_ACORN_positions;

  public:
    inline CUDA_nodes() {_nb_nodes = 0;}
    ~CUDA_nodes ();

    inline int nb_nodes () {return _nb_nodes;}
    void reduce (CUDA_hits<T>&, int);
    void kill ();
    inline uint64_t* cuda_hitID () {return _cuda_hit_id;}
    inline T* cuda_R () {return _cuda_R;}
    inline T* cuda_z () {return _cuda_z;}
    inline T* cuda_eta () {return _cuda_eta;}
    inline T* cuda_phi () {return _cuda_phi;}
    inline int* cuda_ACORN_positions () {return _cuda_ACORN_positions;}
    void DeviceToHost (nodes<T>&);
};
