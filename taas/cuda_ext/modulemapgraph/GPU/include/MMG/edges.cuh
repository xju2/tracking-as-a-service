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
#error Must use C++ for the type CUDA edges
#endif

#include <MMG/edges>
#include <MMG/CUDA_mask>
#include <MMG/CUDA_stream_compaction>
#include <MMG/CUDA_hits>
//#include <MMG/CUDA_quick_sort>

//=================================
template <class T> class CUDA_edges
//=================================
{
  private:
    int _nb_edges;
    int *_edge_source, *_edge_target; // (edge_source = edge input vertex / edge_target = edge output vertex)
    T *_edge_weight;
    T *_cumulative_edge_weight;
    int *_edge_tag;
    int *_shared_hit;
//    int *_edge_global_position;
    int *_ACORN_edge_source, *_ACORN_edge_target;
    bool _ACORN, _ACTS, _decorations;
    T *_dR, *_dz, *_deta, *_dphi, *_phi_slope, *_r_phi_slope;
    int _tag;

  public:
    inline CUDA_edges () {_ACORN=_ACTS=_decorations=false; _nb_edges=0; _tag=-1;}
    CUDA_edges (int);
    CUDA_edges (CUDA_edges<T>*, CUDA_mask&, int);
    CUDA_edges (int, int*, int*, T*);
    CUDA_edges (int, int*, int*, T*, bool);
    ~CUDA_edges ();

    void reduce (int*, int*, T*, T*, T*, CUDA_mask&, int);
    void reduce (int*, int*, T*, T*, T*, CUDA_mask&, CUDA_hits<T>&, int);

    void reduce (CUDA_mask&, int);
    void reduce (CUDA_mask&, int*, int);
    void reverse_reduce (CUDA_mask&, int);
    void build (const CUDA_edges<T>*, CUDA_mask&, int);
    void build (const CUDA_edges<T>*, CUDA_mask&, CUDA_mask&, int);
    void build (CUDA_edges<T>*, CUDA_mask&, int*, int); // ACORN build
    void reverse_build (const CUDA_edges<T>*, CUDA_mask&, int);
    void tag_edges (int, int=0);
    void tag_edges (CUDA_edges<T>&, int);
    void copy_tags (const CUDA_edges<T>&, int);
    void connect_layer_edges (CUDA_mask&, const CUDA_edges<T>&, int);
    void connect_edges (CUDA_edges<T>&, int);
    void cut_new_edges (CUDA_edges<T>&, int);
    void update_weights (CUDA_edges<T>&, int);
    void merge_edges (CUDA_edges<T>&, CUDA_edges<T>&, int);
    void merge_edges (CUDA_edges<T>&, CUDA_edges<T>&, int*, int);
    void find_duplicate_sources (CUDA_mask&, CUDA_mask&, int);
    void untag_duplicate_sources (int);
    void untag_duplicate_targets (int);
    void filter_last_edges (CUDA_mask&, CUDA_edges<T>&, int);
    void filter_low_scores (CUDA_mask&, T, int);
//    void filter (CUDA_mask&, CUDA_mask&, CUDA_mask&, int); // exclude edges that are not in edges given as argument => modify mask
    void features (CUDA_hits<T>&, int);
    void kill ();
    void DeviceToHost (edges<T>&);

    inline int nb_edges () {return _nb_edges;}
    inline int* source () {return _edge_source;}
    inline int* target () {return _edge_target;}
    int* ACORN_source ();
    int* ACORN_target ();
//    inline int* edge_global_position () {return _edge_global_position;}
    inline T* dR () {return _dR;}
    inline T* dz () {return _dz;}
    inline T* deta () {return _deta;}
    inline T* dphi () {return _dphi;}
    inline T* phi_slope () {return _phi_slope;}
    inline T* r_phi_slope () {return _r_phi_slope;}
    inline T* weight () {return _edge_weight;}
    inline T* cumulative_weight () {return _cumulative_edge_weight;}
    inline int* tags () {return _edge_tag;}
    inline int layer_tag () {return _tag;}
    inline void set_source (int* source) {_edge_source = source;}
    inline void set_target (int* target) {_edge_target = target;}
    void save (const std::string&, const std::string&);
    
//    void compact (CUDA_edges<T>&, CUDA_mask&);

    template <class Tf> friend std::ostream& operator << (std::ostream&, const CUDA_edges<Tf>&);
};
