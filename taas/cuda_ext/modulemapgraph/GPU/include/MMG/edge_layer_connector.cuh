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
#error Must use C++ for the type CUDA edge layer connector
#endif

#include <string>
#include <iostream>
#include <MMG/tracks>
#include <boost/program_options.hpp>
#include <boost/regex.hpp>
#include <thrust/device_vector.h>
#include <MMG/parameters>
#include <MMG/CUDA_tools>
#include <MMG/CUDA_vector_tools>
#include <MMG/CUDA_graph>
#include <MMG/CUDA_quick_sort>
#include <MMG/CUDA_prescan>
#include <MMG/CUDA_mask>
#include <MMG/CUDA_stream_compaction>
#include <MMG/CUDA_edges>
#include <MMG/CUDA_tracks>

//================================================
template <class T> class CUDA_edge_layer_connector
//================================================
{
  private:
    CUDA_graph<T> *_G=nullptr;
//    bool *_hits_in_tracks=nullptr; // hits[indice] = true if the hit is part of an existing track / false elsewhere (avoid track duplication)
    int _nb_layers, _max_layers;
    int _nb_edges; // try to delete -> included in (*edges).nb_edges()
//    int *_edges_position;
    CUDA_edges<T> *_edges=nullptr; // full list of edges after the cut of low scores
    CUDA_edges<T> *_edge_layers=nullptr; // list of layers that contains the edges with: next layer edge source = pervious layer edge target
//    CUDA_mask _last_edges_mask; // mask for last edges in *_edges
//    int **_layer_edges_position;
    std::string _events_dir, _event_filename_pattern, _output_dir, _module_map_dir;
    T _weights_cut;
    int _min_hits;
    T _factor;
    CUDA_tracks<T> *_tracks=nullptr;

    // GPU data
    int _blocks;
    void read_event (const std::string&);

  public:
    CUDA_edge_layer_connector (const boost::program_options::variables_map&);
    CUDA_edge_layer_connector (CUDA_graph<T>*, const T&, const int&, int, int=30);
    ~CUDA_edge_layer_connector ();

    void free ();
    inline CUDA_graph<T>* graph () {return _G;}
    inline CUDA_tracks<T>* cuda_tracks () {return _tracks;}
    void build_tracks ();
    void weight_cuts ();
    void first_edges ();
    void last_edges (CUDA_edges<T>&);
    void end_edges (CUDA_mask&);
    void next_edges (CUDA_mask&, int);
    void previous_edges (CUDA_edges<T>&, CUDA_edges<T>&);
    int layers ();
    void layers_duplicate_edges (CUDA_edges<T>*, int);
    void sort_v2 (CUDA_edges<T>&, CUDA_edges<T>&, int);
    void sort_v1 (CUDA_edges<T>&, CUDA_edges<T>&, int);
    void sort_v3 (CUDA_edges<T>&, CUDA_edges<T>&, int);
    void sort_v4 (CUDA_edges<T>&, CUDA_edges<T>&, int);
    void sort_v5 (CUDA_edges<T>&, CUDA_edges<T>&, int);
//    void sort_v5 (CUDA_edges<T>&, CUDA_edges<T>&, CUDA_edges<T>&, int);
    inline std::pair<int*,int*> tracks_nbHits_hitIndices () {return std::pair<int*,int*> ((*_tracks).nb_hits(), (*_tracks).hits());}
    inline void filter () {_tracks->filter (_min_hits, _blocks);}
    void save (const std::string&);
};
