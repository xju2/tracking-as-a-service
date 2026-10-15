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
#error Must use C++ for the type CUDA graph creator
#endif

#include <MMG/colors>
#include <string>
#include <iostream>
#include <numbers>
#include <boost/program_options.hpp>
#include <boost/regex.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <MMG/module_map_triplet>
#include <MMG/CUDA_module_map_doublet>
#include <MMG/CUDA_module_map_triplet>
#include "event_id.cpp"
#include <MMG/hits>
#include <MMG/particles>
#include <MMG/TTree_hits>
#include <MMG/CUDA_hits>
#include <MMG/TTree_particles>
#include <MMG/hits_infos>
#include <MMG/graph>
#include <MMG/vertex>
#include <MMG/edge>
//#include <MMG/graph_true>
#include <MMG/strip_hit_pair>
#include <MMG/strip_module_DB>
#include <MMG/CUDA_quick_sort>
#include <MMG/CUDA_prescan>
#include <MMG/CUDA_mask>
#include <MMG/CUDA_stream_compaction>
#include <MMG/CUDA_geometric_cuts>
#include <MMG/CUDA_nodes>
#include <MMG/CUDA_edges>
//#include <CUDA_strip_module_DB>
#include <MMG/CUDA_error_check>


//==================================================================
template <class T, class module_map_type=T> class CUDA_graph_creator
//==================================================================
{
  private:
    // CPU data
    std::string _events_dir, _output_dir, _module_map_dir;
    bool _true_graph, _save_graph_graphml, _save_graph_npz, _save_graph_pyg, _save_graph_csv;
    int _nb_events;
    CUDA_hits<T> *_cuda_hits;
    bool _strip_hit_pair, _extra_features;

    // GPU data
    int _blocks, _max_blocks;
    int _device;
    cudaStream_t _stream;
    int _start_from, _end_at;
    int *_cuda_nb_edges, *_cuda_edge_sum, *_cuda_module1_hits, *_cuda_module2_hits;
    thrust::device_vector<int> _sort_by_size;

    // CPU and GPU data (mix objects)
    strip_module_DB<data_type2> _strip_module_DB;
    int _nb_doublets, _nb_triplets;
    std::vector<std::string> _event_id;
    CUDA_module_map_doublet<T> _cuda_module_map_doublet;
    CUDA_module_map_triplet<T> _cuda_module_map_triplet;

    void get_program_options (const boost::program_options::variables_map&);
    void read_event (const std::string&, const std::string&);
    void read_events (std::ifstream&, std::ifstream&, const std::string&);
    void initialize (int);
    void initialize (int, module_map_triplet<module_map_type>&);
    void initialize (module_map_triplet<module_map_type>&, int, int, int);
    int count_doublets (int*, const int*, const int&, CUDA_hits<T>&);
    void build_doublets (int*, int*, const int*, int*, CUDA_hits<T>&, int);
    void build_triplets (CUDA_hits<T>&, CUDA_mask&, int, int*, T*, T*, T*, T*, int*, int*, int);

  public:
    inline CUDA_graph_creator () {_nb_events = 0;}
    CUDA_graph_creator (const boost::program_options::variables_map&);
    CUDA_graph_creator (const boost::program_options::variables_map&, int, module_map_triplet<module_map_type>&,
                        std::string, const std::string&);
    CUDA_graph_creator (const boost::program_options::variables_map&, bool);
    CUDA_graph_creator (int, int, const std::string&);
    ~CUDA_graph_creator ();

    void init (const boost::program_options::variables_map&, module_map_triplet<module_map_type>&,
               std::string, const std::string&, int, int, int, cudaStream_t&);

    inline int nb_events () {return _nb_events;}
    inline int device() {return _device;}
    inline std::vector<std::string>& eventID () {return _event_id;}
    void build (CUDA_nodes<T>&, CUDA_edges<T>&, CUDA_hits<T>&);
    void build (CUDA_nodes<T>&, CUDA_edges<T>&, int); // ACORN build
    void build (CUDA_nodes<T>&, CUDA_edges<T>&, std::string, uint64_t*, uint64_t*, T*, T*, T*, int); // ACTS build
    inline void multithreads_build (CUDA_nodes<T>&, CUDA_edges<T>&, cudaStream_t&);
    std::string& get_event_id (int);
    void save (CUDA_nodes<T>*, CUDA_edges<T>*);
    void save (CUDA_nodes<T>&, CUDA_edges<T>&, std::string);
};
