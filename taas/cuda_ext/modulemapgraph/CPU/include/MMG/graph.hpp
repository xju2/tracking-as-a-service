/***********************************************************************
 * Module Map Graph library
 * Track reconstruction for HEP
 * copyright © 2022,2023 ROUGIER Charline
 * copyright © 2022,2023,2024,2025 COLLARD Christophe
 * Laboratoire des 2 Infinis - Toulouse - France (L2IT)
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 **********************************************************************/

#ifndef __cplusplus
#error Must use C++ for the type graph
#endif

#include <boost/graph/adjacency_list.hpp>
#include <boost/property_map/dynamic_property_map.hpp>
#include <boost/graph/graphml.hpp>
#include <MMG/vertex>
#include <MMG/nodes>
#include <MMG/edge>
#include <MMG/edges>

#ifdef MMG_WITH_NUMPY_EXPORT
#include <boost/python/numpy.hpp>
#include <Python.h>
#endif

#ifdef MMG_WITH_PYTORCH_EXPORT
#include <torch/torch.h>
#endif


//#include <torch/script.h>

template <class T> using graph_type = boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS, vertex<T>, edge<T>>;
template <class T> using vertex_type = typename boost::graph_traits<graph_type<T>>::vertex_descriptor;
template <class T> using vertex_iterator = typename boost::graph_traits<graph_type<T>>::vertex_iterator;
template <class T> using edge_type = typename boost::graph_traits<graph_type<T>>::edge_descriptor;
template <class T> using edge_iterator = typename boost::graph_traits<graph_type<T>>::edge_iterator;

//============================
template <class T> class graph
//============================
{
  protected:
    graph_type<T> G;

  public:
    graph () {}
    graph (nodes<T>&, edges<T>&);
    ~graph () {}

    void create (const TTree_hits<T>&, int, std::vector<int>*, std::vector<int>*, std::vector<int>&, std::vector<bool>*);
    vertex_type<T> add_node (const TTree_hits<T>&);
    vertex_type<T> add_vertex (const uint64_t&, const T&, const T&, const T&, const T&);
//    void add_node_graph_true (vertex_type<T>, const hit<T>&);
    void add_edge (const vertex_type<T>&, const vertex_type<T>&, const edge<T>&);
    void add_edge (const int&, const int&, const edge<T>&);
    inline int num_edges () {return boost::num_edges(G);}
    inline int num_vertices () { return boost::num_vertices(G);}
    inline std::pair<edge_type<T>, bool> get_edge (vertex_type<T>& v1, vertex_type<T>& v2) {return boost::edge (v1, v2, G);}
    void save (const std::string&, const std::string&, const bool&);

#ifdef MMG_WITH_NUMPY_EXPORT
    void save_npz (const std::string&, const std::string&, const bool&);
#endif
#ifdef MMG_WITH_PYTORCH_EXPORT
    void save_pyg (const std::string&, const std::string&, const bool&);
#endif
    void save_csv (const std::string&, const std::string&, const bool&);
};
