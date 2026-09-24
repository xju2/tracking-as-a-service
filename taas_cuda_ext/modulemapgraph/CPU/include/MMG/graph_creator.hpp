/***********************************************************************
 * Module Map Graph library
 * Track reconstruction for HEP
 * copyright © 2022,2023 ROUGIER Charline
 * copyright © 2022,2023,2024 COLLARD Christophe
 * Laboratoire des 2 Infinis - Toulouse - France (L2IT)
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 **********************************************************************/

#ifndef __cplusplus
#error Must use C++ for the type graph creator
#endif

#include <MMG/colors>
#include <string>
#include <iostream>
#include <filesystem>
#include <boost/program_options.hpp>
#include <boost/regex.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <MMG/module_map_triplet>
#include <MMG/hits>
#include <MMG/TTree_hits>
#include <MMG/particles>
#include <MMG/TTree_particles>
#include <MMG/hits_infos>
#include <MMG/graph>
#include <MMG/graph_true>
#include <MMG/strip_hit_pair>
#include <MMG/strip_module_DB>

//#include "Acts/Utilities/Logger.hpp"
// #include "ModuleMapCreatorWriter.hpp"
// #include "GraphCreatorWriter.hpp"
// #include "ActsExamples/Utilities/OptionsFwd.hpp"

/// Add GraphCreation options.
///
/// @param desc The options description to add options to

// cf GraphCreationUtil.hpp

//typedef boost::graph_traits<graphdef>::vertex_descriptor vertex;
//typedef boost::graph_traits<graphdef>::edge_descriptor edge;
typedef boost::unordered_map<uint64_t,int> mapB;

//====================================
template <class T> class graph_creator
//====================================
{
  private:
    std::string _events_dir, _event_filename_pattern, _output_dir, _module_map_dir;
    bool _true_graph, _save_graph_graphml, _save_graph_npz, _save_graph_pyg, _save_graph_csv;
    bool _strip_hit_pair, _extra_features;
    T _min_pt, _max_pt;
    long unsigned int _min_N_hits;
    bool _phi_slice;
    T _phi_slice_cut1;
    T _phi_slice_cut2;
    bool _eta_region;
    T _eta_cut1;
    T _eta_cut2;
    strip_module_DB<data_type2> _strip_module_DB;

    module_map_triplet<T> _module_map_triplet;
    DRsel _drsel;

  public:
    graph_creator (boost::program_options::variables_map&);
    ~graph_creator () {}

    void build (std::string, std::string);
};
