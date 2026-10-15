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

#include "MMG/graph_true.hpp"

//----------------------------------------------------
template <class T>
void graph_true<T>::add_node (const TTree_hits<T>& ht)
//----------------------------------------------------
{
  vertex_type<T> v = boost::add_vertex (G);
  G[v].r() = ht.R();
  G[v].z() = ht.z();
//  G[v].eta() = Eta (ht);
  G[v].eta() = ht.Eta ();
  G[v].phi() =  ht.Phi ();
  G[v].hit_id() = ht.hit_id();
}


//---------------------------------------------------------------------------------------------------------------------------------------
template <class T>
void graph_true<T>::add_edge (const vertex_type<T>& v1, const vertex_type<T>& v2, int& n_edges, const T& minPt, const uint64_t& minNHits, const int& hit1, const int& hit2, 
  const std::pair<std::_Rb_tree_iterator<std::pair<const uint64_t, int>>, std::_Rb_tree_iterator<std::pair<const uint64_t, int>>>&  v_hits, TTree_hits<T>& TThits, TTree_particles<T>& TTp)
//---------------------------------------------------------------------------------------------------------------------------------------
//  inline int long ActsExamples::add_TrueEdge(vertex& v1, vertex& v2, int long n_true_edges, graph_true& G, float minPt, int minNHits, 
//                                    ActsExamples::CsvHitReader_athenaDataCsv_graphCreation::Hitinformation h1,
//                                    ActsExamples::CsvHitReader_athenaDataCsv_graphCreation::Hitinformation h2,
                                    //ActsExamples::Range<std::_Rb_tree_const_iterator<std::pair<const long int, std::vector<ActsExamples::CsvHitReader_athenaDataCsv_graphCreation::Hitinformation> > > > v_hits)
{
  /*
    Add a true edge and its features (true boolean flag, pt, mask)
  */

  edge_true_type<T> e;
  bool b;
  boost::tie(e, b) = boost::add_edge (v1, v2, G);
  G[e].mask_edge() = 1;
  G[e].is_segment_true() = 0;
  G[e].region() = define_region (TThits, hit1, hit2);
//  G[e].pT () = 0;
  TTp.getID (TThits.particle_ID(hit1));
  T pT = 0.001 * TTp.pT();

  if (std::distance (v_hits.first, v_hits.second) > 2)
    if (TThits.particle_ID(hit1) == TThits.particle_ID(hit2)) // && std::abs(TTp.pdgID()) != 11) //electrons
    // check marginal particles
      if (std::abs(TTp.eta()) <= 4  &&  std::sqrt(TTp.vx()*TTp.vx() + TTp.vy()*TTp.vy()) <= 260)
        for (std::_Rb_tree_iterator<std::pair<const uint64_t, int>> hit = v_hits.first; hit != v_hits.second; hit++)
          { //        if (std::sort(v_hits.first, v_hits.second)<3) break;
            int it1 = hit->second;
            std::_Rb_tree_iterator<std::pair<const uint64_t, int>> next_hit = hit;
            int it2 = (++next_hit)->second;
            if (next_hit == v_hits.second) break;

                  // There is a possibility that two successives hits of a particles are on the same silicon module, due to clustering effect.
                  // In such case, one must be carefull of how to flag a connection as true as the module map only connect one module to another one.
                  // Taken A->B->C->D four hits of a particle, with B and C on the same module, the true connection must be:
                  // A->B
                  // B->D
                  //check standard successives hits on differents modules
///                  if (TThits[it1].module_ID() == TThits[it2].module_ID())
///                    { //check if hits are on the same modules
///                      //two successives hits are indeed on the same module
///                      //find that B and C are on the same module
///                      it2 = (++next_hit)->second; //hh+2;
///
///                      //let's check if the edge seen is made of B and D
///                      if (TThits[it1].hit_id() == h1.hit_id()  &&  TThits[it2].hit_id() == h2.hit_id())
///                        { if (size - 1 < minNHits) break; //in that case we need to be sure that by removing C the particle still leave 3 hits in the detector
///                          if (pT >= minPt   && TTp.barcode()<200000 )
///                            { G[e].is_segment_true () = 1;
///                              std::cout << "GT = " << G[e].is_segment_true() << std::endl;
///                              n_edges++;
///                            }
///                          else
///                            G[e].mask_edge () = 0;
///                          G[e].pT () = pT;
///                          break;
///                        }
///                    }
///                  else
            if (TThits.hit_id(it1) == TThits.hit_id(hit1) && TThits.hit_id(it2) == TThits.hit_id(hit2))
              { if (pT >= minPt && TTp.barcode () < 200000 && std::abs(TTp.pdgID()) != 11)
                  { G[e].is_segment_true () = 1;
                    n_edges++;
                  }
                else
                  G[e].mask_edge () = 0;
                G[e].pT () = pT;
                break;
              }
          }
}


//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <class T>
void graph_true<T>::modify_flag (vertex_type<T>& v1, vertex_type<T>& v2, int& n_true_edges, const T& minPt, const uint64_t& minNHits, const int& hit1,  const int& hit2,
  const std::pair<std::_Rb_tree_iterator<std::pair<const uint64_t, int>>, std::_Rb_tree_iterator<std::pair<const uint64_t, int>>>&  v_hits, TTree_hits<T>& TThits, TTree_particles<T>& TTp)
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
{
  /*
  Check if an existing edge is a true one
  */
  TTp.getID (TThits.particle_ID(hit1));
  T pT = 0.001 * TTp.pT();
  if (std::distance (v_hits.first, v_hits.second) > 2)
    if (TThits.particle_ID() == TThits.particle_ID(hit2)) // && std::abs(TTp.pdgID()) != 11) electrons
      if ((std::abs(TTp.eta()) <= 4) && (std::sqrt(TTp.vx() * TTp.vx() + TTp.vy() * TTp.vy()) <= 260))
        for (std::_Rb_tree_iterator<std::pair<const uint64_t, int>> hit = v_hits.first; hit != v_hits.second; hit++)
          { std::_Rb_tree_iterator<std::pair<const uint64_t, int>> next_hit = hit;
            int it1 = hit->second;
            int it2 = (++next_hit)->second;
            if (next_hit == v_hits.second) break;

            //There is a possibility that two successives hits of a particles are on the same silicon module,due to clustering effect.
            //In such case, one must be carefull of how to flag a connection as true as the module map only connect one module to another one.
            //Taken A->B->C->D four hits of a particle, with B and C on the same module, the true connection must be:
            //A->B
            //B->D  
              //check standard successives hits on differents modules
///              if (TThits[it1].module_ID() == TThits[it2].module_ID())
///                { //check if hits are on the same modules
                  //two successives hits are indeed on the same module
                  //find that B and C are on the same module
///                 it2 = hh + 2;
///                 exit(0);
//                it2 = (++next_nhit)->second;
                  //let's check if the edge seen is made of B and D
///                if (TThits[it1].hit_id() == h1.hit_id() && TThits[it2].hit_id() == h2.hit_id())
///                  { // if (size-1 < minNHits) break; //in that case we need to be sure that by removing C the particle still leave 3 hits in the detector
///                    if (TTp.pT() >= minPt   && TTp.barcode() < 200000)
///                      { G[get_edge(v1, v2).first].is_segment_true() = 1;
//                      std::cout << "GT = " << G[get_edge(v1, v2).first].is_segment_true() << std::endl;
///                        n_true_edges++;
///                      }
///                    else
///                      G[get_edge(v1, v2).first].mask_edge() = 0;
///                    break;
///                  }
///                }
///              else
                if (TThits.hit_id(it1) == TThits.hit_id(hit1) && TThits.hit_id(it2) == TThits.hit_id(hit2))
                  { if (pT >= minPt && TTp.barcode() < 200000 && std::abs(TTp.pdgID()) != 11)
                      { G[get_edge(v1, v2).first].is_segment_true() = 1;
                        n_true_edges++;
                      }  
                    else
                      G[get_edge(v1, v2).first].mask_edge() = 0;
                    break;
                  }
              }
}


//-----------------------------------------------------------------------------------
template <class T>
void graph_true<T>::save (const std::string& event_id, const std::string& output_dir)
//-----------------------------------------------------------------------------------
{
  boost::dynamic_properties dp (boost::ignore_other_properties);
  //edge
  dp.property ("is_segment_true", boost::get (&edge_true<T>::_is_segment_true, G));
  dp.property ("pt_particle", boost::get (&edge_true<T>::_pt_particle, G));
  dp.property ("mask_edge", boost::get (&edge_true<T>::_mask_edge, G));
  dp.property ("region", boost::get (&edge_true<T>::_region, G));

    //node
  dp.property ("r", boost::get (&vertex_true<T>::_r, G));
  dp.property ("phi", boost::get (&vertex_true<T>::_phi, G));
  dp.property ("z", boost::get (&vertex_true<T>::_z, G));
  dp.property ("hit_id", boost::get (&vertex_true<T>::_hit_id, G));
  dp.property ("pt_particle", boost::get (&vertex_true<T>::_pt_particle, G));
  dp.property ("eta", boost::get (&vertex_true<T>::_eta, G));
//  dp.property ("index", boost::get (&vertex_true<T>::index, G)); //not set yet

    std::string filename = "/event" + event_id + "_TARGET.txt";
    std::string path_SaveGraphs = output_dir + filename;
    std::fstream outGraph_TRUE (path_SaveGraphs,std::ios::out);
    boost::write_graphml (outGraph_TRUE, G, dp, true);
    outGraph_TRUE.close();
}


#ifdef MMG_WITH_NUMPY_EXPORT
//---------------------------------------------------------------------------------------
template <class T>
void graph_true<T>::save_npz (const std::string& event_id, const std::string& output_dir)
//---------------------------------------------------------------------------------------
{
  Py_Initialize();
  boost::python::numpy::initialize();

  //--------------
  // node features
  //--------------
  int nb_nodes = boost::num_vertices (G);
  boost::python::numpy::dtype dtype = boost::python::numpy::dtype::get_builtin<T>();
//  boost::tuple <T, T, T, T, int, T> *node_feat = new boost::tuple <T, T, T, T, int, T> [nb_nodes];
  boost::tuple <T, T, T, T, T, T> *node_feat = new boost::tuple <T, T, T, T, T, T> [nb_nodes];
  int ind = 0;
  vertex_iterator<T> vi, vi_end;

  for (boost::tie (vi, vi_end) = boost::vertices(G); vi != vi_end; ++vi, ind++)
    node_feat[ind] = boost::tuple <T, T, T, T, T, T> (G[*vi].pT(), G[*vi].r(), G[*vi].phi(), G[*vi].z(), G[*vi].hit_id(), G[*vi].eta());
//    node_feat[ind] = boost::tuple <T, T, T, T, int, T> (G[*vi].pT(), G[*vi].r(), G[*vi].phi(), G[*vi].z(), G[*vi].hit_id(), G[*vi].eta());

//  boost::python::numpy::ndarray nd_nodes = boost::python::numpy::from_data (node_feat, dtype, boost::python::make_tuple (nb_nodes, 3), boost::python::make_tuple (sizeof(T) + sizeof(T) + sizeof(T) + sizeof(T) + sizeof(int) + sizeof(T), sizeof(T)), boost::python::object());
  boost::python::numpy::ndarray nd_nodes = boost::python::numpy::from_data (node_feat, dtype, boost::python::make_tuple (nb_nodes, 6), boost::python::make_tuple (sizeof(T) + sizeof(T) + sizeof(T) + sizeof(T) + sizeof(T) + sizeof(T), sizeof(T)), boost::python::object());

  // -------------------------------------
  // edge features / senders and receivers
  //--------------------------------------
  int nb_edges = boost::num_edges (G);
//  boost::tuple <int, T, int, int> *edge_feat = new boost::tuple <int, T, int, int> [nb_edges];
  boost::tuple <T, T, T, T> *edge_feat = new boost::tuple <T, T, T, T> [nb_edges];
  int* senders = new int [nb_edges];
  int* receivers = new int [nb_edges];
  ind = 0;
  edge_true_iterator<T> ei, ei_end;

  for (boost::tie(ei, ei_end) = boost::edges(G); ei != ei_end; ++ei, ind++) {
    edge_feat[ind] = boost::tuple <T, T, T, T> (G[*ei].is_segment_true (), G[*ei].pT(), G[*ei].mask_edge(), G[*ei].region());
//    edge_feat[ind] = boost::tuple <int, T, int, int> (G[*ei].is_segment_true (), G[*ei].pT(), G[*ei].mask_edge(), G[*ei].region());
    receivers[ind] = boost::target (*ei, G);
    senders[ind] = boost::source (*ei, G);
  }

//  boost::python::numpy::ndarray nd_edges = boost::python::numpy::from_data (edge_feat, dtype, boost::python::make_tuple (nb_edges, 4), boost::python::make_tuple (sizeof(int) + sizeof(T) + sizeof(int) + sizeof(int), sizeof(T)), boost::python::object());
  boost::python::numpy::ndarray nd_edges = boost::python::numpy::from_data (edge_feat, dtype, boost::python::make_tuple (nb_edges, 4), boost::python::make_tuple (sizeof(T) + sizeof(T) + sizeof(T) + sizeof(T), sizeof(T)), boost::python::object());
  boost::python::numpy::dtype dtype2 = boost::python::numpy::dtype::get_builtin<int>();
  boost::python::numpy::ndarray nd_receivers = boost::python::numpy::from_data (receivers, dtype2, boost::python::make_tuple (nb_edges), boost::python::make_tuple (sizeof(int)), boost::python::object());
  boost::python::numpy::ndarray nd_senders = boost::python::numpy::from_data (senders, dtype2, boost::python::make_tuple (nb_edges), boost::python::make_tuple (sizeof(int)), boost::python::object());

  // globals
  float glob=0;
  boost::python::numpy::ndarray ndT = boost::python::numpy::from_data (&glob, dtype, boost::python::make_tuple (1), boost::python::make_tuple (sizeof(T)), boost::python::object());

  //----------------------------
  // Write out graph to NPZ file
  //----------------------------
  std::cout << "Writing graph to NPZ file ..." << std::endl;
  boost::python::object my_python_class_module = boost::python::import ("numpy");
  // Retrieve the main module's namespace
  boost::python::object global (my_python_class_module.attr ("__dict__"));
  boost::python::dict kw;
  kw ["nodes"] = nd_nodes;
  kw ["edges"] = nd_edges;
  kw["receivers"] = nd_receivers;
  kw["senders"] = nd_senders;
  kw["globals"] = ndT;
  kw ["n_node"] = nb_nodes;
  kw ["n_edge"] = nb_edges;
  std::string filename="/event" + event_id + "_TARGET.npz";
  std::string path_SaveGraphs = output_dir + filename;
  boost::python::object str_obj = boost::python::str (path_SaveGraphs.c_str());
  boost::python::tuple t = make_tuple (str_obj);
  my_python_class_module.attr ("savez_compressed") (*t, **kw);
  std::cout << "... done." << std::endl;

  delete [] node_feat;
  delete [] edge_feat;
  delete [] senders;
  delete [] receivers;
}
#endif
