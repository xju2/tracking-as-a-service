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

#include "MMG/graph_creator.hpp"
#include "MMG/event_id.hpp"

//----------------------------------------------------------------------------
template <class T>
graph_creator<T>::graph_creator (boost::program_options::variables_map& po_vm)
//----------------------------------------------------------------------------
{
  _events_dir = po_vm["input-dir"].as<std::string> () + "/";
  _event_filename_pattern = po_vm["input-filename-pattern"].as<std::string> ();
  _output_dir = po_vm["output-dir"].as<std::string> ();
  _true_graph = po_vm["give-true-graph"].as<bool> ();
  _module_map_dir = po_vm["input-module-map"].template as<std::string> ();
  _save_graph_graphml = po_vm["save-graph-on-disk-graphml"].as<bool> ();
  _save_graph_npz = po_vm["save-graph-on-disk-npz"].as<bool> ();
  _save_graph_pyg = po_vm["save-graph-on-disk-pyg"].as<bool> ();
  _save_graph_csv = po_vm["save-graph-on-disk-csv"].as<bool> ();
  _min_N_hits = po_vm["min-nhits"].as<long unsigned int> ();
  _min_pt = po_vm["min-pt-cut"].as<T> ();
  _max_pt = po_vm["max-pt-cut"].as<T> ();
  _phi_slice = po_vm["phi-slice"].as<bool> ();
  _phi_slice_cut1 = po_vm["cut1-phi-slice"].as<T> ();
  _phi_slice_cut2 = po_vm["cut2-phi-slice"].as<T> ();
  _eta_region = po_vm["eta-region"].as<bool> ();
  _eta_cut1 = po_vm["cut1-eta"].as<T> ();
  _eta_cut2 = po_vm["cut2-eta"].as<T> ();
  _strip_hit_pair = po_vm["strip-hit-pair"].as<bool> ();
  _extra_features = po_vm["extra-features"].as<bool> ();
  
  _drsel = static_cast<DRsel>(po_vm["drsel"].as<int>());

  if(_drsel==POS_dr) std::cout<<"Keeping only triplets with dr >= 0 "<<std::endl;
  if(_drsel==STRICTPOS_dr) std::cout<<"Keeping only triplets with dr > 0 "<<std::endl;

  if (_strip_hit_pair) {
    _strip_module_DB = strip_module_DB<data_type2> (po_vm["input-strip-modules"].as<std::string> ());
    _extra_features = true;
  }

  if (_module_map_dir.empty()) throw std::invalid_argument ("Missing Module Map");

  _module_map_triplet.read_TTree (_module_map_dir.c_str());
  if (!_module_map_triplet) throw std::runtime_error ("Cannot retrieve ModuleMap from " + _module_map_dir);

  // log all hits events filenames
  std::string log_hits_filenames = "events-hits.txt";
  std::string cmd_hits = "ls " + _events_dir + "|grep " + _event_filename_pattern + " |grep hits.* > " + log_hits_filenames;
  system (cmd_hits.c_str());
  cmd_hits = "ls " + _events_dir + "|grep " +  _event_filename_pattern + " |grep truth.* >> " + log_hits_filenames;
  system (cmd_hits.c_str());

  // log all particles events filenames
  std::string log_particles_filenames = "events-particles.txt";
  std::string cmd_particles = "ls " + _events_dir + "|grep " + _event_filename_pattern + " |grep particles.* > " + log_particles_filenames;
  system (cmd_particles.c_str());

  // open files containing events filenames
  std::ifstream h_file (log_hits_filenames);
  if (h_file.fail()) throw std::invalid_argument ("Cannot open file " + log_hits_filenames);
  std::ifstream p_file (log_particles_filenames);
  if (p_file.fail()) throw std::invalid_argument ("Cannot open file " + log_particles_filenames);

  std::string hits_filename, particles_filename;
  h_file >> hits_filename;
  p_file >> particles_filename;

  int i=0;
  for (; !h_file.eof() && !p_file.eof(); i++) {
    build (hits_filename, particles_filename);

    h_file >> hits_filename;
    p_file >> particles_filename;
  }

  h_file.close();
  p_file.close();

/*
  std::string _module_map_dir_output = "/sps/l2it/collard/MMTriplet_3hits_ptCut300MeV_events_0kto1_woutOverlapSP_particleInfo.out.root";
  _module_map_triplet.save_TTree (_module_map_dir_output);
  module_map_triplet<T> m_modMap2;
  m_modMap2.read_TTree (_module_map_dir_output);
  if (_module_map_triplet == m_modMap2) std::cout << "OK !!!!! YUPI" << std::endl;
*/
}


//--------------------------------------------------------------------------------------
template <class T>
void graph_creator<T>::build (std::string hits_filename, std::string particles_filename)
//--------------------------------------------------------------------------------------
{
  clock_t start, end;
  clock_t start_main, end_main;
  clock_t start_doublets, end_doublets;
  clock_t start_triplets, end_triplets;

  start_main = start = clock();

  std::string hits_ext = hits_filename.substr (_event_filename_pattern.length());
  std::string event_id = extract_event_id(hits_ext);
  std::string particles_ext = particles_filename.substr (_event_filename_pattern.length());
  if (event_id != extract_event_id(particles_ext))
    throw std::invalid_argument ("hits and particules file are not part of the same event");
  std::stringstream ID;
  ID << event_id;
  int eventID = 0;
  ID >> eventID;

  // load event hits
  hits<T> hit_pts (_true_graph, _extra_features);  // define in class data ?
  TTree_hits<T> TThits (_true_graph, _extra_features);
  std::filesystem::path hits_path = hits_filename;
  if (hits_path.extension () == ".csv")
    { hit_pts.read_csv (_events_dir, hits_filename);
      TThits = hit_pts;
    }
  else if (hits_path.extension () == ".root")
    TThits.read (_events_dir + hits_filename);
  else throw std::invalid_argument ("unknow extension for hit file: " + hits_filename);

  // load event particles
  particles<T> p;
  TTree_particles<T> TTp;
  std::filesystem::path particles_path = particles_filename;
  if (particles_path.extension () == ".csv")
    { p.read_csv (_events_dir, particles_filename);
      TTp = p;
    }
  else if (particles_path.extension () == ".root")
    TTp.read (_events_dir + particles_filename);
  else throw std::invalid_argument ("unknow extension for particles file: " + particles_filename);

//  using HitGraphContainer = boost::unordered::iterator_detail::iterator<boost::unordered::detail::ptr_node <std::pair<const uint64_t, int> > >;
  using HitGraphContainer = boost::unordered::unordered_map<long unsigned int, int>::iterator;

  hits_infos<T> HitsInfos (TThits, TTp);

  graph<T> G;
  graph_true<T> G_true;

 //check purpose
  int true_edges_number = 0;

  //map hit-node to not add multiple time the same node
  mapB hit_to_node;
  uint32_t edges = 0;

int nb_connexions = 0;
int nb_kills = 0;
int nb_pass = 0;
int nb_triplets = 0;

  int MMD_size = _module_map_triplet.map_doublet().size();

  std::vector<int> *M1_SP, *M2_SP;
  std::vector<bool> *vertex;
  M1_SP = new std::vector<int> [MMD_size];
  M2_SP = new std::vector<int> [MMD_size];
  vertex = new std::vector<bool> [MMD_size];

//  std::vector<geometric_cuts<T>> cuts[MMD_size];

  // ---------------------------------------------
  // loop over module doublets to kill connections
  // ---------------------------------------------

  int nb_elements = 0;

  start_doublets = clock();
  for (std::_Rb_tree_iterator<std::pair<const std::vector<uint64_t>, module_doublet<T>>> it2 = _module_map_triplet.map_doublet().begin(); it2 != _module_map_triplet.map_doublet().end(); it2++) {

    uint64_t module1 = it2->first[0];
    uint64_t module2 = it2->first[1];
    std::pair<std::_Rb_tree_iterator<std::pair<const uint64_t, int>>, std::_Rb_tree_iterator<std::pair<const uint64_t, int>>> hits1 = TThits.moduleID_equal_range (module1);
    std::pair<std::_Rb_tree_iterator<std::pair<const uint64_t, int>>, std::_Rb_tree_iterator<std::pair<const uint64_t, int>>> hits2 = TThits.moduleID_equal_range (module2);

    //loop over module1 SP
    for (std::_Rb_tree_iterator<std::pair<const uint64_t, int>> SPi_mod1 = hits1.first; SPi_mod1 != hits1.second; SPi_mod1++) {
      int SP1 = SPi_mod1->second;
      TThits.get (SP1);
      uint64_t SP1_hitID = TThits.hit_id();

      T phi_SP1 = TThits.Phi();
      T eta_SP1 = TThits.Eta();

      if (_phi_slice && (phi_SP1 > _phi_slice_cut2 || phi_SP1 < _phi_slice_cut1)) continue;
     	if (_eta_region && (eta_SP1 > _eta_cut2 || eta_SP1 < _eta_cut1)) continue;

      // loop over module2 SP
      for (std::_Rb_tree_iterator<std::pair<const uint64_t, int>> SPi_mod2 = hits2.first; SPi_mod2 != hits2.second; SPi_mod2++) {
        nb_connexions++;
        int SP2 = SPi_mod2->second;
        TThits.get (SP2);
        T phi_SP2 = TThits.Phi();
		    T eta_SP2 = TThits.Eta();

//        if (_strip_hit_pair) {
//          strip_hit_pair<data_type1,data_type2> pair (TThits, SP1, SP2, _strip_module_DB);
//          if (!pair.pass_strip_range()) continue;
//        }

        if (_phi_slice && (phi_SP2 > _phi_slice_cut2 || phi_SP2 < _phi_slice_cut1)) continue;
        if (_eta_region && (eta_SP2 > _eta_cut2 || eta_SP2 < _eta_cut1)) continue;

        //applied cuts
        geometric_cuts<T> cuts (TThits, SP1, SP2);
        if (it2->second.cuts(cuts)) continue;
        nb_pass++;

        M1_SP[nb_elements].push_back (SP1);
        M2_SP[nb_elements].push_back (SP2);
        vertex[nb_elements].push_back (false);
//        cuts[ModuleDoublet.second].push_back (cuts);
      }
    }
      nb_elements++;
  }

  end_doublets = clock();
  std::cout << "cpu time for doublets loop: " << (long double)(end_doublets-start_doublets)/CLOCKS_PER_SEC << std::endl;
  std::cout << green << "nb connexions = " << nb_connexions << reset;
  nb_kills = nb_connexions - nb_pass;
  std::cout << "nb pass = " << nb_pass << std::endl;
  std::cout << red << "nb kills = " << nb_kills << reset;
  std::cout << blue << "% kills = " << 100. * (T) nb_kills / (T) nb_connexions << reset;
  std::cout << blue << "% pass = " << 100. * (T) nb_pass / (T) nb_connexions << reset;

  nb_connexions = nb_pass = nb_kills = 0;

  // -------------------------
  // loop over module triplets
  // -------------------------

  start_triplets = clock();
  for (std::_Rb_tree_iterator<std::pair<const std::vector<uint64_t>, module_triplet<T>>> it3 = _module_map_triplet.map_triplet().begin(); it3 != _module_map_triplet.map_triplet().end(); it3++) {
//  for (std::pair<std::vector<uint64_t>, module_triplet<T,U>> ModuleTriplet : _module_map_triplet.map_triplet()) {

    std::vector<uint64_t> modules12 = {it3->first[0], it3->first[1]};
    std::vector<uint64_t> modules23 = {it3->first[1], it3->first[2]};
//    std::vector<uint64_t> modules12 {ModuleTriplet.first[0], ModuleTriplet.first[1]};
//    std::vector<uint64_t> modules23 {ModuleTriplet.first[1], ModuleTriplet.first[2]};

///    std::pair<std::_Rb_tree_iterator<std::pair<const std::vector<uint64_t>, std::vector<int>>>, std::_Rb_tree_iterator<std::pair<const std::vector<uint64_t>, std::vector<int>>>> hits_M1_M2 = hit_pair_survivors.equal_range (modules12);
///    std::pair<std::_Rb_tree_iterator<std::pair<const std::vector<uint64_t>, std::vector<int>>>, std::_Rb_tree_iterator<std::pair<const std::vector<uint64_t>, std::vector<int>>>> hits_M2_M3 = hit_pair_survivors.equal_range (modules23);

    int ind12 = _module_map_triplet.map_pairs().find(modules12)->second;
    int ind23 = _module_map_triplet.map_pairs().find(modules23)->second;
    std::vector<int> hits_M1 = M1_SP[ind12];
    std::vector<int> hitsM2_M1_M2 = M2_SP[ind12];
    std::vector<int> hitsM2_M2_M3 = M1_SP[ind23];
    std::vector<int> hits_M3 = M2_SP[ind23];
//    std::vector<int> hits_M2;

//    std::vector<int> hits_M3 = M2_SP[MMD.find(modules23)->second];
//    std::vector<int> hits_M2;

    if (hitsM2_M1_M2.size() && hitsM2_M2_M3.size()) {
      std::pair<std::_Rb_tree_iterator<std::pair<const uint64_t, int>>, std::_Rb_tree_iterator<std::pair<const uint64_t, int>>> v_hits;

//      std::set_intersection (hitsM2_M1_M2.begin(), hitsM2_M1_M2.end(), hitsM2_M2_M3.begin(), hitsM2_M2_M3.end(), hits_M2.begin());
      for (int i=0; i<hitsM2_M1_M2.size(); i++) {
        std::vector<int>::iterator it = find (hitsM2_M2_M3.begin(), hitsM2_M2_M3.end(), hitsM2_M1_M2[i]);
        if (it == hitsM2_M2_M3.end()) continue;

        int SP1 = hits_M1[i];
        int SP2 = hitsM2_M1_M2[i];
        geometric_cuts<T> cuts_bc (TThits, SP1, SP2);
        if (it3->second.modules12().cuts(cuts_bc)) continue;

//        T dr_bc = TThits.R(SP2) - TThits.R(SP1);
//        T dz_bc = TThits.z(SP2) - TThits.z(SP1);

        // look for common hits on M2 module between M1-M2 and M2-M3
        for (int j = it - hitsM2_M2_M3.begin(); j<hits_M3.size() && hitsM2_M2_M3[j]==SP2; j++) {
          if (vertex[ind12][i] * vertex[ind23][j]) continue;
          int SP3 = hits_M3[j];
          nb_connexions++;

          geometric_cuts<T> cuts_ct (TThits, SP2, SP3);
          if (it3->second.modules23().cuts(cuts_ct)) continue;

          // cuts on dy_dx and dy_dz
          T diff_dydx = Diff_dydx (TThits, SP1, SP2, SP3);
//            if (diff_dydx < ModuleTriplet.second.diff_dydx_min()  ||  diff_dydx > ModuleTriplet.second.diff_dydx_max()) continue;
          if (diff_dydx < it3->second.diff_dydx_min()  ||  diff_dydx > it3->second.diff_dydx_max()) continue;

          T diff_dzdr = Diff_dzdr (TThits, SP1, SP2, SP3);
//  		      if ((diff_dzdr < ModuleTriplet.second.diff_dzdr_min()) || (diff_dzdr > ModuleTriplet.second.diff_dzdr_max())) continue;
		      if ((diff_dzdr < it3->second.diff_dzdr_min()) || (diff_dzdr > it3->second.diff_dzdr_max())) continue;

nb_triplets++;

 		      /////Graph creation
  	      /// 2 edges to add: bottom -> central and central -> top  
          /// 3 nodes to add: v1 -> v2 -> v3
 		      ///////////////////////////////////////////////////////
          ///////////////////CENTRAL TO TOP//////////////////////
          ///////////////////////////////////////////////////////

 		      //----------------------
          // Graph creation: nodes
          //----------------------
          // So far, I did not find a way to check if a node already exist based on the hit_id feature
 		      // So I had to map the node vertex id to the hit id

          TThits.get (SP2);
 		      vertex_type<T> v2;
          HitGraphContainer hit2_in_graph = hit_to_node.find (TThits.hit_id());

          if (hit2_in_graph == hit_to_node.end()) {
          	v2 = G.add_node (TThits);
  	        G_true.add_node (TThits);
            hit_to_node.insert (std::pair<uint64_t,vertex_type<T>> (TThits.hit_id(), v2));
          }
 		      else
            v2 = hit2_in_graph->second;

          TThits.get (SP3);
          vertex_type<T> v3;
          HitGraphContainer hit3_in_graph = hit_to_node.find (TThits.hit_id());
          if (hit3_in_graph == hit_to_node.end()) {
            v3 = G.add_node (TThits);
            G_true.add_node (TThits);
            hit_to_node.insert (std::pair<uint64_t,vertex_type<T>> (TThits.hit_id(), v3));
          }
          else
            v3 = hit3_in_graph->second;

          //---------------------
          // Graph creation: edge
          //---------------------
          // create an edge if it does not already exist
          TThits.get (SP3);
          T dr_ct = TThits.R() - TThits.R (SP2);
          T dz_ct = TThits.z() - TThits.z(SP2);

          //HERE MODIF
          if (!G.get_edge(v2,v3).second) {
            edge<T> edg (cuts_ct.d_eta(), cuts_ct.d_phi(), dr_ct, dz_ct, cuts_ct.phi_slope(), cuts_ct.r_phi_slope());
            if (_strip_hit_pair)
              edg.barrel_strip_hit (TThits, SP2, SP3, _strip_module_DB);
            G.add_edge (v2, v3, edg);
          }

          if (_true_graph) {
            v_hits = HitsInfos.particleID_equal_range (TThits.particle_ID (SP2));
            if (!G_true.get_edge (v2, v3).second)
              G_true.add_edge (v2, v3, true_edges_number, _min_pt, _min_N_hits, SP2, SP3, v_hits, TThits, TTp);
            else if (G_true[G_true.get_edge (v2, v3).first].is_segment_true() == 0) // the edge exist already. As there are shared hits, let's check if the edge is a true one
              G_true.modify_flag (v2, v3, true_edges_number, _min_pt, _min_N_hits, SP2, SP3, v_hits, TThits, TTp);
          }

          ///////////////////////////////////////////////////////
          ///////////////////BOTTOM TO CENTRAL///////////////////
          ///////////////////////////////////////////////////////

          //----------------------
          // Graph creation: nodes
          //----------------------
          // So far, I did not find a way to check if a node already exist based on the hit_id feature
          // So I had to map the node vertex id to the hit id

          vertex_type<T> v1;
          TThits.get (SP1);
          T dr_bc = TThits.R(SP2) - TThits.R();
          T dz_bc = TThits.z(SP2) - TThits.z();
          HitGraphContainer hit1_in_graph = hit_to_node.find (TThits.hit_id());
          if (hit1_in_graph == hit_to_node.end()) {
            v1 = G.add_node (TThits);
            G_true.add_node (TThits);
            hit_to_node.insert (std::pair<uint64_t,vertex_type<T>> (TThits.hit_id(), v1));
          }
          else
            v1 = hit1_in_graph->second;

          //v2 is already known

          //---------------------
          // Graph creation: edge
          //---------------------
          // create an edge if it does not already exist

          // HERE MODIF
          if (!G.get_edge (v1, v2).second) {
            edge<T> edg (cuts_bc.d_eta(), cuts_bc.d_phi(), dr_bc, dz_bc, cuts_bc.phi_slope(), cuts_bc.r_phi_slope());
            if (_strip_hit_pair)
              edg.barrel_strip_hit (TThits, SP2, SP1, _strip_module_DB);
            G.add_edge (v1, v2, edg);
          }

          if (_true_graph) {
            if (!G_true.get_edge(v1, v2).second) {
              G_true.add_edge (v1, v2, true_edges_number, _min_pt, _min_N_hits, SP1, SP2, v_hits, TThits, TTp);
            } else if (!G_true[G_true.get_edge (v1, v2).first].is_segment_true()) { // the edge exist already. As there are shared hits, let's check if the edge is a true one
              G_true.modify_flag (v1, v2, true_edges_number, _min_pt, _min_N_hits, SP1, SP2, v_hits, TThits, TTp);
            }
          }
          nb_pass++;
          vertex[ind12][i] = vertex[ind23][j] = true;
        }
      }
    }
  }

  delete [] M1_SP;
  delete [] M2_SP;
  delete [] vertex;

		    /*
		    cout << "begin scoring" << endl;
                    /// Seeder scoring 

                    float Rcentrale = R(x_centralSP, y_centralSP);

                    float Ri    = 1./Rcentrale          ;
                    float ax    = x_centralSP*Ri           ;
                    float ay    = y_centralSP*Ri ;


                    float dxt  = t_hits.x-x_centralSP   ;
                    float dyt  = t_hits.y-y_centralSP   ;
                    float xt   = dxt*ax+dyt*ay ;
                    float yt   = dyt*ax-dxt*ay ;
                    float dxyt = xt*xt+yt*yt     ;
                    float r2t  = 1./dxyt      ;
                    float Ut   = xt*r2t        ;
                    float Vt   = yt*r2t        ;

                    float dxb  = b_hits.x-x_centralSP   ;
                    float dyb  = b_hits.y-y_centralSP   ;
                    float xb   = dxb*ax+dyb*ay ;
                    float yb   = dyb*ax-dxb*ay ;
                    float dxyb = xb*xb+yb*yb     ;
                    float r2b  = 1./dxyb      ;
                    float Ub   = xb*r2b        ;
                    float Vb   = yb*r2b        ;

                    float A   = (Vt-Vb)/ (Ut - Ub);
                    float B   = Vb-A*Ub;

                    float d0 = std::abs((A-B*Rcentrale)*Rcentrale);
                    float Q = 100. * d0 + (std::abs(b_hits.z0) - float(vector_top_links.size()) * 100.);
		    cout << "end scoring" << endl;
                 

                    if (G_true[boost::edge(v1, v2, G_true).first].is_segment_true == 1 && G_true[boost::edge(v2, v3, G_true).first].is_segment_true == 1){
                        th1d_seederScore_true->Fill(Q);
                        triplets_score_trueEdges.push_back(Q);
                    }else{
                        th1d_seederScore_false->Fill(Q);
                        triplets_score_fakeEdges.push_back(Q);
                    }
	
        //std::cout<<"\n true edges scores"<<std::endl;
        //for(auto& score : triplets_score_trueEdges) std::cout<<score<<std::endl;

        //std::cout<<"fake edges scores"<<std::endl;
        //for(auto& score : triplets_score_fakeEdges) std::cout<<score<<std::endl;
    	}  // endif loop_edges
      */
  
  end_triplets = clock();
  std::cout << "There are " << nb_triplets << " triplets" << std::endl;

  std::cout << cyan << "cpu time for doublets selection (loop only - no i/o): " << (long double)(end_doublets - start_doublets)/CLOCKS_PER_SEC << reset;
  std::cout << cyan << "cpu time for triplets selection (loop only - no i/o): " << (long double)(end_triplets-start_triplets)/CLOCKS_PER_SEC << reset;
  std::cout << magenta << "cpu time for global selection (loop only - no i/o): " << (long double)(end_triplets-start_doublets)/CLOCKS_PER_SEC << reset;

    std::cout << event_id << " " << "There are " << G.num_edges() << " edges" << std::endl;
// to do
    std::cout << event_id << " " << "There are " << true_edges_number << " true edges" << std::endl;
    std::cout << event_id << " " << "There are " << G.num_vertices() << " nodes" << std::endl;

  std::cout << green << "nb connexions = " << nb_connexions << reset;
  nb_kills = nb_connexions - nb_pass;
  std::cout << "nb pass = " << nb_pass << std::endl;
  std::cout << red << "nb kills = " << nb_kills << reset;
  std::cout << blue << "% kills = " << 100. * (T) nb_kills / (T) nb_connexions << reset;
  std::cout << blue << "% pass = " << 100. * (T) nb_pass / (T) nb_connexions << reset;

  std::cout << green << "nb connexions = " << nb_connexions << reset;
  nb_kills = nb_connexions - nb_pass;
  std::cout << "nb pass = " << nb_pass << std::endl;
  std::cout << red << "nb kills = " << nb_kills << reset;
  std::cout << blue << "% kills = " << 100. * (T) nb_kills / (T) nb_connexions << reset;
  std::cout << blue << "% pass = " << 100. * (T) nb_pass / (T) nb_connexions << reset;

    /*
    string out = "/sps/l2it/crougier/GitLab/GNN4ITkTeam/l2it_acts/ACTS/out/test_seederScoring.root";
    TFile* rootFile = TFile::Open(out.c_str(), "RECREATE");
    rootFile->cd();
    th1d_seederScore_true->Write();
    th1d_seederScore_false->Write();
    rootFile->Close();
    */

  std::cout << event_id << " " << std::setprecision(20) << edges << std::endl;

  if (_save_graph_graphml) {
    std::cout << "writing output in directory " << _output_dir << std::endl;
    std::cout << "event id = " << event_id << std::endl;
    G.save (event_id, _output_dir, _extra_features);
    if (_true_graph) G_true.save (event_id, _output_dir);
  }

  if (_save_graph_npz) {
#ifdef MMG_WITH_NUMPY_EXPORT
    G.save_npz (event_id, _output_dir, _extra_features);
    if (_true_graph) G_true.save_npz (event_id, _output_dir);
#else
    std::cout << "Numpy export not compiled, enable by MMG_WITH_NUMPY_EXPORT=ON in cmake" << std::endl;
#endif
  }

  if (_save_graph_pyg) {
#ifdef MMG_WITH_PYTORCH_EXPORT
    G.save_pyg (event_id, _output_dir, _extra_features);
    //if (_true_graph) G_true.save_pyg (event_id, _output_dir);
#else
    std::cout << "Pytorch export not compiled, enable by MMG_WITH_PYTORCH_EXPORT=ON in cmake" << std::endl;
#endif
  }

  if (_save_graph_csv) {
    G.save_csv (event_id, _output_dir, _extra_features);
//    if (_true_graph) G_true.save_csv (event_id, _output_dir);
  }

  end_main = clock();
  std::cout << "cpu time for GraphCreatorWriterModuleTriplet: " << (long double)(end_main-start_main)/CLOCKS_PER_SEC << std::endl;
}

