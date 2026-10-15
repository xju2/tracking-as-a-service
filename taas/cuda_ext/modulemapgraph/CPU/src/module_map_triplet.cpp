/***********************************************************************
 * Module Map Graph library
 * Track reconstruction for HEP
 * copyright © 2022,2023 ROUGIER Charline
 * copyright © 2022,2023,2024,2025,2026 COLLARD Christophe
 * Laboratoire des 2 Infinis - Toulouse - France (L2IT)
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 **********************************************************************/

#include "MMG/module_map_triplet.hpp"


//=====Private methods for module map triplet===========================================

template <class T>
void module_map_triplet<T>::init(boost::program_options::variables_map& po_vm)
{
  _min_N_hits = po_vm["min-nhits"].as<long unsigned int> ();
  _min_pt = po_vm["min-pt-cut"].as<T> ();
  _max_pt = po_vm["max-pt-cut"].as<T> ();

  _strip_hit_pair = po_vm["strip-hit-pair"].as<bool> ();
  _drsel = static_cast<DRsel>(po_vm["drsel"].as<int>());
  _closest_to_vertex = po_vm["closest-to-vertex"].as<bool> ();

  // Initialise list of events to be ignored
  // ---------------------------------------
  _event_veto = {};

  std::string events_veto = po_vm["event-veto"].as<std::string>();
  if(events_veto != ""){
    std::ifstream events_to_ignore(events_veto);

    if (!events_to_ignore.is_open())
      throw std::invalid_argument("Cannot open file "+events_veto);

    std::cout<<"Ignoring events listed in "<<events_veto<<std::endl;
    ULong64_t event(0);
    while(events_to_ignore >> event){
      _event_veto.insert(event);
    }
    events_to_ignore.close();
  }

  // Initialise list of modules to be ignored
  // ----------------------------------------
  _dead_module_ids = {};
  std::string  list_dead_modules = po_vm["input-file-dead-modules"].as<std::string>();
  if(list_dead_modules != ""){
    std::ifstream modules_to_ignore(list_dead_modules);

    if(!modules_to_ignore.is_open())
      throw std::invalid_argument("Cannot open file "+list_dead_modules);

    std::cout<<"Ignoring modules listed in "<<list_dead_modules<<std::endl;
    uint64_t module_ID(0);    
    while(modules_to_ignore >> module_ID){
      _dead_module_ids.insert(module_ID);
    }
    modules_to_ignore.close();
  }

  if (_drsel == POS_dr) std::cout << "Keeping only triplets with dr >= 0 " << std::endl;
  if (_drsel == STRICTPOS_dr) std::cout << "Keeping only triplets with dr > 0 " << std::endl;
}


template <class T>
void module_map_triplet<T>::make_module_map_singlet()
{
  int loop = 0;
  for (std::pair<std::vector<uint64_t>, module_doublet<T>> ModuleDoublet : _module_map_doublet) {
    
    _module_map_pairs.insert (std::pair<std::vector<uint64_t>, int> (ModuleDoublet.first, loop));
    loop++;
    if (_module_map.find(ModuleDoublet.first[0]) == _module_map.end())
      _module_map.insert (std::pair<uint64_t, int> (ModuleDoublet.first[0],0));
    if (_module_map.find(ModuleDoublet.first[1]) == _module_map.end())
      _module_map.insert (std::pair<uint64_t, int> (ModuleDoublet.first[1],0));
  }

  loop=0;
  for (std::multimap<uint64_t,int>::iterator it= _module_map.begin(); it != _module_map.end(); ++it, loop++) {
    it->second = loop;
    loop++;
  }
  std::cout << green << "# modules in MM = " << _module_map.size() << reset;
}

//=====Public methods for module map triplet============================================

// -----------------------------------------------------
// To build a module map directly from a Root Dump file
// -----------------------------------------------------
template <class T>
module_map_triplet<T>::module_map_triplet(boost::program_options::variables_map& po_vm)
{
  // Basic initialization
  init(po_vm);

  // Open dump file
  std::string dump_file = po_vm["input-dump"].as<std::string>();
  TFile* f = TFile::Open(dump_file.c_str());

  if(!f){
    throw std::invalid_argument ("Cannot open dump file "+dump_file);
  }

  TTree* dump_tree = (TTree*) f->Get("GNN4ITk");

  if(!dump_tree){
    throw std::invalid_argument ("Cannot get TTree GNN4ITk in dump file "+dump_file);
  }

  std::cout<<"Setting up Root Dump reading for file "<<dump_file<<std::endl;
  TTree_dump dump(dump_tree);

  int Nevents = dump_tree->GetEntries();

  std::cout<<"Processing "<<Nevents<<" events"<<std::endl;
  for(int i(0); i<Nevents; ++i){
    dump.GetEntry(i);

    // Some sanity checks on the event
    if(dump.nPartEVT==0){
      std::cout<<"Event "<<dump.event_number<<" has no particle, ignoring it."<<std::endl;
      continue;
    }

    if(dump.nSP==0){
      std::cout<<"Event "<<dump.event_number<<" has no Space Point, ignoring it."<<std::endl;
      continue;
    }

    if(dump.nCL==0){
      std::cout<<"Event "<<dump.event_number<<" has no particle, ignoring it."<<std::endl;
      continue;
    }

    int nlinks=0;
    for(const auto& links : *dump.CLparticleLink_barcode){
        nlinks += links.size();
        if(nlinks>0) break; // to speed up the check
    }
    if(nlinks==0){
        std::cout<<"WARNING : CL->particles links are all empty in this event, ignoring it!"<<std::endl;
        continue;
    }

    if(!_event_veto.empty()){
      if( _event_veto.find(dump.event_number) != _event_veto.end() ) continue;
    }

    // Make TTree_particles and TTree_hits
    TTree_particles<T> TTp(dump);
    TTree_hits<T> TTh(dump,_dead_module_ids);

    // Sanity check in case we are ignoring some modules,
    // we can end up without any space points
    if(!_dead_module_ids.empty() && TTh.size()==0){
      std::cout<<"WARNING : After ignorning some modules, no space points are left ignoring event "<<dump.event_number<<std::endl;
      continue;
    }

    add_event_triplets (TTp, TTh, _min_N_hits, _min_pt, _max_pt);
  }

  // map doublets and modules
  make_module_map_singlet();
}

//-------------------------------------------------------------------------------------------------------------------------
template <class T>
module_map_triplet<T>::module_map_triplet (boost::program_options::variables_map& po_vm, const std::string& hits_filenames, 
                                           const std::string& particles_filenames)
//-------------------------------------------------------------------------------------------------------------------------
{
  std::string events_path = po_vm["input-dir"].as<std::string> () + "/";
  std::string module_map_dir = po_vm["input-module-map"].template as<std::string> ();
  
  // Basic initialization
  init(po_vm);

  // open files containing events filenames
  std::ifstream h_file (hits_filenames);
  if (h_file.fail()) throw std::invalid_argument ("Cannot open file " + hits_filenames);
  std::ifstream p_file (particles_filenames);
  if (p_file.fail()) throw std::invalid_argument ("Cannot open file " + particles_filenames);

  std::string hits_filename, particles_filename;
  h_file >> hits_filename;
  p_file >> particles_filename;

  for (; !h_file.eof() && !p_file.eof();) {
    std::string event_id = boost::regex_replace (hits_filename, boost::regex("[^0-9]*([0-9]+).*"), std::string("\\1"));
    if (event_id != boost::regex_replace (particles_filename, boost::regex("[^0-9]*([0-9]+).*"), std::string("\\1")))
      throw std::invalid_argument ("hits and particules file are not part of the same event");

    add_event_triplets (events_path, hits_filename, particles_filename, _min_N_hits, _min_pt, _max_pt);
    h_file >> hits_filename;
    p_file >> particles_filename;
  }

  h_file.close();
  p_file.close();

  // map doublets and modules
  make_module_map_singlet();
}

//----------------------------------------------------------------------------------------------------------------------
template <class T>
void module_map_triplet<T>::add_event_triplets (TTree_particles<T>& TTp, TTree_hits<T>& TThits, 
                                                const int& min_N_hits, const T& min_pt, const T& max_pt)
//----------------------------------------------------------------------------------------------------------------------
{
  // Retrieve the hits info for all particles, with a specific handling of split clusters
  // When several clusters of a same particle are on the same module, 
  // we keep only the farest to the interaction vertex
  hits_infos<T> HitsInfos (TThits, TTp, _closest_to_vertex);


  //std::ofstream debug_file("debug_particles_seen_event"+std::to_string(TTp.event_ID())+".txt");

  // loop on particles (faster - no particles set to 0 like in hits)
  for (int pID=0; pID<TTp.size(); pID++){
    TTp.get (pID);
    uint64_t particleID = TTp.particle_ID();

    //----------
    // cut on pt
    //----------
    if (0.001*TTp.pT() < min_pt) continue;
    if (max_pt && 0.001*TTp.pT() > max_pt) continue;

    //------------------------------------
    // cut on barcode (ignore secondaries)
    //------------------------------------
    if (TTp.barcode() > 200000) continue;

    //---------------------------
    // We do not follow electrons
    //---------------------------
    if (abs (TTp.pdgID()) == 11) continue;

    //-------------------------------------------------
    // Preselection on number of hits (at least 2 hits)
    //-------------------------------------------------
    std::pair<std::_Rb_tree_iterator<std::pair<const uint64_t, int>>, std::_Rb_tree_iterator<std::pair<const uint64_t, int>>> HitRange = HitsInfos.particleID_equal_range (particleID); // test on HitsInfo (should be faster)
    int size = distance (HitRange.first, HitRange.second);
    if (size < min_N_hits) continue; //remove particles leaving less than min_N_hits here

    // --------------------------
    // Ignore particle of eta > 4
    // --------------------------
    if (abs (TTp.eta ()) > 4) continue;

    // ------------------------------------------------------------
    // Ignore particles produced in a radius 260mm away of the beam
    // ------------------------------------------------------------
    if (std::sqrt(TTp.vx()*TTp.vx() + TTp.vy()*TTp.vy()) > 260) continue;

    //debug_file << particleID << " , " << size << " ; ";//<< std::endl;

    //--------------------
    // Module Map creation
    //--------------------

    for (std::_Rb_tree_iterator<std::pair<const uint64_t, int>> hit = HitRange.first; hit != HitRange.second; hit++){
      int it1 = hit->second;
      //debug_file<< TThits.hit_id(it1)<<" ";
      std::_Rb_tree_iterator<std::pair<const uint64_t, int>> next_hit, next_next_hit;
      next_next_hit = next_hit = hit;
      int it2 = (++next_hit)->second;
      if (next_hit == HitRange.second) {
        //debug_file<< TThits.hit_id(it2)<<" ";
        break;
      }

      int it3 = (++(++next_next_hit))->second;
      if (next_next_hit == HitRange.second){
        //debug_file<< TThits.hit_id(it2)<<" ";
        break;
      } 

      uint64_t module1 = TThits.module_ID(it1);
      uint64_t module2 = TThits.module_ID(it2);
      uint64_t module3 = TThits.module_ID(it3);
      const std::vector<uint64_t> module = {module1, module2, module3};

      geometric_cuts<T> cuts12 (TThits, it1, it2);
      geometric_cuts<T> cuts23 (TThits, it2, it3);

      cuts12.check_cut_values ();
      cuts23.check_cut_values ();

      // If desired, ignore triplets going backwards or with dr==0
      if (_drsel == POS_dr && (cuts12.dr()<0 || cuts23.dr()<0)) continue;
      if (_drsel == STRICTPOS_dr && (cuts12.dr()<=0 || cuts23.dr()<=0)) continue;

      T diff_dzdr = Diff_dzdr (TThits, it1, it2, it3);
      T diff_dydx = Diff_dydx (TThits, it1, it2, it3);

      add_triplet_occurence (module, TThits, TTp, cuts12, cuts23, diff_dzdr, diff_dydx, it1, it2, it3);
      
    }
    //debug_file<<std::endl;
  }

  //debug_file.close();
}

//----------------------------------------------------------------------------------------------------------------------
template <class T>
void module_map_triplet<T>::add_event_triplets (const std::string& path, 
                                                const std::string& hits_filename, const std::string& particles_filename, 
                                                const int& min_N_hits, const T& min_pt, const T& max_pt)
//----------------------------------------------------------------------------------------------------------------------
{
  // load event hits
  hits<T> hit_pts;  // define in class data ?
  TTree_hits<T> TThits;
  std::filesystem::path hits_path = hits_filename;
  if (hits_path.extension () == ".csv")
    { hit_pts.read_csv (path, hits_filename);
      TThits = hit_pts;
    }
  else if (hits_path.extension () == ".root")
    TThits.read (path + hits_filename);
  else throw std::invalid_argument ("unknown extension for hit file" + hits_filename);

  // load event particles
  particles<T> p;
  TTree_particles<T> TTp;
  std::filesystem::path particles_path = particles_filename;
  if (particles_path.extension () == ".csv")
    { p.read_csv (path, particles_filename);
      TTp = p;
    }
  else if (particles_path.extension () == ".root")
    TTp.read (path + particles_filename);
  else throw std::invalid_argument ("unknown extension for particles file" + particles_filename);

  add_event_triplets(TTp,TThits,min_N_hits,min_pt,max_pt);
}


//---------------------------------------------------------------------------------------
template <class Tf>
bool operator == (const module_map_triplet<Tf>& mmt1, const module_map_triplet<Tf>& mmt2)
//---------------------------------------------------------------------------------------
{
  return (mmt1._module_map_triplet == mmt2._module_map_triplet);
}


//---------------------------------------------------------------------------------------
template <class Tf>
bool operator != (const module_map_triplet<Tf>& mmt1, const module_map_triplet<Tf>& mmt2)
//---------------------------------------------------------------------------------------
{
  return !(mmt1 == mmt2);
}


//-------------------------------------------------------------------------------------------------------------------------
template <class T>
bool module_map_triplet<T>::add_triplet_occurence (const std::vector<uint64_t>& triplet, 
                                                   const TTree_hits<T>& TThits, const TTree_particles<T>& TTp, 
                                                   const geometric_cuts<T>& cuts12, const geometric_cuts<T>& cuts23, 
                                                   const T& diff_dzdr, const T& diff_dydx, int it1, int it2, int it3)
//---------------------------------------------------------------------------------------------------------------------------
{
  bool new_particle = true;

  std::pair <std::_Rb_tree_iterator<std::pair <const std::vector<uint64_t>, std::vector<T>>>, std::_Rb_tree_iterator <std::pair <const std::vector<uint64_t>, std::vector<T>>>>  mod_triplet = _modules_particlesKey.equal_range (triplet);
  int size = distance (mod_triplet.first, mod_triplet.second);
  std::vector<T> key; // = TTp.key();
  key.push_back (TTp.pT());
  key.push_back (TTp.pdgID());
  key.push_back (TTp.eta());
  key.push_back (TTp.vx());
  key.push_back (TTp.vy());
  key.push_back (TTp.vz());

  // Determine if this particle was already seen or not (in pile-up events we reuse some events)
  for (std::_Rb_tree_iterator<std::pair <const std::vector<uint64_t>, std::vector<T>>> it = mod_triplet.first; it != mod_triplet.second && new_particle; it++)
    new_particle *= (it -> second != key);

  // Keep record of particles seen
  if (new_particle) {
    _modules_particlesKey.insert (std::pair <std::vector<uint64_t>, std::vector<T>> (triplet, key));
    if (_particlesSPECS_TTp_link.find (key) == _particlesSPECS_TTp_link.end())
      _particlesSPECS_TTp_link.insert (std::pair <std::vector<T>, int> (key, _ModMap_particles.size()));
    _cuts12.push_back (cuts12);
    _cuts23.push_back (cuts23);
  }

  std::vector<uint64_t> doublet12 {triplet.at(0), triplet.at(1)};
  std::vector<uint64_t> doublet23 {triplet.at(1), triplet.at(2)};

  // First time we see this triplet
  if (!size) {
    if (_module_map_doublet.find (doublet12) == _module_map_doublet.end())
      _module_map_doublet.insert (std::pair <std::vector<uint64_t>, module_doublet<T>> (doublet12, module_doublet<T> (cuts12)));
    else
      (_module_map_doublet.find (doublet12) -> second).add_occurence (cuts12);
    if (_module_map_doublet.find (doublet23) == _module_map_doublet.end())
      _module_map_doublet.insert (std::pair <std::vector<uint64_t>, module_doublet<T>> (doublet23, module_doublet<T> (cuts23)));
    else
      (_module_map_doublet.find (doublet23) -> second).add_occurence (cuts23);
    _module_map_triplet.insert (std::pair <std::vector<uint64_t>, module_triplet<T>> (triplet, module_triplet<T> (cuts12, cuts23, diff_dzdr, diff_dydx)));
  }
  else {
    (_module_map_doublet.find (doublet12) -> second).add_occurence (cuts12);
    (_module_map_doublet.find (doublet23) -> second).add_occurence (cuts23);
    (_module_map_triplet.find (triplet) -> second).add_occurence (cuts12, cuts23, diff_dzdr, diff_dydx);
  }

  return new_particle;
}


//-------------------------------------------------------------------------------
template <class T>
void module_map_triplet<T>::merge (const module_map_triplet<T>& ModuleMapTriplet)
//-------------------------------------------------------------------------------
{
  // Loop on all entries of the module map in input
  for (const std::pair<std::vector<uint64_t>, module_triplet<T>> MM_entry : ModuleMapTriplet._module_map_triplet)
    if (_module_map_triplet.find (MM_entry.first) == _module_map_triplet.end())
      _module_map_triplet.insert (std::pair <std::vector<uint64_t>, module_triplet<T>> (MM_entry.first, MM_entry.second));
    else
      (_module_map_triplet.find (MM_entry.first) -> second).add_occurences (MM_entry.second);

  for (const std::pair<std::vector<uint64_t>, module_doublet<T>> MM_entry : ModuleMapTriplet._module_map_doublet)
    if (_module_map_doublet.find (MM_entry.first) == _module_map_doublet.end())
      _module_map_doublet.insert (std::pair <std::vector<uint64_t>, module_doublet<T>> (MM_entry.first, MM_entry.second));
    else
      (_module_map_doublet.find (MM_entry.first) -> second).add_occurences (MM_entry.second);
}


//---------------------------------
template <class T>
void module_map_triplet<T>::sort ()
//---------------------------------
{
  std::multimap <std::vector<uint64_t>, unsigned int> ModulesTriplets_to_keep;
  std::multimap <std::vector<uint64_t>, module_triplet<T>> sorted_module_map_triplet = _module_map_triplet;
int step = 0;

  for (std::pair<std::vector<uint64_t>, module_triplet<T>> MM_entry : _module_map_triplet) {
    bool unique = (MM_entry.second.occurence() == 1) || (MM_entry.second.min_equal_max());
    uint64_t module1 = MM_entry.first[0];
    uint64_t module2 = MM_entry.first[1];
    uint64_t module3 = MM_entry.first[2];
    bool add = true;

    for (std::pair<std::vector<uint64_t>, unsigned int> modules : ModulesTriplets_to_keep) {
      uint64_t m1 = modules.first[0];
      uint64_t m2 = modules.first[1];
      uint64_t m3 = modules.first[2];
      unsigned int unique2 = modules.second;
      if (unique2 > 1) std::cout << "unique2 = "  << unique2 << std::endl;

//      if (module1 == m3) && ((module2 == m1 || module2 == m2 || )
      if (((m1 == module2 && m2 == module1)  ||  (m2 == module2) && (m1 == module3)) ||
          ((m3 == module2 && m2 == module3)  ||  (m2 == module2) && (m3 == module1)) ||
          ((m3 == module1 && m1 == module2)  ||  (m3 == module2) && (m1 == module3))) {
            add = false;
            if (unique2 && !unique) sorted_module_map_triplet.erase ({m1, m2, m3});
            else if (unique && !unique2) sorted_module_map_triplet.erase ({module1, module2, module3});
            else if (unique > unique2) sorted_module_map_triplet.erase ({m1, m2, m3});
            else sorted_module_map_triplet.erase ({module1, module2, module3});
      }
    }

//    if (add) ModulesTriplets_to_keep.insert (std::pair<std::vector<uint64_t>, unsigned int> ({module1, module2, module3}, MM_entry.second.occurence()));
    if (add) ModulesTriplets_to_keep.insert (std::pair<std::vector<uint64_t>, unsigned int> ({module1, module2, module3}, unique));
//      std::cout << "step = " << ++step << std::endl;
  }
  _module_map_triplet = sorted_module_map_triplet;
}


/*
//-----------------------------------
template <class T>
void module_map_triplet<T>::remove_cycles()
//-----------------------------------
{
  std::multimap <std::vector<uint64_t>, unsigned int> ModulesTriplets_to_keep;
  std::multimap <std::vector<uint64_t>, module_triplet<T>> sorted_module_map_triplet = _module_map_triplet;

  // Loop on all initial MM entries
  // ------------------------------
  for (std::pair<std::vector<uint64_t>, module_triplet<T>> MM_entry : _module_map_triplet) {

    // Force occurence to 1 if for some cuts min==max (same particle was seen several times, making occurence > 1)
    unsigned int occurence_unique = (MM_entry.second.min_equal_max())? 1 : MM_entry.second.occurence();
    uint64_t module1 = MM_entry.first[0];
    uint64_t module2 = MM_entry.first[1];
    uint64_t module3 = MM_entry.first[2];
    bool add = true;

    // Loop on all triplets already kept
    // if one cycle will appear, we remove the triplet with less occurence (or only one)
    for (std::pair<std::vector<uint64_t>, unsigned int> modules : ModulesTriplets_to_keep) {


      uint64_t m1 = modules.first[0];
      uint64_t m2 = modules.first[1];
      uint64_t m3 = modules.first[2];
      unsigned int occurence_unique2 = modules.second;

      //if (unique2 > 1) std::cout << "unique2 = "  << unique2 << std::endl;

      // //CHarline's if
      // if ( ( m1 == module2 && m2 == module1)  || ( m2 == module2 && m1 == module3) ){
      // else if ((m3 == module1 && m2 == module2) || (m3==module2 && m2 == module3)){
      // else if ((m3 == module1 && m1 == module2) || (m3 == module2 && m1 == module3) ){

      if (((m1 == module2 && m2 == module1)  ||  (m2 == module2) && (m1 == module3)) ||
          ((m3 == module2 && m2 == module3)  ||  (m2 == module2) && (m3 == module1)) ||
          ((m3 == module1 && m1 == module2)  ||  (m3 == module2) && (m1 == module3))) {
            add = false;
            if (occurence_unique2==1 && occurence_unique>1) sorted_module_map_triplet.erase ({m1, m2, m3});
            else if (occurence_unique==1 && occurence_unique2>1) sorted_module_map_triplet.erase ({module1, module2, module3});
            else if (occurence_unique > occurence_unique2) sorted_module_map_triplet.erase ({m1, m2, m3});
            else sorted_module_map_triplet.erase ({module1, module2, module3});
      }
    }

    if (add) ModulesTriplets_to_keep.insert (std::pair<std::vector<uint64_t>, unsigned int> ({module1, module2, module3}, occurence_unique));

  }
  _module_map_triplet = sorted_module_map_triplet;
}
*/
//
//----------------------------------------------------
template <class T>
void module_map_triplet<T>::remove_unique_occurence ()
//----------------------------------------------------
{
  int deleted = 0;
  for (std::_Rb_tree_iterator<std::pair<const std::vector<uint64_t>, module_triplet<T>>> it=_module_map_triplet.begin(); it!=_module_map_triplet.end(); it++) {
    unsigned int occurence = (it->second.min_equal_max()) ? 1 : it->second.occurence();
    uint64_t module1 = it->first[0];
    uint64_t module2 = it->first[1];
    uint64_t module3 = it->first[2];
    if (occurence == 1) {
      it--;
      _module_map_triplet.erase ({module1, module2, module3});
      deleted++;
    }
  }
  std::cout << "nb deleted triplets = " << deleted << std::endl;
}


//------------------------------------------
template <class T>
void module_map_triplet<T>::remove_cycles ()
//------------------------------------------
{
//  std::multimap <std::vector<uint64_t>, unsigned int> ModulesTriplets_to_keep;
//  std::multimap <std::vector<uint64_t>, module_triplet<T>> sorted_module_map_triplet = _module_map_triplet;
//  std::multimap <std::vector<uint64_t>, module_triplet<T>> sorted_module_map_triplet;
int count = 0;

  // Loop on all initial MM entries
  // ------------------------------
  for (std::_Rb_tree_iterator<std::pair<const std::vector<uint64_t>, module_triplet<T>>> it=_module_map_triplet.begin(); it!=_module_map_triplet.end(); it++) {
    unsigned int occurence_unique = (it->second.min_equal_max()) ? 1 : it->second.occurence();
    uint64_t module1 = it->first[0];
    uint64_t module2 = it->first[1];
    uint64_t module3 = it->first[2];
//    bool add = true;
      count++;
    if (count == 10000) std::cout << "10k" << std::endl;
    if (count == 100000) std::cout << "100k" << std::endl;
    if (count == 200000) std::cout << "200k" << std::endl;
    if (count == 500000) std::cout << "500k" << std::endl;
    if (count == 1000000) std::cout << "1M" << std::endl;

    for (std::_Rb_tree_iterator<std::pair<const std::vector<uint64_t>, module_triplet<T>>> it2=_module_map_triplet.begin(); it2!=it; it2++) {
      uint64_t m1 = it2->first[0];
      uint64_t m2 = it2->first[1];
      uint64_t m3 = it2->first[2];
      unsigned int occurence_unique2 = it2->second.occurence();
      std::vector<T> triplet_to_delete;

//  for (std::pair<std::vector<uint64_t>, module_triplet<T>> MM_entry : _module_map_triplet) {

    // Force occurence to 1 if for some cuts min==max (same particle was seen several times, making occurence > 1)
//    unsigned int occurence_unique = (MM_entry.second.min_equal_max())? 1 : MM_entry.second.occurence();
//    uint64_t module1 = MM_entry.first[0];
//    uint64_t module2 = MM_entry.first[1];
//    uint64_t module3 = MM_entry.first[2];
//    bool add = true;

    // Loop on all triplets already kept
    // if one cycle will appear, we remove the triplet with less occurence (or only one)
//    for (std::pair<std::vector<uint64_t>, unsigned int> modules : ModulesTriplets_to_keep) {


//      uint64_t m1 = modules.first[0];
//      uint64_t m2 = modules.first[1];
//      uint64_t m3 = modules.first[2];
//      unsigned int occurence_unique2 = modules.second;

      //if (unique2 > 1) std::cout << "unique2 = "  << unique2 << std::endl;

      // //CHarline's if
      // if ( ( m1 == module2 && m2 == module1)  || ( m2 == module2 && m1 == module3) ){
      // else if ((m3 == module1 && m2 == module2) || (m3==module2 && m2 == module3)){
      // else if ((m3 == module1 && m1 == module2) || (m3 == module2 && m1 == module3) ){

      if ((m1 == module2 && m2 == module1) || (m1 == module3 && m2 == module1) || (m1 == module3 && m2 == module2) ||
          (m1 == module2 && m3 == module1) || (m1 == module3 && m3 == module1) || (m1 == module3 && m3 == module2) ||
          (m2 == module2 && m3 == module1) || (m2 == module3 && m3 == module1) || (m2 == module3 && m3 == module2) ) {
        if (occurence_unique > occurence_unique2) {
          std::cout << "kill" << std::endl;
          it--; it2--;
          _module_map_triplet.erase ({m1, m2, m3});
        }
        else {
          it--;
          std::cout << "kill++" << std::endl;
          _module_map_triplet.erase ({module1, module2, module3});
          continue;
        }
      }
    }
  }

//    if (add) ModulesTriplets_to_keep.insert (std::pair<std::vector<uint64_t>, unsigned int> ({module1, module2, module3}, occurence_unique));

//  }
//  _module_map_triplet = sorted_module_map_triplet;
}

/*
//-----------------------------------------
template <class T>
void module_map_triplet<T>::remove_cycles()
//-----------------------------------------
{
  std::multimap <std::vector<uint64_t>, unsigned int> ModulesTriplets_to_keep;
  std::multimap <std::vector<uint64_t>, module_triplet<T>> sorted_module_map_triplet = _module_map_triplet;

  // Loop on all initial MM entries
  // ------------------------------
  for (std::pair<std::vector<uint64_t>, module_triplet<T>> MM_entry : _module_map_triplet) {

    // Force occurence to 1 if for some cuts min==max (same particle was seen several times, making occurence > 1)
    unsigned int occurence_unique = (MM_entry.second.min_equal_max())? 1 : MM_entry.second.occurence();
    uint64_t module1 = MM_entry.first[0];
    uint64_t module2 = MM_entry.first[1];
    uint64_t module3 = MM_entry.first[2];
    bool add = true;

    // Loop on all triplets already kept
    // if one cycle will appear, we remove the triplet with less occurence (or only one)
    for (std::pair<std::vector<uint64_t>, unsigned int> modules : ModulesTriplets_to_keep) {


      uint64_t m1 = modules.first[0];
      uint64_t m2 = modules.first[1];
      uint64_t m3 = modules.first[2];
      unsigned int occurence_unique2 = modules.second;

      //if (unique2 > 1) std::cout << "unique2 = "  << unique2 << std::endl;

      // //CHarline's if
      // if ( ( m1 == module2 && m2 == module1)  || ( m2 == module2 && m1 == module3) ){
      // else if ((m3 == module1 && m2 == module2) || (m3==module2 && m2 == module3)){
      // else if ((m3 == module1 && m1 == module2) || (m3 == module2 && m1 == module3) ){

      if (((m1 == module2 && m2 == module1)  ||  (m2 == module2) && (m1 == module3)) ||
          ((m3 == module2 && m2 == module3)  ||  (m2 == module2) && (m3 == module1)) ||
          ((m3 == module1 && m1 == module2)  ||  (m3 == module2) && (m1 == module3))) {
            add = false;
            if (occurence_unique2==1 && occurence_unique>1) sorted_module_map_triplet.erase ({m1, m2, m3});
            else if (occurence_unique==1 && occurence_unique2>1) sorted_module_map_triplet.erase ({module1, module2, module3});
            else if (occurence_unique > occurence_unique2) sorted_module_map_triplet.erase ({m1, m2, m3});
            else sorted_module_map_triplet.erase ({module1, module2, module3});
      }
    }

    if (add) ModulesTriplets_to_keep.insert (std::pair<std::vector<uint64_t>, unsigned int> ({module1, module2, module3}, occurence_unique));

  }
  _module_map_triplet = sorted_module_map_triplet;
}
*/


//------------------------------------------------
template <class T>
void module_map_triplet<T>::add_numerical_error ()
//------------------------------------------------
{
  for (std::pair <const std::vector<uint64_t>, module_doublet<T> >& MMentry : _module_map_doublet)
    MMentry.second.add_numerical_error ();

  for (std::pair <const std::vector<uint64_t>, module_triplet<T> >& MMentry : _module_map_triplet)
    MMentry.second.add_numerical_error ();
}


//-------------------------------------------------------------------------------
template <class T>
void module_map_triplet<T>::save_txt (std::string MM_output, const int precision)
//-------------------------------------------------------------------------------
{
  std::ofstream file ((MM_output+".triplets.txt").c_str(), std::ios::out);
  assert (!file.fail());
  for (std::pair <const std::vector<uint64_t>, module_triplet<T> >& MMentry : _module_map_triplet) {
    file << MMentry.first[0] << " " << MMentry.first[1] << " " << MMentry.first[2] << " ";
    file << MMentry.second.occurence() << " ";
    file << std::setprecision(precision) << MMentry.second.modules12().z0_max() << " " << std::setprecision(precision) << MMentry.second.modules12().z0_min() << " ";
    file << std::setprecision(precision) << MMentry.second.modules12().dphi_max() << " " << std::setprecision(precision) << MMentry.second.modules12().dphi_min() << " ";
    file << std::setprecision(precision) << MMentry.second.modules12().phi_slope_max() << " " << std::setprecision(precision) << MMentry.second.modules12().phi_slope_min() << " ";
    file << std::setprecision(precision) << MMentry.second.modules12().deta_max() << " " << std::setprecision(precision) << MMentry.second.modules12().deta_min() << " ";
    file << std::setprecision(precision) << MMentry.second.modules23().z0_max() << " " << std::setprecision(precision) << MMentry.second.modules23().z0_min() << " ";
    file << std::setprecision(precision) << MMentry.second.modules23().dphi_max() << " " << std::setprecision(precision) << MMentry.second.modules23().dphi_min() << " ";
    file << std::setprecision(precision) << MMentry.second.modules23().phi_slope_max() << " " << std::setprecision(precision) << MMentry.second.modules23().phi_slope_min() << " ";
    file << std::setprecision(precision) << MMentry.second.modules23().deta_max() << " " << std::setprecision(precision) << MMentry.second.modules23().deta_min() << " ";
    file << std::setprecision(precision) << MMentry.second.diff_dzdr_max() << " " << std::setprecision(precision) << MMentry.second.diff_dzdr_min () << " ";
    file << std::setprecision(precision) << MMentry.second.diff_dydx_max() << " " << std::setprecision(precision) << MMentry.second.diff_dydx_min ();
    file << std::endl;
  }
  file.close();

  file = std::ofstream ((MM_output+".doublets.txt").c_str(), std::ios::out);
  assert (!file.fail());
  for (std::pair <const std::vector<uint64_t>, module_doublet<T> >& MMentry : _module_map_doublet) {
    file << std::setprecision(precision) << MMentry.first[0] << " " << std::setprecision(precision) << MMentry.first[1] << " ";
    file << std::setprecision(precision) << MMentry.second.z0_max() << " " << std::setprecision(precision) << MMentry.second.z0_min() << " ";
    file << std::setprecision(precision) << MMentry.second.dphi_max() << " " << std::setprecision(precision) << MMentry.second.dphi_min() << " ";
    file << std::setprecision(precision) << MMentry.second.phi_slope_max() << " " << std::setprecision(precision) << MMentry.second.phi_slope_min() << " ";
    file << std::setprecision(precision) << MMentry.second.deta_max() << " " << std::setprecision(precision) << MMentry.second.deta_min() << " ";
    file << std::endl;
  }
  file.close();
}


//------------------------------------------------------------
template <class T>
void module_map_triplet<T>::save_TTree (std::string MM_output)
//------------------------------------------------------------
{
  clock_t start, end;
  start = clock();

  uint64_t module1, module2, module3;
  module1 = module2 = module3 = 0;
  module_triplet<T> ModuleTriplet;

  if (_drsel == POS_dr) MM_output += ".pos_dr";
  if (_drsel == STRICTPOS_dr) MM_output += ".strictpos_dr";

  std::cout << "Save the module map as " << MM_output << std::endl;

  //  string path_RootFile = joinPaths(outputDir, MMname);
  TFile* RootFile = TFile::Open ((MM_output + ".triplets.root").c_str(), "RECREATE");
  TTree* TreeModuleTriplets = new TTree ("TreeModuleTriplet", "Tree containing the module triplet' features");

  TreeModuleTriplets -> Branch ("MMG_min_version", &_vs);

  //Set up the branches
  TreeModuleTriplets -> Branch ("Module1", &module1);
  TreeModuleTriplets -> Branch ("Module2", &module2);
  TreeModuleTriplets -> Branch ("Module3", &module3);
  ModuleTriplet.branch (TreeModuleTriplets);

  //Save module per module connection in the MM (because of the 1GB limit per buffer in ROOT)
  int size = _module_map_triplet.size();
  for (std::pair <const std::vector<uint64_t>, module_triplet<T> >  &MMentry : _module_map_triplet)
    { module1 = MMentry.first[0]; //mmentry.first[0];
      module2 = MMentry.first[1];
      module3 = MMentry.first[2];

      std::vector<uint64_t> triplet = {module1, module2, module3};
      ModuleTriplet = MMentry.second;

      TreeModuleTriplets -> Fill();
    }

  RootFile -> cd ();
  TreeModuleTriplets -> Write ();
  RootFile -> Close (); // also deletes TTree if not already deleted

  // save Module Map Doublets
  module_doublet<T> ModuleDoublet;
  RootFile = TFile::Open ((MM_output + ".doublets.root").c_str(), "RECREATE");
  TTree* TreeModuleDoublets = new TTree ("TreeModuleDoublet", "Tree containing the module doublet' features");

  TreeModuleDoublets -> Branch ("MMG_min_version", &_vs);

  // Set up the branches
  TreeModuleDoublets -> Branch ("Module1", &module1);//, "Module1/l");
  TreeModuleDoublets -> Branch ("Module2", &module2);//, "Module2/l");
  ModuleDoublet.branch (TreeModuleDoublets, "12");

  size = _module_map_doublet.size();
  for (std::pair <const std::vector<uint64_t>, module_doublet<T> >  &MMentry : _module_map_doublet)
    { module1 = MMentry.first[0]; //mmentry.first[0];
      module2 = MMentry.first[1];

      std::vector<uint64_t> doublet = {module1, module2};
      ModuleDoublet = MMentry.second;

      TreeModuleDoublets -> Fill();
    }

  RootFile -> cd ();
  TreeModuleDoublets -> Write ();
  RootFile -> Close (); // also deletes TTree if not already deleted

  end = clock ();
  std::cout << "Module Map saved to disk" << std::endl;
  std::cout << "cpu time : " << (double)(end-start)/CLOCKS_PER_SEC << std::endl;
}


//-----------------------------------------------------------
template <class T>
void module_map_triplet<T>::read_TTree (std::string MM_input)
//-----------------------------------------------------------
{
  clock_t start, end;
  start = clock();

  if(_drsel==POS_dr) MM_input += ".pos_dr";
  if(_drsel==STRICTPOS_dr) MM_input += ".strictpos_dr";

  std::cout << "Reading the Module Map: " << MM_input << std::endl;

  uint64_t module1, module2, module3;
  module_triplet<T> ModuleTriplet;

  TFile* RootFile = TFile::Open ((MM_input + ".triplets.root").c_str(), "READ");
  if (!RootFile) throw std::invalid_argument ("Cannot open file " + MM_input + ".triplet.root");

  TTree* TreeModuleTriplets = (TTree*) RootFile -> Get ("TreeModuleTriplet");

  float version;
  TreeModuleTriplets -> SetBranchAddress ("MMG_min_version", &version);
  TreeModuleTriplets -> GetEntry (0);
  if (version < _vs){
    std::cout << red << "ERROR: Module Map " << MM_input << ".triplet.root is not compatible with this version of MMG \n";
    std::cout << "use minimum version " << _vs << reset;
    exit(1);
  }

  //Set up the branches
  TreeModuleTriplets -> SetBranchAddress ("Module1", &module1);
  TreeModuleTriplets -> SetBranchAddress ("Module2", &module2);
  TreeModuleTriplets -> SetBranchAddress ("Module3", &module3);
  ModuleTriplet.set_branch_address (TreeModuleTriplets);

  int nb_entries = TreeModuleTriplets -> GetEntries();
//  std::cout << "module map triplet: nb entries = " << nb_entries << std::endl;

  for (int connection=0; connection < nb_entries; connection++)
    { TreeModuleTriplets -> GetEntry (connection);
      std::vector<uint64_t> triplet = {module1, module2, module3};
      _module_map_triplet.insert (std::pair<std::vector<uint64_t>, module_triplet<T>> (triplet, ModuleTriplet));
    }

  RootFile -> Close ("R");

  // reading the Module Map Doublet
  module_doublet<T> ModuleDoublet;
  RootFile = TFile::Open ((MM_input + ".doublets.root").c_str(), "READ");
  if (!RootFile) throw std::invalid_argument ("Cannot open file " + MM_input + "doublet.root");

  TTree* TreeModuleDoublets = (TTree*) RootFile -> Get ("TreeModuleDoublet");

  TreeModuleDoublets -> SetBranchAddress ("MMG_min_version", &version);
  TreeModuleDoublets -> GetEntry (0);
  if (version < _vs){
    std::cout << red << "ERROR: Module Map " << MM_input << ".doublet.root is not compatible with this version of MMG \n";
    std::cout << "use minimum version " << _vs << reset;
    exit(1);
  }

  //Set up the branches
  TreeModuleDoublets -> SetBranchAddress ("Module1", &module1);
  TreeModuleDoublets -> SetBranchAddress ("Module2", &module2);
  ModuleDoublet.set_branch_address (TreeModuleDoublets, "12");

  nb_entries = TreeModuleDoublets -> GetEntries();
  std::cout << "module map doublet: nb entries = " << nb_entries << std::endl;

  for (int connection=0; connection < nb_entries; connection++)
    { TreeModuleDoublets -> GetEntry (connection);
      std::vector<uint64_t> doublet = {module1, module2};
      _module_map_doublet.insert (std::pair<std::vector<uint64_t>, module_doublet<T>> (doublet, ModuleDoublet));
      _module_map_pairs.insert (std::pair<std::vector<uint64_t>, int> (doublet, connection));
    }

  RootFile -> Close ("R");

  // map modules
  int loop = 0;
  for (std::pair<std::vector<uint64_t>, module_doublet<T>> ModuleDoublet : _module_map_doublet) {
    if (_module_map.find(ModuleDoublet.first[0]) == _module_map.end()) _module_map.insert (std::pair<uint64_t, int> (ModuleDoublet.first[0],0));
    if (_module_map.find(ModuleDoublet.first[1]) == _module_map.end()) _module_map.insert (std::pair<uint64_t, int> (ModuleDoublet.first[1],0));
    loop++;
  }

  loop=0;
  for (std::multimap<uint64_t,int>::iterator it = _module_map.begin(); it != _module_map.end(); ++it, loop++)
    it->second = loop;

  std::cout << red << "# modules in MM = " << _module_map.size() << reset;

  end = clock();
  std::cout << "Module Map loaded in memory" << std::endl;
  std::cout << "cpu time : " << (double)(end-start)/CLOCKS_PER_SEC << std::endl;
}

//template class module_map_triplet<data_type2>;
//template class module_map_triplet<double>;