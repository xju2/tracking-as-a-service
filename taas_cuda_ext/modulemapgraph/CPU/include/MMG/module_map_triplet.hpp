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

#ifndef __cplusplus
#error Must use C++ for the type module map triplet
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
#include <MMG/edge>
#include <iterator>
#include <MMG/TTree_dump>

enum DRsel {ALL_dr=-1, POS_dr=0, STRICTPOS_dr=1};

//==========================================================
template <class T> class module_map_triplet : public TObject
//==========================================================
{
  private:
    // minimum vs of MMG for this Module Map
    float _vs=1.3;

    bool _strip_hit_pair;
    bool _closest_to_vertex;
    // event nb / modules triplet nb (ex: 1,2,3) / module_triplet
    // std::multimap <std::pair<int, std::vector<uint64_t>, module_triplet<T,U>> _module_map_triplet;
    TTree_particles<T> _ModMap_particles;
    std::vector<geometric_cuts<T>> _cuts12, _cuts23;

    std::multimap <std::vector<uint64_t>, module_triplet<T>> _module_map_triplet;
    std::multimap <std::vector<uint64_t>, module_doublet<T>> _module_map_doublet;
    std::multimap <std::vector<uint64_t>, int> _module_map_pairs;
    std::multimap <uint64_t, int> _module_map;

//    std::multimap <std::vector<uint64_t>, unsigned int> _module_map_doublet;
//    module_doublet<T> *_module_doublets;
    ///    std::multimap <std::vector<uint64_t>, std::vector<int>> _moduleID_particles;
    std::multimap <std::vector<uint64_t>, std::vector<int>> _modules_particles; // module1, module2, module3 <-> particles position in TTp
    std::multimap <std::vector<uint64_t>, std::vector<T>> _modules_particlesKey; // probasbly useless ?
    std::multimap <std::vector<T>, int> _particlesSPECS_TTp_link;

    DRsel _drsel;

    long unsigned int _min_N_hits;
    T _min_pt;
    T _max_pt;

    std::set<ULong64_t> _event_veto;
    std::set<uint64_t> _dead_module_ids;

    void init(boost::program_options::variables_map& po_vm);
    void make_module_map_singlet();
    void add_event_triplets (const std::string&, const std::string&, const std::string&, const int&, const T&, const T&);
    void add_event_triplets (TTree_particles<T>&, TTree_hits<T>&, const int&, const T&, const T&);

  public:
    module_map_triplet () {_drsel=ALL_dr;} //p = new _Rb_tree_iterator <pair <const vector<long long unsigned int>, module_triplet<float> > >;}
//    template <class Tf> module_map_triplet (module_map_triplet<Tf>&);
    module_map_triplet (boost::program_options::variables_map&, const std::string&, const std::string&);
    module_map_triplet (boost::program_options::variables_map&); // constructor for Root Dump as input 
    ~module_map_triplet () {} //delete p;}

    bool operator ! () {return !_module_map_triplet.size();}
//    bool operator += (const module_triplet<T>);
//    template <class Tf> module_map_triplet<T>& operator = (const module_map_triplet<Tf>&);
    template <class Tf> friend bool operator == (const module_map_triplet<Tf>&, const module_map_triplet<Tf>&);
    template <class Tf> friend bool operator != (const module_map_triplet<Tf>&, const module_map_triplet<Tf>&);
    inline std::_Rb_tree_iterator <std::pair <const std::vector<uint64_t>, module_triplet<T>>> begin() {return _module_map_triplet.begin();}
    inline std::_Rb_tree_iterator <std::pair <const std::vector<uint64_t>, module_triplet<T>>> end() {return _module_map_triplet.end();}
    inline int size () const {return _module_map_triplet.size();}

    inline const std::multimap <std::vector <uint64_t>, module_triplet<T>>* operator () () const {return &_module_map_triplet;}
    inline const std::multimap <std::vector<uint64_t> , module_triplet<T> >& map_triplet () const {return _module_map_triplet;}
    inline std::multimap <std::vector<uint64_t> , module_triplet<T> >& map_triplet () {return _module_map_triplet;}
    inline const std::multimap <std::vector<uint64_t> , module_doublet<T> >& map_doublet () const {return _module_map_doublet;}
    inline const std::multimap <std::vector<uint64_t>, int>& map_pairs () const {return _module_map_pairs;}
    inline int map_pair (const std::vector<uint64_t>& modules) {return _module_map_pairs.find (modules)->second;}
    inline std::multimap <uint64_t, int> module_map () {return _module_map;}
    inline std::multimap <std::vector<uint64_t> , module_doublet<T> >& map_doublet () {return _module_map_doublet;}
    //inline std::vector<T> particles_key (const std::vector<uint64_t>& modules) {return _modules_particlesKey.find(modules);} // probably useless
    //std::vector<T> particles_key (const std::vector<uint64_t>& modules) {return _modules_particlesKey.find(modules);} // probably useless
    inline std::vector<int>& particles_position (const std::vector<uint64_t>& modules) {return _modules_particles.find(modules)->second;}
//    bool add_triplet_occurence (const std::vector<uint64_t>&, const TTree_hits<T>&, const TTree_particles<T>&, const geometric_cuts<T>&, const geometric_cuts<T>&, const T&, const T&);
    bool add_triplet_occurence (const std::vector<uint64_t>&, const TTree_hits<T>&, const TTree_particles<T>&, const geometric_cuts<T>&, const geometric_cuts<T>&, const T&, const T&, int, int, int);
    void merge (const module_map_triplet<T>&);
    void sort ();
    void remove_unique_occurence ();
    void remove_cycles ();
    void add_numerical_error();
    std::vector<geometric_cuts<T>>& cuts12 () {return _cuts12;}

    void save_txt (std::string, const int=16);
    void save_TTree (std::string);
    void read_TTree (std::string);
};
