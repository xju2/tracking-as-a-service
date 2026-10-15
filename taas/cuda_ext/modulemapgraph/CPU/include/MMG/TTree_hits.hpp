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
#error Must use C++ for the type TTree hits
#endif

#include <MMG/colors>
#include <MMG/parameters>
#include <iostream>
#include <fstream>
#include <assert.h>
#include <utility>
#include <set>
#include <vector>
#include <TFile.h>
#include <TTree.h>
#include <boost/type_index.hpp>
#include <MMG/hit>
#include <MMG/TTree_dump>
#include <MMG/space_point>

/*
//---------------------------------------------------------------------
template <class _A, class _B, class _Compare=std::less<_A>>
  class sorted_multimap : public std::set <std::pair<_A, _B>, _Compare>
//---------------------------------------------------------------------
{
  public :
    sorted_multimap () : std::set<std::pair<_A, _B >, _Compare> () {};
    ~sorted_multimap () {};
};

template <typename InPair>
struct MMapComp{
        bool operator() (InPair a , InPair b){
                if( a.first == b.first ) return a.second > b.second;
                else
                        return a.first < b.first;
        }
};
*/

//=================================
template <class T> class TTree_hits
//=================================
{
//  friend hits;
  protected:
    int _size;
    mutable int _position;
    bool _true_graph;
    bool _extra_features;

    // particles properties for std root files compatibility only
    std::vector<uint64_t> _hit_id;
    std::vector<T> _x, _y, _z;
    std::vector<uint64_t> _particle_ID;
    std::vector<uint64_t> _module_ID;
    std::vector<std::string> _hardware;
    std::vector<int> _barrel_endcap, _layer_disk, _eta_module, _phi_module;;
    std::vector<uint64_t> *_particle_ID1;
    std::vector<uint64_t> *_particle_ID2;
    std::vector<T> *_cluster1_x, *_cluster1_y, *_cluster1_z, *_cluster2_x, *_cluster2_y, *_cluster2_z;
    std::vector<T> *_module_x, *_module_y, *_module_z;

//    std::vector<uint64_t>* athena_moduleId;
//    vector<int>* index;
//    vector<int long>* barcode;
//    vector<int>* subevent;

    std::vector<T> _R, _eta, _phi;
    std::vector<T> *_R_cluster1, *_eta_cluster1, *_phi_cluster1;
    std::vector<T> *_R_cluster2, *_eta_cluster2, *_phi_cluster2;

    std::multimap<uint64_t, int> _moduleID_TTreeHits_map;
    std::multimap<uint64_t, int> _particleID_TTreeHits_map;

    void allocate_memory();

  public:
    TTree_hits (bool=false, bool=false);
    TTree_hits (const TTree_hits<T>&); // copy constructor
//    TTree_hits (hits<T>&);  // cast conversion to hits
    TTree_hits(const TTree_dump&, const std::set<uint64_t>& dead_module_ids); // constructor from one event in dump
    ~TTree_hits ();

//    inline TTree_hits<T>& operator [] (const int& i) {get(i); return *this;}
    TTree_hits<T>& operator = (const TTree_hits<T>&);
    template <class Tf> friend bool operator == (const TTree_hits<Tf>&, const TTree_hits<Tf>&);
    template <class Tf> friend bool operator != (const TTree_hits<Tf>&, const TTree_hits<Tf>&);

    void add_hit (const space_point<T>&);
    void get (int) const;
    hit<T> get_hit ();
    hit<T>* get_hit_addr ();
//    inline void build_R () {std::transform (std::execution::par_unseq, std::begin(_x), std::end(_x), std::begin(_y), std::begin(*_R), [](data_type2 xi, data_type2 yi) {return sqrt (xi * xi + yi * yi);});}
    inline int size () const {return _size;}
    inline int location () const {return _position;}
    inline bool true_graph () const {return _true_graph;}
    inline bool extra_features () const {return _extra_features;}
    inline const uint64_t& hit_id () const {return _hit_id[_position];}
    inline const T& x () const {return _x[_position];}
    inline const T& y () const {return _y[_position];}
    inline const T& z () const {return _z[_position];}
    inline const uint64_t& particle_ID () const {return _particle_ID[_position];}
    inline const uint64_t& module_ID () {return _module_ID[_position];}
    inline const std::string& hardware () const {return _hardware[_position];}
    inline const int& barrel_endcap () const {return _barrel_endcap[_position];}
    inline const uint64_t& particle_ID1 () const {return (*_particle_ID1)[_position];}
    inline const uint64_t& particle_ID2 () const {return (*_particle_ID2)[_position];}

    inline const int& layer_disk () const {return _layer_disk[_position];}
    inline const int& eta_module () const {return _eta_module[_position];}
    inline const int& phi_module () const {return _phi_module[_position];}
    inline const T& x_cluster1 () const {return (*_cluster1_x)[_position];}
    inline const T& y_cluster1 () const {return (*_cluster1_y)[_position];}
    inline const T& z_cluster1 () const {return (*_cluster1_z)[_position];}
    inline const T& x_cluster2 () const {return (*_cluster2_x)[_position];}
    inline const T& y_cluster2 () const {return (*_cluster2_y)[_position];}
    inline const T& z_cluster2 () const {return (*_cluster2_z)[_position];}

    inline T R () const {return _R[_position];}
    inline T Eta () const {return _eta[_position];}
    inline T Phi () const {return _phi[_position];}

    inline T R_cluster1 () const {return (*_R_cluster1)[_position];}
    inline T Eta_cluster1 () const {return (*_eta_cluster1)[_position];}
    inline T Phi_cluster1 () const {return (*_phi_cluster1)[_position];}
    inline T R_cluster2 () const {return (*_R_cluster2)[_position];}
    inline T Eta_cluster2 () const {return (*_eta_cluster2)[_position];}
    inline T Phi_cluster2 () const {return (*_phi_cluster2)[_position];}

    const uint64_t& hit_id (int) const;
    const T& x (int) const;
    const T& y (int) const;
    const T& z (int) const;
    const uint64_t& particle_ID (int) const;
    const uint64_t& module_ID (int) const;

    T R (int) const;
    T Eta (int) const;
    T Phi (int) const;

    // data for cuda code
    inline const T* event_x () {return _x.data();}
    inline const T* event_y () {return _y.data();}
    inline const T* event_z () {return _z.data();}
    inline const uint64_t* event_hit_ID () {return _hit_id.data();}
    inline const uint64_t* event_module_ID () {return _module_ID.data();}
    inline const int* event_barrel_endcap () const {return _barrel_endcap.data();}
    inline const std::string* event_hardware () const {return _hardware.data();}
    inline const int* event_layer_disk () const {return _layer_disk.data();}
    inline const int* event_eta_module () const {return _eta_module.data();}
    inline const int* event_phi_module () const {return _phi_module.data();}

    // end data for cuda code

    const std::string& hardware (int) const;
    const int& barrel_endcap (int) const;

    // part for Heberth's code
    const int& layer_disk (int) const;
    const int& eta_module (int) const;
    const int& phi_module (int) const;
    const T& x_cluster1 (int) const;
    const T& y_cluster1 (int) const;
    const T& z_cluster1 (int) const;
    const T& x_cluster2 (int) const;
    const T& y_cluster2 (int) const;
    const T& z_cluster2 (int) const;

    T R_cluster1 (int) const;
    T Eta_cluster1 (int) const;
    T Phi_cluster1 (int) const;
    T R_cluster2 (int) const;
    T Eta_cluster2 (int) const;
    T Phi_cluster2 (int) const;
    // end of part for Heberth's code

    template <class Tf> friend int define_region (const TTree_hits<Tf>&, const int&, const int&);

    inline std::pair<std::_Rb_tree_iterator<std::pair<const uint64_t, int>>, std::_Rb_tree_iterator<std::pair<const uint64_t, int>>> moduleID_equal_range (const uint64_t& ID) {return _moduleID_TTreeHits_map.equal_range(ID);}
    inline std::pair<std::_Rb_tree_iterator<std::pair<const uint64_t, int>>, std::_Rb_tree_iterator<std::pair<const uint64_t, int>>> particleID_equal_range (const uint64_t& ID) {return _particleID_TTreeHits_map.equal_range(ID);}
    inline std::multimap<uint64_t, int>& hits_map () {return _moduleID_TTreeHits_map;}

    void branch (TTree*);
    void set_branch_address (TTree*);
    void push_back (const hit<T>&);
    void create_hits_multimap(hit<T>&);
    void save (const std::string&);
    void read (const std::string&);
};
