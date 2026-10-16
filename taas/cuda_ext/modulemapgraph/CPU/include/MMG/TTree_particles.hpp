/***********************************************************************
 * Module Map Graph library
 * Track reconstruction for HEP
 * copyright © 2022,2023 ROUGIER Charline
 * copyright © 2022,2023 COLLARD Christophe
 * Laboratoire des 2 Infinis - Toulouse - France (L2IT)
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 **********************************************************************/

#ifndef __cplusplus
#error Must use C++ for the type TTree particles
#endif

#include <iostream>
#include <fstream>
#include <assert.h>
#include <vector>
#include <TFile.h>
#include <TTree.h>
#include <boost/type_index.hpp>
#include <MMG/particle>
#include <MMG/TTree_dump>

//======================================
template <class T> class TTree_particles
//======================================
{
  private:
    int _size;
    mutable int _position;

    // particles properties
    std::vector<int> *_event_id;
    std::vector<uint64_t>* _particle_ID;
    std::vector<int>* _subevent; //PU or HS
    std::vector<int>* _barcode;
    std::vector<int>* _pdgID;
    std::vector<T>* _eta;
//    std::vector<T>* phi;
    std::vector<T>* _pT;
    std::vector<T>* _vx;
    std::vector<T>* _vy;
    std::vector<T>* _vz;

    std::multimap<uint64_t, int> _particleID_TTreeParticles_map;

    void allocate_memory();


  public:
    TTree_particles ();
    TTree_particles (const TTree_particles&); // copy constructor
//    TTree_hits (const string&);  // constructor with TTree build
    TTree_particles(const TTree_dump&); // constructor from one event in dump
    ~TTree_particles ();

    virtual TTree_particles<T>& operator = (const TTree_particles<T>&);
    template <class Tf> friend bool operator == (const TTree_particles<Tf>&, const TTree_particles<Tf>&);
    template <class Tf> friend bool operator != (const TTree_particles<Tf>&, const TTree_particles<Tf>&);
    TTree_particles<T>& operator += (const TTree_particles<T>&);

    void get (int) const;
    void getID (const uint64_t&) const;
    particle<T> get_particle ();
    inline int size () const {return _size;}
    inline int location () const {return _position;}
    inline int event_ID () const {return (*_event_id)[_position];}
    inline const uint64_t& particle_ID () const {return (*_particle_ID)[_position];}
    inline const T& vx () const {return (*_vx)[_position];}
    inline const T& vy () const {return (*_vy)[_position];}
    inline const T& vz () const {return (*_vz)[_position];}
    inline const T& pT () const {return (*_pT)[_position];}
    inline const int& pdgID () const {return (*_pdgID)[_position];}
    inline const T& eta () const {return (*_eta)[_position];}
    inline const int& subevent () const {return (*_subevent)[_position];}
    inline const int& barcode () const {return (*_barcode)[_position];}
//    inline const std::vector<T>& key () const {return std::vector<T> (_pT[_position], _pdgID[_position], _eta[_position], _vx[_position], _vy[_position], _vz[_position]);}

    inline std::pair<std::_Rb_tree_iterator<std::pair<const uint64_t, int>>, std::_Rb_tree_iterator<std::pair<const uint64_t, int>>> particleID_equal_range (const uint64_t& ID) {return _particleID_TTreeParticles_map.equal_range(ID);}
    int find_particle (const uint64_t& ID);

    void branch (TTree*);
    void set_branch_address (TTree*);
    void push_back (particle<T>&);
    void create_hits_multimap(particle<T>&);
    void save (const std::string&);
    void read (const std::string&);
};
