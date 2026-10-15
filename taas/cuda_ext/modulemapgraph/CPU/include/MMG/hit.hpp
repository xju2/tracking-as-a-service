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
#error Must use C++ for the type hit
#endif

#include <assert.h>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iterator>
#include <stdlib.h>
#include <vector>
#include <TTree.h>
#include <map>
#include <string>

//==========================
template <class T> class hit
//==========================
{
  private:
    bool _extra_features;
    uint64_t _hit_id;
    T _x, _y, _z;

    // needed by graph_true
    // position of the hit relative to the mother particle
//    T _vx, _vy, _vz;
    uint64_t _particle_id, _ID;
    std::string _hardware;
    int _barrel_endcap;
    uint64_t _particle_ID1;
    uint64_t _particle_ID2;

    int _layer_disk;
    int _eta_module, _phi_module;
    T _cluster1_x, _cluster1_y, _cluster1_z, _cluster2_x, _cluster2_y, _cluster2_z;

  public:
    hit (bool extra_feat=true) {_extra_features = extra_feat;}
    explicit hit(const std::map<std::string,int>&, std::string, bool=false);
//    hit(const uint64_t&, const T&, const T&, const T&, const T&, const T&, const T&, const uint64_t&, const uint64_t&, const T&, const int&);
    hit(const uint64_t&, const T&, const T&, const T&, const uint64_t&, const uint64_t&, const std::string&, const int&, const uint64_t&, const uint64_t&);
    hit(const uint64_t&, const T&, const T&, const T&, const uint64_t&, const uint64_t&, const std::string&, const int&, const uint64_t&, const uint64_t&, const int&, const int&, const int&, const T&, const T&, const T&, const T&, const T&, const T&);
    ~hit() {}

    template <class Tf> friend bool operator == (const hit<Tf>&, const hit<Tf>&);
    template <class Tf> friend bool operator != (const hit<Tf>&, const hit<Tf>&);

    inline bool extra_features () const {return _extra_features;}
    inline const uint64_t& hit_id () const {return _hit_id;}
    inline const T& x () const {return _x;}
    inline const T& y () const {return _y;}
    inline const T& z () const {return _z;}
//    inline const T& vx () const {return _vx;}
//    inline const T& vy () const {return _vy;}
//    inline const T& vz () const {return _vz;}
    inline const uint64_t& particle_id () const {return _particle_id;}
    inline uint64_t module_ID () const {return _ID;}
    inline const std::string& hardware () const {return _hardware;}
    inline const int& barrel_endcap () const {return _barrel_endcap;}
    inline const uint64_t& particle_ID1 () const {return _particle_ID1;}
    inline const uint64_t& particle_ID2 () const {return _particle_ID2;}

    inline const int& layer_disk () const {return _layer_disk;}
    inline const int& eta_module () const {return _eta_module;}
    inline const int& phi_module () const {return _phi_module;}
    inline const T& cluster1_x () const {return _cluster1_x;}
    inline const T& cluster1_y () const {return _cluster1_y;}
    inline const T& cluster1_z () const {return _cluster1_z;}
    inline const T& cluster2_x () const {return _cluster2_x;}
    inline const T& cluster2_y () const {return _cluster2_y;}
    inline const T& cluster2_z () const {return _cluster2_z;}

//    inline const T& eta_particle () const {return _eta_particle;}
//    inline const int particle_pdgId () const {return _particle_pdgId;}
    inline T R () const {return sqrt(_x*_x + _y*_y);}
    inline T norm () const {return sqrt(_x*_x + _y*_y + _z*_z);}

    void save_csv(std::ofstream&);
    bool read_csv(const std::map<std::string,int>&, std::ifstream&);
};
