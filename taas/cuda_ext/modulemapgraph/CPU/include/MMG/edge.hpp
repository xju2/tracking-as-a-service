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
#error Must use C++ for the type edge
#endif

#include <assert.h>
#include <iostream>
#include <iomanip>
#include <MMG/TTree_hits>
#include <MMG/hit>
#include <MMG/strip_hit_pair>
#include <MMG/strip_module_DB>

template <class T> class graph;

//===========================
template <class T> class edge
//===========================
{
  friend void graph<T>::save (const std::string&, const std::string&, const bool&);

  private:
    bool _extra_features;
    T _dEta;
    T _dPhi;
    T _dr;
    T _dz;
    T _phi_slope;
    T _r_phi_slope;
    // do not store those data here - access via _strip_hit_pair methods ?
    T _new_hit_in_x, _new_hit_in_y, _new_hit_in_z;
    T _new_hit_out_x, _new_hit_out_y, _new_hit_out_z;
    T _new_dEta, _new_dPhi, _new_dr, _new_dz, _new_PhiSlope, _new_rPhiSlope;
    bool _pass_strip_range;
    strip_hit_pair<data_type2, data_type1> _strip_hit_pair;

  public:
    edge () {_dEta = _dPhi = _dr = _dz = _phi_slope = _r_phi_slope;}
    edge (const T&, const T&, const T&, const T&, const T&, const T&);
    ~edge () {}

    edge<T>& operator = (const edge<T>&);
    void barrel_strip_hit (const TTree_hits<T>&, const int&, const int&, strip_module_DB<data_type2>&);

    T& dEta () {return _dEta;}
    T& dPhi () {return _dPhi;}
    T& dr () {return _dr;}
    T& dz () {return _dz;}
    T& phi_slope () {return _phi_slope;}
    T& r_phi_slope () {return _r_phi_slope;}
    const T& new_hit_in_x () {return _new_hit_in_x;}
    const T& new_hit_in_y () {return _new_hit_in_y;}
    const T& new_hit_in_z () {return _new_hit_in_z;}
    const T& new_hit_out_x () {return _new_hit_out_x;}
    const T& new_hit_out_y () {return _new_hit_out_y;}
    const T& new_hit_out_z () {return _new_hit_out_z;}
    const bool& pass_strip_range () {return _pass_strip_range;}
    const T& new_dEta () {return _new_dEta;}
    const T& new_dPhi () {return _new_dPhi;}
    const T& new_dr () {return _new_dr;}
    const T& new_dz () {return _new_dz;}
    const T& new_phi_slope () {return _new_PhiSlope;}
    const T& new_r_phi_slope () {return _new_rPhiSlope;}
};
