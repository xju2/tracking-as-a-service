/***********************************************************************
 * Module Map Graph library
 * Track reconstruction for HEP
 * copyright © 2023 TORRES Heberth
 * copyright © 2026 COLLARD Christophe
 * Laboratoire des 2 Infinis - Toulouse - France (L2IT)
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 **********************************************************************/

#ifndef __cplusplus
#error Must use C++ for the type strip hits pair
#endif

#include <MMG/parameters>
#include <iostream>
#include <iomanip>
#include <functional>
#include <tuple>
#include <TROOT.h>
#include <TMath.h>
#include <MMG/TTree_hits>
#include <MMG/strip_module_DB>

//==========================
template <class T, class DT>
class strip_hit_pair
//==========================
{
  private:
//    const TTree_hits<T>* hits;
    const TTree_hits<DT>* _TThits;
    int _hit_in, _hit_out;
    T _Perp_in, _Perp_out;
    T _eta_in, _eta_out;
    T _theta, _alpha_in, _alpha_out;
    std::vector<T> _new_spoint_in;
    std::vector<T> _new_spoint_out;
    bool _pass_strip_range;

    T perp (const int&);
    T perp (const std::vector<T>&);
    T norm2_cluster1 (const int&);
    T norm2 (const std::vector<T>&);
//    T Eta (const int&);
    T Phi (const int&);
    T Phi (const std::vector<T>&);
    T delta_phi (const T&, const T&);
    T delta_phi (const std::vector<T>&, const std::vector<T>&);
    std::vector<T> rotate_z_axis (const std::vector<T>&, const T&);
    std::vector<T> rotate_z_axis (const T&, const T&, const T&, const T&);
    T barrel_strip_module_tilt (const char&);
    T barrel_strip_module_sin_tilt (const char&);
    T get_edge_theta (const std::vector<T>&, const T&);
    std::tuple<T, T> get_edge_alphas (const T&);
    std::tuple<T, T> get_edge_alphas (const std::vector<T>&, const std::vector<T>&, const T&);
    std::vector<T> hit_local_position (const std::vector<T>&, const std::vector<T>&, const T&, const T&);
    std::vector<T> hit_global_position (const std::vector<T>&, const int&, const T&, const T&, const T&);

  public:
    strip_hit_pair ();
    strip_hit_pair (const TTree_hits<DT>&, const int&, const int&, strip_module_DB<T>&);
    void reconstruction ();

    inline T R_in () {return _Perp_in;}
    T Eta (const int&);
    T eta (const std::vector<T>&);
    T d_eta ();
    T d_phi ();
    inline T d_r () {return _Perp_out - _Perp_in;}
    inline T d_z () {return (_new_spoint_out.at(2) - _new_spoint_in.at(2));}
    inline T sum_Perp () {return _Perp_in + _Perp_out;}
    T theta () {return _theta;}
    T alpha_in () {return _alpha_in;}
    T alpha_out () {return _alpha_out;}
    inline bool pass_strip_range () {return _pass_strip_range;}
    const std::vector<T>& new_hit_in () {return _new_spoint_in;}
    const std::vector<T>& new_hit_out () {return _new_spoint_out;}
};
