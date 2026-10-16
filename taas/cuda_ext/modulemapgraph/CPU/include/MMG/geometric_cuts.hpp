/***********************************************************************
 * Module Map Graph library
 * Track reconstruction for HEP
 * copyright © 2023 ROUGIER Charline
 * copyright © 2023,2024,2025 COLLARD Christophe
 * Laboratoire des 2 Infinis - Toulouse - France (L2IT)
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 *
 * contributors:
 * 2024 Alexis Vallier
 **********************************************************************/

#ifndef __cplusplus
#error Must use C++ for the type geometric cuts
#endif

#include <iostream>
#include <cmath>
#include <numbers>
#include <MMG/TTree_hits>

//=====================================
template <class T> class geometric_cuts
//=====================================
{
  private:
    T _z0;
    T _dphi;
    T _phi_slope;
    T _r_phi_slope;
    T _deta;

    T _dr; // not a feature used for cuts, simlpy to check z0 and phi_slope validity

  public:
    geometric_cuts () {}
    geometric_cuts (T, T, T, T);
    template <class V> geometric_cuts (const TTree_hits<V>&, const int&, const int&);
    ~geometric_cuts () {}

    void branch (TTree*, const std::string&, const std::string&);
    void set_branch_address (TTree*, const std::string&, const std::string&);

    template <class Tf> friend bool operator == (const geometric_cuts<Tf>&, const geometric_cuts<Tf>&);
    template <class Tf> friend bool operator != (const geometric_cuts<Tf>&, const geometric_cuts<Tf>&);
    template <class Tf> friend std::ostream& operator << (std::ostream&, const geometric_cuts<Tf>&);

    inline const T& z0 () const {return _z0;}
    inline const T& d_phi () const {return _dphi;}
    inline const T& phi_slope () const {return _phi_slope;}
    inline const T& r_phi_slope () const {return _r_phi_slope;}
    inline const T& d_eta () const {return _deta;}
    inline const T& dr () const {return _dr;}

    inline T& z0 () {return _z0;}
    inline T& d_phi () {return _dphi;}
    inline T& phi_slope () {return _phi_slope;}
    inline T& d_eta () {return _deta;}
    inline T& dr () {return _dr;}

    bool check_cut_values () const;
    void add_numerical_error (T);
    void min (const geometric_cuts<T>&);
    void max (const geometric_cuts<T>&);
    void sum (const geometric_cuts<T>&);
    void sumSq (const geometric_cuts<T>&);
    inline bool min_cuts (const geometric_cuts<T>& cuts) {return _z0 > cuts._z0 || _dphi > cuts._dphi || _phi_slope > cuts._phi_slope || _deta > cuts._deta;}
    inline bool max_cuts (const geometric_cuts<T>& cuts) {return _z0 < cuts._z0 || _dphi < cuts._dphi || _phi_slope < cuts._phi_slope || _deta < cuts._deta;}
};

template <class Tf> Tf Diff_dydx (const TTree_hits<Tf>&, const int&, const int&, const int&);
template <class Tf> Tf Diff_dzdr (const TTree_hits<Tf>&, const int&, const int&, const int&);
