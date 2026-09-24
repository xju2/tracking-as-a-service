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
#error Must use C++ for the type vertex
#endif

#include <string>

template <class T> class graph;

//=============================
template <class T> class vertex
//=============================
{
  friend void graph<T>::save (const std::string&, const std::string&, const bool&);

  protected:
    T _r;
    T _phi;
    T _z;
    T _eta; 
    T _r_cluster1;
    T _phi_cluster1;
    T _z_cluster1;
    T _eta_cluster1; 
    T _r_cluster2;
    T _phi_cluster2;
    T _z_cluster2;
    T _eta_cluster2; 
    int _hit_id;

  public:
    vertex () {_r = _phi = _z = _eta = _hit_id = _r_cluster1 = _phi_cluster1 = _z_cluster1 = _eta_cluster1 = _r_cluster2 = _phi_cluster2 = _z_cluster2 = _eta_cluster2 = _hit_id = 0;}
    vertex (const T&, const T&, const T&, const T&);
    vertex (const T&, const T&, const T&, const T&, const T&, const T&, const T&, const T&, const T&, const T&, const T&, const T&, const T&);

    T& r () {return _r;}
    T& phi () {return _phi;}
    T& z () {return _z;}
    T& eta () {return _eta;}
    T& r_cluster1 () {return _r_cluster1;}
    T& phi_cluster1 () {return _phi_cluster1;}
    T& z_cluster1 () {return _z_cluster1;}
    T& eta_cluster1 () {return _eta_cluster1;}
    T& r_cluster2 () {return _r_cluster2;}
    T& phi_cluster2 () {return _phi_cluster2;}
    T& z_cluster2 () {return _z_cluster2;}
    T& eta_cluster2 () {return _eta_cluster2;}
    int& hit_id () {return _hit_id;}
};
