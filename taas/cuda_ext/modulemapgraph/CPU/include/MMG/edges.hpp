/***********************************************************************
 * Module Map Graph library
 * Track reconstruction for HEP
 * copyright © 2025 COLLARD Christophe
 * Laboratoire des 2 Infinis - Toulouse - France (L2IT)
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 **********************************************************************/

#ifndef __cplusplus
#error Must use C++ for the type edges
#endif

//============================
template <class T> class edges
//============================
{
  private:
    int _nb_edges;
    std::vector<int> _edge_source, _edge_target;
    std::vector<int> _ACORN_edge_source, _ACORN_edge_target;
    bool _ACORN;
    std::vector<T> _dR, _dz, _deta, _dphi, _phi_slope, _r_phi_slope;

  public:
    edges () {_nb_edges=0;}
    edges (int);

    inline int nb_edges () {return _nb_edges;}
    inline std::vector<int>& edge_source () {return _edge_source;}
    inline std::vector<int>& edge_target () {return _edge_target;}
    inline std::vector<T>& dR () {return _dR;}
    inline std::vector<T>& dz () {return _dz;}
    inline std::vector<T>& deta () {return _deta;}
    inline std::vector<T>& dphi () {return _dphi;}
    inline std::vector<T>& phi_slope () {return _phi_slope;}
    inline std::vector<T>& r_phi_slope () {return _r_phi_slope;}
};
