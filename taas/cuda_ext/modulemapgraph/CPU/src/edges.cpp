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

#include "MMG/edges.hpp"

//---------------------
template <class T>
edges<T>::edges (int n)
//---------------------
{
  _nb_edges = n;
  _edge_source = std::vector<int> (_nb_edges);
  _edge_target = std::vector<int> (_nb_edges);
  //std::vector<int> _ACORN_edge_source, _ACORN_edge_target;
  //  bool _ACORN;
  _dR = std::vector<T> (_nb_edges);
  _dz = std::vector<T> (_nb_edges);
  _deta = std::vector<T> (_nb_edges);
  _dphi = std::vector<T> (_nb_edges);
  _phi_slope = std::vector<T> (_nb_edges);
  _r_phi_slope = std::vector<T> (_nb_edges);
}
