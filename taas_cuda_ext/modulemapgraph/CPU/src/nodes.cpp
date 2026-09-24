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

#include "MMG/nodes.hpp"

//---------------------
template <class T>
nodes<T>::nodes (int n)
//---------------------
{
  _nb_nodes = n;
  _hit_id = std::vector<uint64_t> (_nb_nodes);
  _R = std::vector<T> (_nb_nodes);
  _z = std::vector<T> (_nb_nodes);
  _eta = std::vector<T> (_nb_nodes);
  _phi = std::vector<T> (_nb_nodes);
  _ACORN_positions = std::vector<int> (_nb_nodes);;
}
