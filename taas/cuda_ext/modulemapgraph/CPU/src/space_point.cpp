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

#include "MMG/space_point.hpp"

//---------------------------------------------------------------------------------------------------------------------
template <class T>
space_point<T>::space_point (
                        const std::string& hardware, const int& barrel_endcap, const int& layer_disk,
                        const int& eta_module, const int& phi_module,
                        const std::vector<T>& spoint, const std::vector<T>& cluster1, const std::vector<T>& cluster2){
//---------------------------------------------------------------------------------------------------------------------
  _hardware = hardware;
  _barrel_endcap = barrel_endcap;
  _eta_module = eta_module;
  _phi_module = phi_module;
  _layer_disk = layer_disk;
  _spoint = spoint;
  _cluster1 = cluster1;
  _cluster2 = cluster2;
}
