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

#include "MMG/edge_true.hpp"

//-----------------------------------------------------------------------------------------------------------------
template <class T>
edge_true<T>::edge_true (const int& is_segment_true, const T& pt_particle, const int& mask_edge, const int& region)
//-----------------------------------------------------------------------------------------------------------------
{
  _is_segment_true = is_segment_true;
  _pt_particle = pt_particle;
  _mask_edge = mask_edge;
  _region = region;
}


//------------------------------------------------------------
template <class T>
edge_true<T>& edge_true<T>::operator = (const edge_true<T>& e)
//------------------------------------------------------------
{
  _is_segment_true = e._is_segment_true;
  _pt_particle = e._pt_particle;
  _mask_edge = e._mask_edge;
  _region = e._region;

  return (*this);
}
