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
#error Must use C++ for the type vectex true
#endif

#include <MMG/vertex>

template <class T> class graph_true;

//=====================================================
template <class T> class vertex_true : public vertex<T>
//=====================================================
{
  friend void graph_true<T>::save (const std::string&, const std::string&);

  protected:
    T _pt_particle;
    T _eta;

  public:
    vertex_true () : vertex<T>() {_pt_particle = _eta = 0;}
    T& pT () {return _pt_particle;}
    T& eta () {return _eta;}
};
