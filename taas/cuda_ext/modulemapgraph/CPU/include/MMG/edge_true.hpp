/***********************************************************************
 * Module Map Graph library
 * Track reconstruction for HEP
 * copyright © 2022 ROUGIER Charline
 * copyright © 2022 COLLARD Christophe
 * Laboratoire des 2 Infinis - Toulouse - France (L2IT)
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 **********************************************************************/

#ifndef __cplusplus
#error Must use C++ for the type edge true
#endif

#include <assert.h>
#include <iostream>
#include <iomanip>

template <class T> class graph_true;

//================================
template <class T> class edge_true
//================================
{
  friend void graph_true<T>::save (const std::string&, const std::string&);

  private:
    int _is_segment_true;
    T _pt_particle;
    int _mask_edge;
    int _region;

  public:
    edge_true () {_is_segment_true = _pt_particle = 0; _mask_edge = 1;}
    edge_true (const int&, const T&, const int&, const int&);
    ~edge_true () {}

    edge_true<T>& operator = (const edge_true<T>&);

    inline int& is_segment_true () {return _is_segment_true;}
    inline T& pT () {return _pt_particle;}
    inline int& mask_edge () {return _mask_edge;}
    inline int& region () {return _region;}
};
