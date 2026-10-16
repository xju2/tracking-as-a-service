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
#error Must use C++ for the type hits infos
#endif

#include <iostream>
#include <fstream>
#include <assert.h>
#include <vector>
#include <MMG/TTree_hits>
#include <MMG/TTree_particles>


//=====================================================================
template <class T> class hits_infos : TTree_hits<T>, TTree_particles<T>
//=====================================================================
{
//  friend hits;

  private:
    int _size;
    int _position;
//    std::vector<T> r;
//    std::vector<int> particle;
//    vector<T> _x_v, _y_v, _z_v;
    std::multimap<uint64_t, int> _particleID_TTreeHits_map;

    // particles properties for std root files compatibility only

  public:
    hits_infos ();
    hits_infos (TTree_hits<T>&, TTree_particles<T>&, bool=false);
    inline int size () {return _size;}
    inline std::pair<std::_Rb_tree_iterator<std::pair<const uint64_t, int>>, std::_Rb_tree_iterator<std::pair<const uint64_t, int>>> particleID_equal_range (const uint64_t& ID) {return _particleID_TTreeHits_map.equal_range(ID);}

};
