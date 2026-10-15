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

#include "MMG/hits_infos.hpp"

//---------------------------------------------------------------------------------------------
template <class T>
hits_infos<T>::hits_infos (TTree_hits<T>& TTh, TTree_particles<T>& TTp, bool closest_to_vertex)
//---------------------------------------------------------------------------------------------
{
  assert (TTh.size() && TTp.size());

  // loop on particles (faster - no particles set to 0 like in hits)
  for (int pID=0; pID<TTp.size(); pID++){
    TTp.get(pID);
    uint64_t particleID = TTp.particle_ID();
    std::pair<std::_Rb_tree_iterator<std::pair<const uint64_t, int>>, std::_Rb_tree_iterator<std::pair<const uint64_t, int>>> HitRange = TTh.particleID_equal_range (particleID);
    std::multimap<uint64_t, std::pair<int, T>> moduleID_HitRange_map; // moduleID / hit nb / norm of (x_v, y_v, z_v)
    std::multimap<T, int> norm_Hit_map;
    for (std::_Rb_tree_iterator<std::pair<const uint64_t, int>> hit = HitRange.first; hit != HitRange.second; hit++) {
      TTh.get(hit->second);
      // sort HitRange by moduleID
      T x_v = TTh.x() - TTp.vx();
      T y_v = TTh.y() - TTp.vy();
      T z_v = TTh.z() - TTp.vz();
      T norm = std::sqrt (x_v * x_v + y_v * y_v + z_v * z_v);
      moduleID_HitRange_map.insert (std::pair<uint64_t, std::pair<int, T>> (TTh.module_ID(), std::pair<int, T> (hit->second, norm)));
    }

    //-------------------------
    // Split cluster management
    //-------------------------
    // it happens that two hits (or even more) belonging to the same particle are on the same module (i.e split clusters)
    // in that case, as the module map connect a module to another one, only one hit is kept (the farest from the interaction vertex)

    for (std::_Rb_tree_iterator<std::pair<const uint64_t, std::pair<int, T>>> hit = moduleID_HitRange_map.begin(); hit != moduleID_HitRange_map.end();) {
      std::_Rb_tree_iterator<std::pair<const uint64_t, std::pair<int, T>>> next_hit = hit;
      next_hit++;

      if (next_hit == moduleID_HitRange_map.end()) {
        norm_Hit_map.insert (std::pair<T, int> (hit->second.second, hit->second.first));
        hit++;
      }
      else {
        // If on same module keep only the farest (by default), or the closest,
        // depending on the setting
        if (hit->first == next_hit->first) {
          bool erase_next_hit (false);
          if (closest_to_vertex)
            erase_next_hit = hit->second.second < next_hit->second.second;
          else
            erase_next_hit = hit->second.second > next_hit->second.second;
        assert (next_hit != hit);

          if (erase_next_hit)
            moduleID_HitRange_map.erase (next_hit);
          else {
            moduleID_HitRange_map.erase (hit);
            hit = next_hit;
          }
        }
        else {
          norm_Hit_map.insert (std::pair<T, int> (hit->second.second, hit->second.first));
          hit++;
        }
      }
    }

    for (std::pair<T, int> hit : norm_Hit_map)
      _particleID_TTreeHits_map.insert (std::pair<uint64_t, int> (particleID, hit.second));
  }

  _size = _particleID_TTreeHits_map.size();
}
