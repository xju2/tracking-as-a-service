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

#include <vector>

//==================================
template <class T> class space_point
//==================================
{
  std::string _hardware;
  int _barrel_endcap, _layer_disk, _eta_module, _phi_module;
  std::vector<T> _spoint, _cluster1, _cluster2;

  public:
    space_point (const std::string&, const int&, const int&, const int&, const int&,
                 const std::vector<T>&, const std::vector<T>&, const std::vector<T>&);

    inline const std::string& hardware () const {return _hardware;}
    inline const int& barrel_endcap () const {return _barrel_endcap;}
    inline const int& layer_disk () const {return _layer_disk;}
    inline const int& eta_module () const {return _eta_module;}
    inline const int& phi_module () const {return _phi_module;}
    inline const T& x () const {return _spoint[0];}
    inline const T& y () const {return _spoint[1];}
    inline const T& z () const {return _spoint[2];}
    inline const T& cluster1_x () const {return _cluster1[0];}
    inline const T& cluster1_y () const {return _cluster1[1];}
    inline const T& cluster1_z () const {return _cluster1[2];}
    inline const T& cluster2_x () const {return _cluster2[0];}
    inline const T& cluster2_y () const {return _cluster2[1];}
    inline const T& cluster2_z () const {return _cluster2[2];}
};
