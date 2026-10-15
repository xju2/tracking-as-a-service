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

#ifndef __cplusplus
#error Must use C++ for the type strip module DB
#endif

#include <iostream>
#include <TString.h>
#include <TChain.h>
#include <vector>
#include <MMG/TTree_hits>

//======================================
template <class T> class strip_module_DB
//======================================
{
  private:
    int _nMods;
    std::vector<T> db_x0;
    std::vector<T> db_y0;
    std::vector<T> db_z0;
    std::vector<int> db_barrel_endcap;
    std::vector<int> db_layer_disk;
    std::vector<int> db_eta_module;
    std::vector<int> db_phi_module;
    std::multimap <int, std::vector<T>> _positions;

  public:
    strip_module_DB () {_nMods=0;}
    strip_module_DB (const std::string&);

    inline int nb_modules () {return _nMods;}
    inline const std::vector<T>& x0 () {return db_x0;}
    inline const std::vector<T>& y0 () {return db_y0;}
    inline const std::vector<T>& z0 () {return db_z0;}
    inline const std::vector<int>& barrel_endcap () {return db_barrel_endcap;}
    inline const std::vector<int>& layer_disk () {return db_layer_disk;}
    inline const std::vector<int>& eta_module () {return db_eta_module;}
    inline const std::vector<int>& phi_module () {return db_phi_module;}

    std::vector<T> get_module_position (const int&, const int&, const int&, const int&);
    std::vector<T> get_module_position (const std::string&, const int&, const int&, const int&, const int&);
    template <class DT> std::vector<T> get_module_position (const int&, const TTree_hits<DT>&);
};
