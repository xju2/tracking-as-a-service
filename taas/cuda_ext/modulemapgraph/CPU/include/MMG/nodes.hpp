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

#ifndef __cplusplus
#error Must use C++ for the type nodes
#endif

//============================
template <class T> class nodes
//============================
{
  private:
    int _nb_nodes;
    std::vector<uint64_t> _hit_id;
    std::vector<T> _R, _z, _eta, _phi;
    std::vector<int> _ACORN_positions;

  public:
    nodes () {_nb_nodes = 0;}
    nodes (int);

  inline int nb_nodes () {return _nb_nodes;}
  inline std::vector<uint64_t>& hitID () {return _hit_id;}
  inline std::vector<T>& R () {return _R;}
  inline std::vector<T>& z () {return _z;}
  inline std::vector<T>& eta () {return _eta;}
  inline std::vector<T>& phi () {return _phi;}
  inline std::vector<T>& ACORN_positions () {return _ACORN_positions;}
};
