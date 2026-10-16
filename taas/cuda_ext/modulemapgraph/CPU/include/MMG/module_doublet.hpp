/***********************************************************************
 * Module Map Graph library
 * Track reconstruction for HEP
 * copyright © 2022,2023 ROUGIER Charline
 * copyright © 2022,2023,2024,2025 COLLARD Christophe
 * Laboratoire des 2 Infinis - Toulouse - France (L2IT)
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 *
 * contributors:
 * 2024 Alexis Vallier
 ***********************************************************************/

#ifndef __cplusplus
#error Must use C++ for the type module doublet
#endif

#include <iostream>
#include <iomanip>
#include <TTree.h>
#include <MMG/particles> // "particles.hpp"
#include <MMG/geometric_cuts>

//=====================================
template <class T> class module_doublet
//=====================================
{
  private:
    geometric_cuts<T> _cuts_min;
    geometric_cuts<T> _cuts_max;
    geometric_cuts<T> _cuts_sum;
    geometric_cuts<T> _cuts_sumSq;

  public:
    module_doublet ();
    module_doublet (const geometric_cuts<T>&);
    ~module_doublet () {}

    template <class Tf> friend bool operator == (const module_doublet<Tf>&, const module_doublet<Tf>&);
    template <class Tf> friend bool operator != (const module_doublet<Tf>&, const module_doublet<Tf>&);
    // overload for iostream
    template <class Tf> friend std::ostream& operator << (std::ostream&, const module_doublet<Tf>&);
//    friend istream& operator >> (istream&, const module_triplet&);

    // overload for iostream
    template <class Tf> friend std::ostream& operator << (std::ostream&, const module_doublet<Tf>&);

    void branch (TTree*, const std::string&);
    void set_branch_address (TTree*, const std::string&);

    inline const T& z0_min () const {return _cuts_min.z0();}
    inline const T& z0_max () const {return _cuts_max.z0();}
    inline const T& dphi_min () const {return _cuts_min.d_phi();}
    inline const T& dphi_max () const {return _cuts_max.d_phi();}
    inline const T& phi_slope_min () const {return _cuts_min.phi_slope();}
    inline const T& phi_slope_max () const {return _cuts_max.phi_slope();}
    inline const T& deta_min () const {return _cuts_min.d_eta();}
    inline const T& deta_max () const {return _cuts_max.d_eta();}

    inline const T& z0_sum () const {return _cuts_sum.z0();}
    inline const T& deta_sum () const {return _cuts_sum.d_eta();}
    inline const T& dphi_sum () const {return _cuts_sum.d_phi();}
    inline const T& phi_slope_sum () const {return _cuts_sum.phi_slope();}
    inline const T& z0_sumSq () const {return _cuts_sumSq.z0();}
    inline const T& deta_sumSq () const {return _cuts_sumSq.d_eta();}
    inline const T& dphi_sumSq () const {return _cuts_sumSq.d_phi();}
    inline const T& phi_slope_sumSq () const {return _cuts_sumSq.phi_slope();}

    inline geometric_cuts<T> cuts_min () const {return _cuts_min;}
    inline geometric_cuts<T> cuts_max () const {return _cuts_max;}
    inline geometric_cuts<T> cuts_sum () const {return _cuts_sum;}
    inline geometric_cuts<T> cuts_sumSq () const {return _cuts_sumSq;}

    inline T& z0_min () {return _cuts_min.z0();}
    inline T& z0_max () {return _cuts_max.z0();}
    inline T& dphi_min () {return _cuts_min.d_phi();}
    inline T& dphi_max () {return _cuts_max.d_phi();}
    inline T& phi_slope_min () {return _cuts_min.phi_slope();}
    inline T& phi_slope_max () {return _cuts_max.phi_slope();}
    inline T& deta_min () {return _cuts_min.d_eta();}
    inline T& deta_max () {return _cuts_max.d_eta();}

    inline T& z0_sum () {return _cuts_sum.z0();}
    inline T& deta_sum () {return _cuts_sum.d_eta();}
    inline T& dphi_sum () {return _cuts_sum.d_phi();}
    inline T& phi_slope_sum () {return _cuts_sum.phi_slope();}
    inline T& z0_sumSq () {return _cuts_sumSq.z0();}
    inline T& deta_sumSq () {return _cuts_sumSq.d_eta();}
    inline T& dphi_sumSq () {return _cuts_sumSq.d_phi();}
    inline T& phi_slope_sumSq () {return _cuts_sumSq.phi_slope();}

    inline bool min_equal_max () {return _cuts_min == _cuts_max;}
    inline bool cuts (const geometric_cuts<T> SP_cuts) {return _cuts_min.min_cuts (SP_cuts) || _cuts_max.max_cuts (SP_cuts);}
    bool add_occurence (const geometric_cuts<T>&);
    void add_occurences (const module_doublet<T>&);
    void add_numerical_error ();
};
