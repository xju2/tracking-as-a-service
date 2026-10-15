/***********************************************************************
 * Module Map Graph library
 * Track reconstruction for HEP
 * copyright © 2023 ROUGIER Charline
 * copyright © 2023,2024,2025 COLLARD Christophe
 * Laboratoire des 2 Infinis - Toulouse - France (L2IT)
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 *
 * contributors:
 * 2024 Alexis Vallier
 **********************************************************************/

#include "MMG/geometric_cuts.hpp"

//---------------------------------------------------------------------
template<class T>
geometric_cuts<T>::geometric_cuts (T z0_, T dphi, T phi_slope_, T deta)
//---------------------------------------------------------------------
{
  _z0 = z0_;
  _dphi = dphi;
  _phi_slope = phi_slope_;
  _deta = deta;
  _dr = -9999;
}

//---------------------------------------------------------------------------------------------
template<class T>
template<class V>
geometric_cuts<T>::geometric_cuts (const TTree_hits<V>& TThits, const int& it1, const int& it2)
//---------------------------------------------------------------------------------------------
{
  T R1 = TThits.R(it1);
  T z1 = TThits.z(it1);

  T R2 = TThits.R(it2);
  T z2 = TThits.z(it2);

  T R = 0.5 * (R1 + R2);
  _dr = R2 - R1;
  T dz = z2 - z1;

  _deta = TThits.Eta(it2) - TThits.Eta(it1);
  _dphi = TThits.Phi(it2) - TThits.Phi(it1);

  if (_dphi > std::numbers::pi_v<T>) _dphi -= 2 * std::numbers::pi_v<T>;
  else if (_dphi < -std::numbers::pi_v<T>) _dphi += 2 * std::numbers::pi_v<T>;

  if (abs(_dr) > epsilon<T>()) {
    _phi_slope = _dphi / _dr;
    _r_phi_slope = R * _phi_slope;
    _z0 = z1 - R1 * dz / _dr;
  }
  else {
    _z0 = 0;
    _phi_slope = 0;
    _r_phi_slope = 0;
  }
}


//---------------------------------------------------------------------------------------
template <class T>
T Diff_dydx (const TTree_hits<T>& TThits, const int& it1, const int& it2, const int& it3)
//---------------------------------------------------------------------------------------
{
  T dy_12 = TThits.y(it2) - TThits.y(it1);
  T dy_23 = TThits.y(it3) - TThits.y(it2);
  T dx_12 = TThits.x(it2) - TThits.x(it1);
  T dx_23 = TThits.x(it3) - TThits.x(it2);

  if (abs(dx_12) < epsilon<T>()) dx_12 = 0;
  if (abs(dx_23) < epsilon<T>()) dx_23 = 0;

  T diff_dydx = 0;
  if (dx_12 && dx_23)
    diff_dydx = (dy_12 / dx_12) - (dy_23 / dx_23);
  else
    if (dx_12)
      diff_dydx = dy_12 / dx_12; // force dy_23/dx_23 to 0
    else
      if (dx_23)
        diff_dydx = -dy_23 / dx_23; // force dy_12/dx_12 to 0

  return diff_dydx;
}


//----------------------------------------------------------------------------------------
template <class T>
 T Diff_dzdr (const TTree_hits<T>& TThits, const int& it1, const int& it2, const int& it3)
//----------------------------------------------------------------------------------------
{
  T dz_12 = TThits.z(it2) - TThits.z(it1);
  T dz_23 = TThits.z(it3) - TThits.z(it2);
  T dr_12 = TThits.R(it2) - TThits.R(it1);
  T dr_23 = TThits.R(it3) - TThits.R(it2);

  if (abs(dr_12) < epsilon<T>()) dr_12 = 0;
  if (abs(dr_23) < epsilon<T>()) dr_23 = 0;
  T diff_dzdr = 0;

  if (dr_12 && dr_23)
    diff_dzdr = (dz_12 / dr_12) - (dz_23 / dr_23);
  else
    if (dr_12)
      diff_dzdr = dz_12 / dr_12; // force dz_23/dr_23 to 0
    else
      if (dr_23)
        diff_dzdr = -dz_23 / dr_23; // force dz_12/dr_12 to 0

    return diff_dzdr;
}


//------------------------------------------------------------------------------------------------------------
template <class T>
void geometric_cuts<T>::branch (TTree* treeGeometricCuts, const std::string& modules, const std::string& name)
//------------------------------------------------------------------------------------------------------------
{
  treeGeometricCuts -> Branch ((std::string("z0")+name+std::string("_")+modules).c_str(), &_z0);
  treeGeometricCuts -> Branch ((std::string("dphi")+name+std::string("_")+modules).c_str(), &_dphi);
  treeGeometricCuts -> Branch ((std::string("phiSlope")+name+std::string("_")+modules).c_str(), &_phi_slope);
  treeGeometricCuts -> Branch ((std::string("deta")+name+std::string("_")+modules).c_str(), &_deta);
}


//------------------------------------------------------------------------------------------------------------------------
template <class T>
void geometric_cuts<T>::set_branch_address (TTree* treeGeometricCuts, const std::string& modules, const std::string& name)
//------------------------------------------------------------------------------------------------------------------------
{
  treeGeometricCuts -> SetBranchAddress ((std::string("z0")+name+std::string("_")+modules).c_str(), &_z0);
  treeGeometricCuts -> SetBranchAddress ((std::string("dphi")+name+std::string("_")+modules).c_str(), &_dphi);
  treeGeometricCuts -> SetBranchAddress ((std::string("phiSlope")+name+std::string("_")+modules).c_str(), &_phi_slope);
  treeGeometricCuts -> SetBranchAddress ((std::string("deta")+name+std::string("_")+modules).c_str(), &_deta);
}


//-----------------------------------------------------------------------------
template <class Tf>
bool operator == (const geometric_cuts<Tf>& ct1, const geometric_cuts<Tf>& ct2)
//-----------------------------------------------------------------------------
{
  return (ct1._z0 == ct2._z0) && (ct1._dphi == ct2._dphi) && (ct1._phi_slope == ct2._phi_slope) && (ct1._deta == ct2._deta);
}


//-----------------------------------------------------------------------------
template <class Tf>
bool operator != (const geometric_cuts<Tf>& ct1, const geometric_cuts<Tf>& ct2)
//-----------------------------------------------------------------------------
{
  return !(ct1==ct2);
}


//-----------------------------------------------------------------------
template <class Tf>
std::ostream& operator << (std::ostream& s, const geometric_cuts<Tf>& ct)
//-----------------------------------------------------------------------
{
  s << "z0 = " << ct._z0 << std::endl;
  s << "Dphi = " << ct._dphi << std::endl;
  s << "Phi Slope = " << ct._phi_slope << std::endl;
  s << "Deta = " << ct._deta << std::endl;

  return s;
}


//-----------------------------------------------
template<class T>
bool geometric_cuts<T>::check_cut_values () const
//-----------------------------------------------
{
  if (std::isnan (_z0)) {
    std::cout << "Error: z0 Not A Number, you cannot add the link to the module map." << std::endl;
    return false;
  }

  if (std::isnan (_dphi)) {
    std::cout << "Error: dphi Not A Number, you cannot add the link to the module map." << std::endl;
    return false;
  }

  if (std::isnan (_phi_slope)) {
    std::cout << "Error: phi_slope Not A Number, you cannot add the link to the module map." << std::endl;
    return false;
  }

  if (std::isnan(_deta)) {
    std::cout << "Error: deta Not A Number, you cannot add the link to the module map." << std::endl;
    return false;
  }

  if (std::isinf (_z0)) {
    std::cout << "Error: z0 is infinite, you cannot add the link to the module map. " << std::endl;
    return false;
  }

  if (std::isinf (_dphi)) {
    std::cout << "Error: dphi is infinite, you cannot add the link to the module map." << std::endl;
    return false;
  }

  if (std::isinf (_phi_slope)) {
    std::cout << "Error: phi_slope is infinite, you cannot add the link to the module map." << std::endl;
    return false;
  }

  if (std::isinf(_deta)) {
    std::cout << "Error: deta is infinite, you cannot add the link to the module map." << std::endl;
    return false;
  }

  return true;
}


//-----------------------------------------------------
template<class T>
void geometric_cuts<T>::add_numerical_error (T epsilon)
//-----------------------------------------------------
{
  _z0 += abs(_z0) * epsilon;
  _dphi += abs(_dphi) * epsilon;
  _phi_slope += abs(_phi_slope) * epsilon;
  _deta += abs(_deta) * epsilon;
  exit(0);
}


//---------------------------------------------------------
template<class T>
void geometric_cuts<T>::min (const geometric_cuts<T>& cuts)
//---------------------------------------------------------
{
  if (_z0 > cuts.z0()) _z0 = cuts.z0();
  if (_dphi > cuts.d_phi()) _dphi = cuts.d_phi();
  if (_phi_slope > cuts.phi_slope()) _phi_slope = cuts.phi_slope();
  if (_deta > cuts.d_eta()) _deta = cuts.d_eta();
}


//---------------------------------------------------------
template<class T>
void geometric_cuts<T>::max (const geometric_cuts<T>& cuts)
//---------------------------------------------------------
{
  if (_z0 < cuts.z0()) _z0 = cuts.z0();
  if (_dphi < cuts.d_phi()) _dphi = cuts.d_phi();
  if (_phi_slope < cuts.phi_slope()) _phi_slope = cuts.phi_slope();
  if (_deta < cuts.d_eta()) _deta = cuts.d_eta();
}

//---------------------------------------------------------
template<class T>
void geometric_cuts<T>::sum (const geometric_cuts<T>& cuts)
//---------------------------------------------------------
{
  _z0 += cuts.z0();
  _dphi += cuts.d_phi();
  _phi_slope += cuts.phi_slope();
  _deta += cuts.d_eta();
}


//-----------------------------------------------------------
template<class T>
void geometric_cuts<T>::sumSq (const geometric_cuts<T>& cuts)
//-----------------------------------------------------------
{
  // pow uses double instead of float when T=float - same computation time than x*x for double
  _z0 += pow (cuts.z0(), 2);
  _dphi += pow (cuts.d_phi(), 2);
  _phi_slope += pow (cuts.phi_slope(), 2);
  _deta += pow (cuts.d_eta(), 2);
}
