/***********************************************************************
 * Module Map Graph library
 * Track reconstruction for HEP
 * copyright © 2022,2023 ROUGIER Charline
 * copyright © 2022,2023,2024 COLLARD Christophe
 * Laboratoire des 2 Infinis - Toulouse - France (L2IT)
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 **********************************************************************/

#include "MMG/edge.hpp"

//--------------------------------------------------------------------------------------------------------------
template <class T>
edge<T>::edge (const T& deta, const T& dphi, const T& dr, const T& dz, const T& phi_slope, const T& r_phi_slope)
//--------------------------------------------------------------------------------------------------------------
{
  _extra_features = false;
  _dEta = deta;
  _dPhi = dphi;
  _dr = dr;
  _dz = dz;
  _phi_slope = phi_slope;
  _r_phi_slope = r_phi_slope;
}

//---------------------------------------------
template <class T>
edge<T>& edge<T>::operator = (const edge<T>& e)
//---------------------------------------------
{
  _dEta = e._dEta;
  _dPhi = e._dPhi;
  _dr = e._dr;
  _dz = e._dz;
  _phi_slope = e._phi_slope;
  _r_phi_slope = e._r_phi_slope;
  _new_hit_in_x = e._new_hit_in_x;
  _new_hit_in_y = e._new_hit_in_y;
  _new_hit_in_z = e._new_hit_in_z;
  _new_hit_out_x = e._new_hit_out_x;
  _new_hit_out_y = e._new_hit_out_y;
  _new_hit_out_z = e._new_hit_out_z;
  _extra_features = e._extra_features;

  if (_extra_features) {
    _pass_strip_range = e._pass_strip_range;
    _new_dEta = e._new_dEta;
    _new_dPhi = e._new_dPhi;
    _new_dr = e._new_dr;
    _new_dz = e._new_dz;
    _new_PhiSlope = e._new_PhiSlope;
    _new_rPhiSlope = e._new_rPhiSlope;
  }

  return (*this);
}

//--------------------------------------------------------------------------------------------------------------------------------------
template <class T>
void edge<T>::barrel_strip_hit (const TTree_hits<T>& TThits, const int& it1, const int& it2, strip_module_DB<data_type2>& StripModuleDB)
//--------------------------------------------------------------------------------------------------------------------------------------
{
  _extra_features = true;
  int in, out = in = it1;
  if (TThits.R(it1) > TThits.R(it2))
    in = it2;
  else
    out = it2;

  std::vector<data_type2> in_module  = StripModuleDB.get_module_position (in, TThits); //StripModuleDB.get_module_position (in, TThits.hardware(in), TThits.barrel_endcap(in), TThits.layer_disk(in), TThits.eta_module(in), TThits.phi_module(in));
  std::vector<data_type2> out_module = StripModuleDB.get_module_position (out, TThits); //.hardware(out), TThits.barrel_endcap(out), TThits.layer_disk(out), TThits.eta_module(out), TThits.phi_module(out));0

  _strip_hit_pair = strip_hit_pair (TThits, in, out, StripModuleDB);

//  _strip_hit_pair = strip_hit_pair (TThits, in, out, in_module, out_module);
 
  std::vector<data_type2> new_hit_in = _strip_hit_pair.new_hit_in ();
  _new_hit_in_x = new_hit_in.at(0);
  _new_hit_in_y = new_hit_in.at(1);
  _new_hit_in_z = new_hit_in.at(2);

  std::vector<data_type2> new_hit_out = _strip_hit_pair.new_hit_out ();
  _new_hit_out_x = new_hit_out.at(0);
  _new_hit_out_y = new_hit_out.at(1);
  _new_hit_out_z = new_hit_out.at(2);

  _pass_strip_range = _strip_hit_pair.pass_strip_range();
  assert (_pass_strip_range == 0  ||  _pass_strip_range == 1);

  _new_dEta = _strip_hit_pair.d_eta ();
  _new_dPhi = _strip_hit_pair.d_phi () / std::numbers::pi_v<T>;
  _new_dr = _strip_hit_pair.d_r() * 0.001;
  _new_dz = _strip_hit_pair.d_z() * 0.001;

  _new_PhiSlope = (_new_dr != 0.) ? _new_dPhi / _new_dr : 0;
  if (_new_PhiSlope < -1000.) _new_PhiSlope = -1000.;
  else if (_new_PhiSlope > 1000.) _new_PhiSlope = 1000.;

  _new_rPhiSlope = _new_PhiSlope * _strip_hit_pair.sum_Perp() * .5;
  if (_new_rPhiSlope<-1000.) _new_rPhiSlope = -1000.;
  else if (_new_rPhiSlope>1000.) _new_rPhiSlope = 1000.;

/*

  char hdw_in = (TThits.hardware(in)=="PIXEL") ? 1 : 2;
  char hdw_out = (TThits.hardware(out)=="PIXEL") ? 1 : 2;
  assert(hdw_in == 1 || hdw_in == 2);
  assert(hdw_out == 1 || hdw_out == 2);


  StripHitPair strip_pair (hdw_in, TThits.barrel_endcap(in), TThits.layer_disk(in),
         TThits.x(in), TThits.y(in), TThits.z(in),
			   TThits.x_cluster1(in), TThits.y_cluster1(in), TThits.z_cluster1(in),
			   TThits.x_cluster2(in), TThits.y_cluster2(in), TThits.z_cluster2(in),
			   in_module.at(0), in_module.at(1), in_module.at(2),
			   hdw_out, TThits.barrel_endcap(out), TThits.layer_disk(out),
			   TThits.x(out), TThits.y(out), TThits.z(out),
			   TThits.x_cluster1(out), TThits.y_cluster1(out), TThits.z_cluster1(out),
			   TThits.x_cluster2(out), TThits.y_cluster2(out), TThits.z_cluster2(out),
			   out_module.at(0), out_module.at(1), out_module.at(2));

  bool _old_pass_strip_range = strip_pair.PassStripRange();

  double _old_hit_in_x = strip_pair.NewSpointInX();
  double _old_hit_in_y = strip_pair.NewSpointInY();
  double _old_hit_in_z = strip_pair.NewSpointInZ();
  double _old_hit_out_x = strip_pair.NewSpointOutX();
  double _old_hit_out_y = strip_pair.NewSpointOutY();
  double _old_hit_out_z = strip_pair.NewSpointOutZ();

  TVector3 old_in (_old_hit_in_x, _old_hit_in_y, _old_hit_in_z);
  TVector3 old_out (_old_hit_out_x, _old_hit_out_y, _old_hit_out_z);

  double _old_dEta = old_out.Eta() - old_in.Eta();
  double _old_dr = (old_out.Perp() - old_in.Perp()) * 0.001;
  double _old_dPhi = old_out.DeltaPhi (old_in) / TMath::Pi();
  double _old_dz = (old_out.z() - old_in.z()) * 0.001;

  double _old_PhiSlope = (_old_dr != 0.) ? _old_dPhi / _old_dr : 0.;
  if (_old_PhiSlope<-1000.) _old_PhiSlope = -1000.;
  else if (_old_PhiSlope>1000.) _old_PhiSlope = 1000.;

  double _old_rPhiSlope = _old_PhiSlope * (old_in.Perp() + old_out.Perp()) / 2.;
  if (_old_rPhiSlope<-1000.) _old_rPhiSlope = -1000.;
  else if (_old_rPhiSlope>1000.) _old_rPhiSlope = 1000.;



  if (abs (_old_hit_in_x - _new_hit_in_x) > 100 * epsilon * std::max (1., 10 * abs(_old_hit_in_x))) {
    std::cout << "in x" << std::endl;
    std::cout << abs (_old_hit_in_x - _new_hit_in_x) << std::endl;
    std::cout << 10 * epsilon * std::max (1., abs(_old_hit_in_x)) << std::endl;
    std::cout << std::setprecision (8) << _new_hit_in_x << std::endl;
    std::cout << std::setprecision (8) << _old_hit_in_x << std::endl;
    exit(0);
  }

  if (abs (_old_hit_in_y - _new_hit_in_y) > 100 * epsilon * std::max (1.,10 * abs(_old_hit_in_y))) {
    std::cout << "in y" << std::endl;
    std::cout << std::setprecision (8) << _new_hit_in_y << std::endl;
    std::cout << std::setprecision (8) << _old_hit_in_y << std::endl;
    exit(0);
  }

  if (abs (_old_hit_in_z - _new_hit_in_z) > 100 * epsilon * std::max (1., 10 * abs(_old_hit_in_z))) {
    std::cout << "in z" << std::endl;
    std::cout << std::setprecision (8) << _new_hit_in_z << std::endl;
    std::cout << std::setprecision (8) << _old_hit_in_z << std::endl;
    exit(0);
  }

  if (abs (_old_hit_out_x - _new_hit_out_x) > 100 * epsilon * std::max (1., 10 * abs(_old_hit_out_x))) {
    std::cout << "out x" << std::endl;
    std::cout << abs (_old_hit_out_x - _new_hit_out_x) << std::endl;
    std::cout << 10 * epsilon * std::max (1., abs(_old_hit_out_x)) << std::endl;
    std::cout << std::setprecision (8) << _new_hit_out_x << std::endl;
    std::cout << std::setprecision (8) << _old_hit_out_x << std::endl;
    exit(0);
  }

  if (abs (_old_hit_out_y - _new_hit_out_y) > 100 * epsilon * std::max (1., 10 * abs(_old_hit_out_y))) {
    std::cout << "out y" << std::endl;
    std::cout << std::setprecision (8) << _new_hit_out_y << std::endl;
    std::cout << std::setprecision (8) << _old_hit_out_y << std::endl;
    exit(0);
  }

  if (abs (_old_hit_out_z - _new_hit_out_z) > 100 * epsilon * std::max (1., 10 * abs(_old_hit_out_z))) {
    std::cout << "out z" << std::endl;
    std::cout << std::setprecision (8) << _new_hit_out_z << std::endl;
    std::cout << std::setprecision (8) << _old_hit_out_z << std::endl;
    exit(0);
  }

  if (abs (_old_dEta - _new_dEta) > 100 * epsilon * std::max (1., 10 * abs(_old_dEta))) {
    std::cout << "out dEta" << std::endl;
    std::cout << std::setprecision (8) << _new_dEta << std::endl;
    std::cout << std::setprecision (8) << _old_dEta << std::endl;
    exit(0);
  }

  if (abs (_old_dr - _new_dr) > 100 * epsilon * std::max (1., 10 * abs(_old_dr))) {
    std::cout << "out dr" << std::endl;
    std::cout << std::setprecision (8) << _new_dr << std::endl;
    std::cout << std::setprecision (8) << _old_dr << std::endl;
    exit(0);
  }

  if (abs (_old_dPhi - _new_dPhi) > 100 * epsilon * std::max (1., abs(_old_dPhi))) {
    std::cout << "out dPhi" << std::endl;
    std::cout << std::setprecision (8) << _new_dPhi << std::endl;
    std::cout << std::setprecision (8) << _old_dPhi << std::endl;
    exit(0);
  }

  if (abs (_old_dz - _new_dz) > 100 * epsilon * std::max (1., 10 * abs(_old_dz))) {
    std::cout << "out dz" << std::endl;
    std::cout << std::setprecision (8) << _new_dz << std::endl;
    std::cout << std::setprecision (8) << _old_dz << std::endl;
    exit(0);
  }

  if (abs (_old_PhiSlope - _new_PhiSlope) > 10 * epsilon * std::max (1., 10 * abs(_old_PhiSlope))) {
    std::cout << "out PhiSlope" << std::endl;
    std::cout << std::setprecision (8) << _new_PhiSlope << std::endl;
    std::cout << std::setprecision (8) << _old_PhiSlope << std::endl;
    std::cout << _old_dz << std::endl;
    std::cout << std::endl;
//    exit(0);
  }

  if (abs (_old_rPhiSlope - _new_rPhiSlope) > 100 * epsilon * std::max (1., 10 * abs(_old_rPhiSlope))) {
    std::cout << "out rPhiSlope" << std::endl;
    std::cout << std::setprecision (8) << _new_rPhiSlope << std::endl;
    std::cout << std::setprecision (8) << _old_rPhiSlope << std::endl;
    exit(0);
  }


    _new_hit_in_x = _old_hit_in_x;
    _new_hit_in_y = _old_hit_in_y;
    _new_hit_in_z = _old_hit_in_z;
    _new_hit_out_x = _old_hit_out_x;
    _new_hit_out_y = _old_hit_out_y;
    _new_hit_out_z = _old_hit_out_z;
    _new_dEta = _old_dEta;
    _new_dPhi = _old_dPhi;
    _new_dr = _old_dr;
    _new_dz = _old_dz;
    _new_PhiSlope = _old_PhiSlope;
    _new_rPhiSlope = _old_rPhiSlope;
    _pass_strip_range = _old_pass_strip_range;
*/
}
