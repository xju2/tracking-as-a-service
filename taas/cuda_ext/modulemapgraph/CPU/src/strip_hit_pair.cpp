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

#include <MMG/strip_hit_pair.hpp>


//-------------------------------------
template <class T, class DT>
strip_hit_pair<T,DT>::strip_hit_pair ()
//-------------------------------------
{
  _hit_in = _hit_out = 0;
}

//---------------------------------------------------------------------------------------------------------------------
template <class T, class DT>
strip_hit_pair<T,DT>::strip_hit_pair (const TTree_hits<DT>& TThits, const int& ht1,
                                      const int& ht2, strip_module_DB<T>& StripModuleDB){
//---------------------------------------------------------------------------------------------------------------------
  _TThits = &TThits;
  if (TThits.R(ht1) > TThits.R(ht2)) {
    _hit_in = ht2;
    _hit_out = ht1;
  }
  else {
    _hit_in = ht1;
    _hit_out = ht2;
  }

  // retrieve the position of the strip modules
  std::vector<T> in_module  = StripModuleDB.get_module_position (_hit_in, TThits);
  std::vector<T> out_module = StripModuleDB.get_module_position (_hit_out, TThits);

//  TThits.get (_hit_in);
//  std::vector<T> in_module = StripModuleDB.get_module_position (TThits.hardware(), TThits.barrel_endcap(),
//                                                        TThits.layer_disk(), TThits.eta_module(), TThits.phi_module());

//  TThits.get (_hit_out);
//  std::vector<T> out_module = StripModuleDB.get_module_position (TThits.hardware(), TThits.barrel_endcap(),
//                                                        TThits.layer_disk(), TThits.eta_module(), TThits.phi_module());

  bool in_strip_barrel = (TThits.hardware() == "STRIP"  &&  TThits.barrel_endcap() == 0);
  bool out_strip_barrel = (TThits.hardware() == "STRIP" && TThits.barrel_endcap() == 0);

  _pass_strip_range = true;

  if (in_strip_barrel || out_strip_barrel){
    T tilt_in  = barrel_strip_module_tilt (TThits.layer_disk(_hit_in));
    T tilt_out = barrel_strip_module_tilt (TThits.layer_disk(_hit_out));

    // 1st iteration

    std::vector<T> edge;
    edge.push_back (_TThits->x_cluster1(_hit_out) - _TThits->x_cluster1(_hit_in));
    edge.push_back (_TThits->y_cluster1(_hit_out) - _TThits->y_cluster1(_hit_in));
    edge.push_back (_TThits->z_cluster1(_hit_out) - _TThits->z_cluster1(_hit_in));
    T perp_ = perp (edge);

    T theta0 = get_edge_theta (edge, perp_);

    T alphaIn0, alphaOut0;
    std::tie (alphaIn0, alphaOut0) = get_edge_alphas (perp_);

    std::vector<T> hIn0, hOut0;
    if (in_strip_barrel) hIn0 = hit_global_position (in_module, _hit_in, tilt_in, alphaIn0, theta0);
//  if (inStripBarrel)   hIn  = GetGlobPosBarrHit   (module[0], cluster1[0], cluster2[0], sinTiltIn,  alpha0[0], theta0, 1);

    else hIn0 = std::vector<T> {TThits.x(_hit_in), TThits.y(_hit_in), TThits.z(_hit_in)};

    if (out_strip_barrel) hOut0 = hit_global_position (out_module, _hit_out, tilt_out, alphaOut0, theta0);
    else hOut0 = std::vector<T> {TThits.x(_hit_out), TThits.y(_hit_out), TThits.z(_hit_out)};

    // 2nd iteration

    edge = hOut0;
    std::transform (edge.begin(), edge.end(), hIn0.begin(), edge.begin(), std::minus<T>());
    perp_ = perp (edge);
    _theta = get_edge_theta (edge, perp_);
    std::tie (_alpha_in, _alpha_out) = get_edge_alphas (hIn0, hOut0, perp_);

    if (in_strip_barrel) {
      _new_spoint_in = hit_global_position (in_module, _hit_in, tilt_in, _alpha_in, _theta);
    // check compatibility of the hit pair direction with the strip cluster pairs
      T cut = TThits.layer_disk (_hit_in) <= 1 ? 17.05 : 29.1;
      T zrel = std::abs (_new_spoint_in.at(2) - TThits.z_cluster1 (_hit_in));
      if (zrel > cut) _pass_strip_range = false;
    }
    else _new_spoint_in = std::vector<T> {TThits.x(_hit_in), TThits.y(_hit_in), TThits.z(_hit_in)};

    if (out_strip_barrel) {
      _new_spoint_out = hit_global_position (out_module, _hit_out, tilt_out, _alpha_out, _theta);
    // check compatibility of the hit pair direction with the strip cluster pairs
      T cut = TThits.layer_disk (_hit_out) <= 1 ? 17.05 : 29.1;
      T zrel = std::abs (_new_spoint_out.at(2) - TThits.z_cluster1 (_hit_out));
      if (zrel > cut) _pass_strip_range = false;
    }
    else _new_spoint_out = std::vector<T> {TThits.x(_hit_out), TThits.y(_hit_out), TThits.z(_hit_out)};
  }
  else {
    _theta = -9999;
    _alpha_in = -9999;
    _alpha_out = -9999;
    _new_spoint_in = std::vector<T> {TThits.x(_hit_in), TThits.y(_hit_in), TThits.z(_hit_in)};
    _new_spoint_out = std::vector<T> {TThits.x(_hit_out), TThits.y(_hit_out), TThits.z(_hit_out)};
  }

  _Perp_in = perp (_new_spoint_in);
  _Perp_out = perp (_new_spoint_out);
}

//---------------------------------------------------------------------------------------------------------------------
template <class T, class DT>
T strip_hit_pair<T,DT>::perp (const std::vector<T>& v){
//---------------------------------------------------------------------------------------------------------------------
  return sqrt (pow (v.at(0), 2) + pow (v.at(1), 2)); //std::sqrt (x*x + y*y);
}

//---------------------------------------------------------------------------------------------------------------------
template <class T, class DT>
T strip_hit_pair<T,DT>::perp (const int& i){
//---------------------------------------------------------------------------------------------------------------------
  T x = _TThits -> x_cluster1 (i);
  T y = _TThits -> y_cluster1 (i);

  return std::sqrt (x*x + y*y);
}

//---------------------------------------------------------------------------------------------------------------------
template <class T, class DT>
T strip_hit_pair<T,DT>::norm2 (const std::vector<T>& v){
//---------------------------------------------------------------------------------------------------------------------
  return std::sqrt (pow (v.at(0), 2) + pow (v.at(1), 2) + pow (v.at(2), 2));
}

//---------------------------------------------------------------------------------------------------------------------
template <class T, class DT>
T strip_hit_pair<T,DT>::norm2_cluster1 (const int& i){
//---------------------------------------------------------------------------------------------------------------------
  T x = _TThits -> x_cluster1(i);
  T y = _TThits -> y_cluster1(i);
  T z = _TThits -> z_cluster1(i);

  return std::sqrt (x*x + y*y + z*z);
}

//---------------------------------------------------------------------------------------------------------------------
template <class T, class DT>
T strip_hit_pair<T,DT>::Eta (const int& ht){
//---------------------------------------------------------------------------------------------------------------------
  T norm = norm2_cluster1 (ht);
  T cos_theta = !norm ? 1 : (T) _TThits -> z(ht) / norm;

   if (abs (cos_theta) < 1) return -0.5 * log ((1. - cos_theta) / (1. + cos_theta));
   if (! _TThits -> z(ht)) return 0;
   //Warning("PseudoRapidity","transvers momentum = 0! return +/- 10e10");
   if (_TThits -> z(ht) > 0) return 10e10;
   else return -10e10;
}

//---------------------------------------------------------------------------------------------------------------------
template <class T, class DT>
T strip_hit_pair<T,DT>::d_eta (){
//---------------------------------------------------------------------------------------------------------------------
  T deta = 0;
  T norm = norm2 (_new_spoint_out);
  T cos_theta = !norm ? 1 : _new_spoint_out.at(2) / norm;

  if (pow (cos_theta, 2) < 1) deta = -0.5 * log ((1 - cos_theta) / (1 + cos_theta));
  else if (!_new_spoint_out.at(2)) deta = 0;
  //Warning("PseudoRapidity","transverse momentum = 0! return +/- 10e10");
       else if (_new_spoint_out.at(2) > 0) deta = 10e10;
            else deta= -10e10;

  norm = norm2 (_new_spoint_in);
  cos_theta = !norm ? 1 : _new_spoint_in.at(2) / norm;

  if (pow (cos_theta, 2) < 1) deta -= -0.5 * log ((1 - cos_theta) / (1 + cos_theta));
  else if (_new_spoint_in.at(2) > 0) deta -= 10e10;
  else if (_new_spoint_in.at(2) < 0) deta -= -10e10;

  return deta;
}

//---------------------------------------------------------------------------------------------------------------------
template <class T, class DT>
T strip_hit_pair<T,DT>::Phi (const int& hit){
//---------------------------------------------------------------------------------------------------------------------
  return _TThits -> x_cluster1(hit) == 0 && _TThits -> y_cluster1(hit) == 0 ? 0 : TMath::ATan2(_TThits -> y_cluster1(hit), _TThits -> x_cluster1(hit));
}

//---------------------------------------------------------------------------------------------------------------------
template <class T, class DT>
T strip_hit_pair<T,DT>::Phi (const std::vector<T>& hit){
//---------------------------------------------------------------------------------------------------------------------
  return (!hit.at(0) && !hit.at(1)) ? 0 : TMath::ATan2(hit.at(1), hit.at(0));
}

//---------------------------------------------------------------------------------------------------------------------
template <class T, class DT>
T strip_hit_pair<T,DT>::delta_phi (const T& hitOut, const T& hitIn){
//---------------------------------------------------------------------------------------------------------------------
  T kPI = std::numbers::pi_v<T>;
  T kTWOPI = 2 * kPI;

  T Delta_Phi = Phi (hitOut) - Phi(hitIn);
  if(std::isnan (Delta_Phi))
    gROOT -> Error ("TVector2::Phi_mpi_pi","function called with NaN");

  else {
    while (Delta_Phi >= kPI) Delta_Phi -= kTWOPI;
    while (Delta_Phi < -kPI) Delta_Phi += kTWOPI;
  }

  return Delta_Phi;
}

//---------------------------------------------------------------------------------------------------------------------
template <class T, class DT>
T strip_hit_pair<T,DT>::delta_phi (const std::vector<T>& hitOut, const std::vector<T>& hitIn){
//---------------------------------------------------------------------------------------------------------------------
  T kPI = std::numbers::pi_v<T>;
  T kTWOPI = 2 * kPI;

  T Delta_Phi = Phi (hitOut) - Phi(hitIn);
  if(std::isnan (Delta_Phi))
    gROOT -> Error ("TVector2::Phi_mpi_pi","function called with NaN");

  else {
    while (Delta_Phi >= kPI) Delta_Phi -= kTWOPI;
    while (Delta_Phi < -kPI) Delta_Phi += kTWOPI;
  }

  return Delta_Phi;
}

//---------------------------------------------------------------------------------------------------------------------
template <class T, class DT>
T strip_hit_pair<T,DT>::d_phi (){
//---------------------------------------------------------------------------------------------------------------------
  T kPI = std::numbers::pi_v<T>;
  T kTWOPI = 2 * kPI;

  T Delta_Phi = Phi (_new_spoint_out) - Phi (_new_spoint_in);
  if (std::isnan (Delta_Phi))
    gROOT -> Error ("TVector2::Phi_mpi_pi","function called with NaN");
  else {
    while (Delta_Phi >= kPI) Delta_Phi -= kTWOPI;
    while (Delta_Phi < -kPI) Delta_Phi += kTWOPI;
  }

  return Delta_Phi;
}

//---------------------------------------------------------------------------------------------------------------------
template <class T, class DT>
std::vector<T> strip_hit_pair<T,DT>::rotate_z_axis (const std::vector<T>& v, const T& angle){
//---------------------------------------------------------------------------------------------------------------------
  T sin = std::sin (angle);
  T cos = std::cos (angle);

  std::vector<T> v_rot;
  v_rot.push_back (cos * v.at(0) - sin * v.at(1));
  v_rot.push_back (sin * v.at(0) + cos * v.at(1));
  v_rot.push_back (v.at(2));

  return v_rot;
}

//---------------------------------------------------------------------------------------------------------------------
template <class T, class DT>
std::vector<T> strip_hit_pair<T,DT>::rotate_z_axis (const T& x, const T& y, const T& z, const T& angle){
//---------------------------------------------------------------------------------------------------------------------
  T sin = std::sin (angle);
  T cos = std::cos (angle);

  std::vector<T> v_rot;
  v_rot.push_back (cos * x - sin * y);
  v_rot.push_back (sin * x + cos * y);
  v_rot.push_back (z);

  return v_rot;
}

//---------------------------------------------------------------------------------------------------------------------
template <class T, class DT>
T strip_hit_pair<T,DT>::barrel_strip_module_tilt (const char& layer){
//---------------------------------------------------------------------------------------------------------------------
  T asin;

  switch (layer) {
    case 0: asin = std::asin (-0.226728); break;
    case 1: asin = std::asin (-0.20908); break;
    case 2: asin = std::asin (-0.208772); break;
    default: asin = std::asin (-0.191412); break;
  }

  return asin;
}

//---------------------------------------------------------------------------------------------------------------------
template <class T, class DT>
T strip_hit_pair<T,DT>::barrel_strip_module_sin_tilt (const char& layer){
//---------------------------------------------------------------------------------------------------------------------

  switch (layer) {
    case 0: return -0.226728;
    case 1: return -0.20908;
    case 2: return -0.208772;
    default: return -0.191412;
  }
}

//---------------------------------------------------------------------------------------------------------------------
template <class T, class DT>
T strip_hit_pair<T,DT>::get_edge_theta (const std::vector<T>& edge, const T& perp){
//---------------------------------------------------------------------------------------------------------------------
  return edge.at(0) == 0 && edge.at(1) == 0 && edge.at(2) == 0 ? 0 : std::atan2 (perp, edge.at(2));
}

//---------------------------------------------------------------------------------------------------------------------
template <class T, class DT>
std::tuple<T, T> strip_hit_pair<T,DT>::get_edge_alphas (const T& dT){
//---------------------------------------------------------------------------------------------------------------------
  // it returns alpha values on the GLOBAL coordinate system
  // w.r.t. the r^hat direction
  T Delta_Phi = delta_phi (_hit_out, _hit_in);

  T sinDPhi = std::sin (Delta_Phi);
  T ratio = -9999;
  if (dT) ratio = std::sin (Delta_Phi) / dT;
  else {
    std::cout << "WARNING: strip_hit_pair::get_edge_alphas: Transverse distance between hits dT==0. Returning default -9999 alpha values." << std::endl;
    exit(0);
    return {-9999, -9999};
  }

  T Perp_hitIn = ratio * perp (_hit_in);
  T Perp_hitOut = ratio * perp (_hit_out);

  T rMax = std::max (Perp_hitIn, Perp_hitOut);

  if (std::abs (rMax) > 1.) {
    std::cout << "ERROR: StripHitPair::GetEdgeAlphas: Computing ASin for out of range value: r x sin(dphi)/dT = " << ratio*rMax
	            << "\t r " << rMax
              << "\t sin(dphi)/dT " << ratio
	            << "\t sin(dphi) " << sinDPhi
	            << "\t dT " << dT
	            << std::endl;
    exit (1);
  }

  return {std::asin (Perp_hitIn), std::asin (Perp_hitOut)};
}

//---------------------------------------------------------------------------------------------------------------------
template <class T, class DT>
std::tuple<T, T> strip_hit_pair<T,DT>::get_edge_alphas (const std::vector<T>& hit_in,
                                                        const std::vector<T>& hit_out, const T& dT){
//---------------------------------------------------------------------------------------------------------------------
  // it returns alpha values on the GLOBAL coordinate system
  // w.r.t. the r^hat direction
  T sin_Delta_Phi_dT = std::sin (delta_phi (hit_out, hit_in)) / dT;

  if (!dT) {
    std::cout << "WARNING: strip_hit_pair::get_edge_alphas: Transverse distance between hits dT==0. Returning default -9999 alpha values." << std::endl;
    exit(0);
    return {-9999, -9999};
  }

  T Perp_hitIn = sin_Delta_Phi_dT * perp (hit_in);
  T Perp_hitOut = sin_Delta_Phi_dT * perp (hit_out);
  if (Perp_hitIn > 1) Perp_hitIn = 1;
  if (Perp_hitOut > 1) Perp_hitOut = 1;
  if (Perp_hitIn < -1) Perp_hitIn = -1;
  if (Perp_hitOut < -1) Perp_hitOut = -1;

  T rMax = std::max (Perp_hitIn, Perp_hitOut);
  if (abs (rMax) > 1) {
    std::cout << "ERROR: strip_hit_pair::get_edge_alphas: Computing ASin for out of range value: r . sin (dphi) / dT = " << sin_Delta_Phi_dT << std::endl;
	  std::cout << "r " << std::setprecision (13) << rMax << std::endl;
    std::cout << "hit in" << std::setprecision (8) << hit_in.at(0) << " " << hit_in.at(1) << " " << hit_in.at(2) << std::endl;
    std::cout << "Perpin = " << sqrt (pow (hit_in.at(0), 2) + pow(hit_in.at(1), 2)) << std::endl;
    std::cout << "hit out" << std::setprecision (8) << hit_out.at(0) << " " << hit_out.at(1) << " " << hit_out.at(2) << std::endl;
    std::cout << "Perpout = " << sqrt (pow (hit_out.at(0), 2) + pow(hit_out.at(1), 2)) << std::endl;
	  std::cout << "dT " << dT << std::endl;
	  std::cout << std::endl;
    exit (1);
  }

  return {std::asin (Perp_hitIn), std::asin (Perp_hitOut)};
}

//---------------------------------------------------------------------------------------------------------------------
template <class T, class DT>
std::vector<T> strip_hit_pair<T,DT>::hit_local_position (const std::vector<T>& cl1, const std::vector<T>& cl2,
                                                         const T& alpha, const T& theta){
//---------------------------------------------------------------------------------------------------------------------
  // inputs (cl1, cl2, alpha, theta) have to be w.r.t.
  // the LOCAL coordinate system at the module,
  // with alpha = alpha_local!
  std::vector<T> hit {0};

  T interSensor = 6.42;
  T m1 = std::tan (-0.0259971);
  T y1_0 = cl1.at(1) - m1 * cl1.at(2);
  T y2_0 = cl2.at(1) + m1 * cl2.at(2);

// if cos alpha is close to 0 then the hit coordinates are big and the edge is excluded later in the GNN
  if (!std::cos (alpha)) {
    std::cout << "WARNING: strip_hit_pair::Hit_local_position: alpha_local == 90 degrees. Returning default hit position values." << std::endl;
    hit.push_back ((T) -9999);
    hit.push_back ((T) -9999);
  }
  else {
    T z1 = 0.5 * (y2_0 - y1_0 - interSensor * (std::sin (alpha) + m1 / std::tan (theta)) /  std::cos (alpha));
    T y1 = z1 + y1_0;
    z1 /= m1;
    hit.push_back (y1);
    hit.push_back (z1);
  }

  return hit;
}

//---------------------------------------------------------------------------------------------------------------------
template <class T, class DT>
std::vector<T> strip_hit_pair<T,DT>::hit_global_position (const std::vector<T>& module, const int& it, const T& tilt,
                                                          const T& alpha, const T& theta){
//  GetGlobPosBarrHit(module[0], cluster1[0], cluster2[0], sinTiltIn,  alpha0[0], theta0, 1);
//---------------------------------------------------------------------------------------------------------------------
  // input alpha = alpha_GLOBAL, w.r.t. the r^hat direction
  _TThits -> get (it);
  T phi = - Phi (module);
 
  std::vector<T> modRZ = rotate_z_axis (module, phi);
  std::vector<T> cl1 = rotate_z_axis (_TThits -> x_cluster1 (it), _TThits -> y_cluster1 (it), _TThits -> z_cluster1 (it), phi);
  std::vector<T> cl2 = rotate_z_axis (_TThits -> x_cluster2 (it), _TThits -> y_cluster2 (it), _TThits -> z_cluster2 (it), phi);
  std::vector<T> xhat = rotate_z_axis ((T) 1, 0, 0, tilt);

  T tilt_cl1 = delta_phi (xhat, cl1);
  T alpha_m_cl1 = alpha - tilt_cl1;

  std::transform (cl1.begin(), cl1.end(), modRZ.begin(), cl1.begin(), std::minus<T>());
  std::transform (cl2.begin(), cl2.end(), modRZ.begin(), cl2.begin(), std::minus<T>());

  cl1 = rotate_z_axis (cl1, -tilt);
  cl2 = rotate_z_axis (cl2, -tilt);

  // at this point cl1 and cl2 have been transformed to the module local coord. system

  // 1st iteration, using alpha_m_cl1
  std::vector<T> hit = hit_local_position (cl1, cl2, alpha_m_cl1, theta);
  hit = rotate_z_axis (hit, tilt);
  std::transform (hit.begin(), hit.end(), modRZ.begin(), hit.begin(), std::plus<T>());

  // 2nd iteration, using alpha_m_hit1, already enough to get the best possible result

  T tilt_hit1 = delta_phi (xhat, hit);
  T alpha_m_hit1 = alpha - tilt_hit1;
  hit = hit_local_position (cl1, cl2, alpha_m_hit1, theta);
  hit = rotate_z_axis (hit, tilt);
  std::transform (hit.begin(), hit.end(), modRZ.begin(), hit.begin(), std::plus<T>());
  hit = rotate_z_axis (hit, -phi);

  return hit;
}
