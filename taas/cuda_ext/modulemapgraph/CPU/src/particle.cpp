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

#include "MMG/particle.hpp"

/*!
  \brief Constructor that analyzes csv text and initializes object data
  \param cs_v_content string containing object data in csv format
  \n \n
  Analyzes csv string, extracts values from the stream and stores them in the object variables
*/

//-----------------------------------------------------------------------------------------------------------
template <class T>
particle<T>::particle (const std::map<std::string,int>& index, const std::string& csv_content, const int& ID)
//-----------------------------------------------------------------------------------------------------------
{
  std::stringstream ss (csv_content);
  std::vector<std::string> column_id;
  std::string line;
  char delim = ',';
  for (;std::getline (ss, line, delim);)
      column_id.push_back (line);

  std::istringstream (column_id[index.find("particle_id")->second]) >> _particle_ID;
  std::istringstream (column_id[index.find("subevent")->second]) >> _subevent;
  std::istringstream (column_id[index.find("barcode")->second]) >> _barcode;
//  std::istringstream (column_id[3]) >> _p_x;
//  std::istringstream (column_id[4]) >> _p_y;
//  std::istringstream (column_id[5]) >> _p_z;

  std::vector<int> _N_SP; //on different modules, e.g split SPs = 1
  std::vector<float> _phi;
  std::istringstream (column_id[index.find("pt")->second]) >> _pT;
  std::istringstream (column_id[index.find("eta")->second]) >> _eta;
  std::istringstream (column_id[index.find("vx")->second]) >> _v_x;
  std::istringstream (column_id[index.find("vy")->second]) >> _v_y;
  std::istringstream (column_id[index.find("vz")->second]) >> _v_z;
  std::istringstream (column_id[index.find("pdgId")->second]) >> _pdgID;
  _event_id = ID;
}


//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <class T>
particle<T>::particle (const int& event_id_, const uint64_t& particle_ID_, const int& subevent_, const int& barcode_, const int& pdgID_, const T& eta_, const T& pT_, const T& v_x_, const T& v_y_, const T& v_z_)
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
{
  _event_id = event_id_;
  _particle_ID = particle_ID_;
  _subevent = subevent_;
  _barcode = barcode_;
//  _p_x= p_x_;
//  _p_y = p_y_;
//  _p_z = p_z_;
  _pdgID = pdgID_;
  _eta = eta_;
//  _phi = phi_;
  _pT = pT_;
  _v_x = v_x_;
  _v_y = v_y_;
  _v_z = v_z_;
}


//---------------------------------------------------------------
template <class Tf>
bool operator == (const particle<Tf>& p1, const particle<Tf>& p2)
//---------------------------------------------------------------
{
  bool test = true;

  test *= (p1._event_id == p2._event_id);
  if (test) test *= (p1._particle_ID == p2._particle_ID);
  if (test) test *= (p1._subevent == p2._subevent);
  if (test) test *= (p1._barcode == p2._barcode);
//  if (test) test *= (p1._p_x == p2._p_x) && (p1._p_y == p2._p_y) && (p1._p_z == p2._p_z);
  if (test) test *= (p1._N_SP == p2._N_SP);
  if (test) test *= (p1._pdgID == p2._pdgID);
  if (test) test *= (p1._eta == p2._eta);// && (p1._phi == p2._phi);
  if (test) test *= (p1._pT == p2._pT);
  if (test) test *= (p1._v_x == p2._v_x) && (p1._v_y == p2._v_y) && (p1._v_z == p2._v_z);

  return test;
}


//---------------------------------------------------------------
template <class Tf>
bool operator != (const particle<Tf>& p1, const particle<Tf>& p2)
//---------------------------------------------------------------
{
  return !(p1 == p2);
}


/*!
  \brief Output stream for particle
  \param ostream s output stream
  \param particle p single particle
  \n \n
  overloads operator << for particle (send to ostream particle data informations)
*/

//----------------------------------------------------------------
template <class Tf>
std::ostream& operator << (std::ostream& s, const particle<Tf>& p)
//----------------------------------------------------------------
{
  s << "event ID = " << p._event_id;
  s << "particle ID = " << p._particle_ID << std::endl;
  s << "_subevent = " << p._subevent << std::endl;
  s << "_barcode = " << p._barcode << std::endl;
  s << "px = " << p._p_x << std::endl;
  s << "py = " << p._p_y << std::endl;
  s << "pz = " << p._p_z << std::endl;
//    int _N_SP; //on different modules, e.g split SPs = 1
  s << "pgdID = " << p._pdgID << std::endl;
//    float _eta;
//    float _phi;
  s << "_pT = " << p._pT << std::endl;
  s << "_eta = " << p._eta << std::endl;
  s << "vx = " << p._v_x << std::endl;
  s << "vy = " << p._v_y << std::endl;
  s << "vz = " << p._v_z << std::endl;
  s << std::endl;

  return s;
}


//----------------------------------------------
template <class T>
void particle<T>::save_csv (std::ofstream& file)
//----------------------------------------------
{
  std::string delim = ",";
  file << _event_id << delim << _particle_ID << delim << _subevent << delim << _barcode << delim << // _p_x << delim << _p_y << delim << _p_z << delim << 
  _N_SP << delim << _pdgID << delim << _eta << delim << _pT << delim << _v_x << delim << _v_y << delim << _v_z << std::endl;
//  _N_SP << delim << _pdgID << delim << _eta << delim << _phi << delim << _pT << delim << _v_x << delim << _v_y << delim << _v_z << std::endl;
   // << delim << _cut_z0 << delim << _cut_dphi << delim << _cut_phiSlope << delim << _cut_deta = cut_deta_ << delim << _cut_diff_dzdr = cut_diff_dzdr_ << delim << _cut_diff_dxdy;
}


/*!
  \brief Read particle data from csv file
  \param file ifstream name
  \n \n
  Gets line from file and converts csv, extracts and stores data in particle object
*/

//-----------------------------------------------------------------------------------------------------
template <class T>
bool particle<T>::read_csv (const std::map<std::string,int>& index, std::ifstream& file, const int& ID)
//-----------------------------------------------------------------------------------------------------
{
  std::string content;
  file >> content;

  if (file.eof()) return true;
  (*this) = particle (index, content, ID);

  return false; // not the eof
}
