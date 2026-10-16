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

#include "MMG/hit.hpp"

//--------------------------------------------------------------------------------------------
template <class T>
hit<T>::hit (const std::map<std::string,int>& index, std::string csv_content, bool extra_feat)
//--------------------------------------------------------------------------------------------
{
  _extra_features = extra_feat;
  std::string ID_name = (index.find("ID") != index.end()) ? "ID" : "module_id";
  std::stringstream ss (csv_content);
  std::vector<std::string> column_id;
  std::string line;
  char delim = ',';
  for (;getline (ss, line, delim);)
      column_id.push_back (line);

  std::istringstream (column_id[index.find("hit_id")->second]) >> _hit_id;
  std::istringstream (column_id[index.find("x")->second]) >> _x;
  std::istringstream (column_id[index.find("y")->second]) >> _y;
  std::istringstream (column_id[index.find("z")->second]) >> _z;
  std::istringstream (column_id[index.find("particle_id")->second]) >> _particle_id;
  std::istringstream (column_id[index.find(ID_name.c_str())->second]) >> _ID;
  std::istringstream (column_id[index.find("hardware")->second]) >> _hardware;
  std::istringstream (column_id[index.find("barrel_endcap")->second]) >> _barrel_endcap;
  std::istringstream (column_id[index.find("particle_id_1")->second]) >> _particle_ID1;
  std::istringstream (column_id[index.find("particle_id_2")->second]) >> _particle_ID2;

  if (_extra_features) {
    std::istringstream (column_id[index.find("layer_disk")->second]) >> _layer_disk;
    std::istringstream (column_id[index.find("eta_module")->second]) >> _eta_module;
    std::istringstream (column_id[index.find("phi_module")->second]) >> _phi_module;
    std::istringstream (column_id[index.find("cluster_x_1")->second]) >> _cluster1_x;
    std::istringstream (column_id[index.find("cluster_y_1")->second]) >> _cluster1_y;
    std::istringstream (column_id[index.find("cluster_z_1")->second]) >> _cluster1_z;
    std::istringstream (column_id[index.find("cluster_x_2")->second]) >> _cluster2_x;
    std::istringstream (column_id[index.find("cluster_y_2")->second]) >> _cluster2_y;
    std::istringstream (column_id[index.find("cluster_z_2")->second]) >> _cluster2_z;
  }
}


//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <class T>
hit<T>::hit (const uint64_t& hit_id_, const T& x_, const T& y_, const T& z_, const uint64_t& particle_id_, const uint64_t& module_ID_, const std::string& hardware_, const int& barrel_endcap_, const uint64_t& particle_ID1_, const uint64_t& particle_ID2_)
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
{
  _extra_features = false;
  _hit_id = hit_id_;
  _x = x_;
  _y = y_;
  _z = z_;
  _particle_id = particle_id_;
  _ID = module_ID_;
  _hardware = hardware_;
  _barrel_endcap = barrel_endcap_;
  _particle_ID1 = particle_ID1_;
  _particle_ID2 = particle_ID2_;
}


//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <class T>
hit<T>::hit(const uint64_t& hit_id_, const T& x_, const T& y_, const T& z_, const uint64_t& particle_id_, const uint64_t& module_ID_, const std::string& hardware_, const int& barrel_endcap_, const uint64_t& particle_ID1_, const uint64_t& particle_ID2_,
  const int& layer_disk_, const int& eta_module_, const int& phi_module_, const T& cluster1_x_, const T& cluster1_y_, const T& cluster1_z_, const T& cluster2_x_, const T& cluster2_y_, const T& cluster2_z_)
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
{
  _extra_features = true;
  _hit_id = hit_id_;
  _x = x_;
  _y = y_;
  _z = z_;
//  _vx = vx_;
//  _vy = vy_;
//  _vz = vz_;
  _particle_id = particle_id_;
  _ID = module_ID_;
  _hardware = hardware_;
  _barrel_endcap = barrel_endcap_;
  _particle_ID1 = particle_ID1_;
  _particle_ID2 = particle_ID2_;
  _layer_disk = layer_disk_;
  _eta_module = eta_module_;
  _phi_module = phi_module_;
  _cluster1_x = cluster1_x_;
  _cluster1_y = cluster1_y_;
  _cluster1_z = cluster1_z_;
  _cluster2_x = cluster2_x_;
  _cluster2_y = cluster2_y_;
  _cluster2_z = cluster2_z_;
//  _eta_particle = eta_particle_;
//  _particle_pdgId = particle_pdgId_;
}


//-------------------------------------------------------
template <class Tf>
bool operator == (const hit<Tf>& ht1, const hit<Tf>& ht2)
//-------------------------------------------------------
{
  bool test = ((ht1._hit_id) == (ht2._hit_id));
  if (test) test *= (ht1._particle_id == ht2._particle_id);
  if (test) test *= (ht1._x == ht2._x);
  if (test) test *= (ht1._y == ht2._y);
  if (test) test *= (ht1._z == ht2._z);
  if (test) test *= (ht1._ID == ht2._ID);
  if (test) test *= (ht1._hardware == ht2._hardware);
  if (test) test *= (ht1._barrel_endcap == ht2._barrel_endcap);
  if (test) test *= (ht1._particle_ID1 == ht2._particle_ID1);
  if (test) test *= (ht1._particle_ID2 == ht2._particle_ID2);

  if (ht1._extra_features) {
    if (! ht2._extra_features) test = false;
    if (test) test *= (ht1._layer_disk == ht2._layer_disk);
    if (test) test *= (ht1._eta_module == ht2._eta_module);
    if (test) test *= (ht1._phi_module == ht2._phi_module);
    if (test) test *= (ht1._cluster1_x == ht2._cluster1_x);
    if (test) test *= (ht1._cluster1_y == ht2._cluster1_y);
    if (test) test *= (ht1._cluster1_z == ht2._cluster1_z);
    if (test) test *= (ht1._cluster2_x == ht2._cluster2_x);
    if (test) test *= (ht1._cluster2_y == ht2._cluster2_y);
    if (test) test *= (ht1._cluster2_z == ht2._cluster2_z);
  }

  return test;
}


//-------------------------------------------------------
template <class Tf>
bool operator != (const hit<Tf>& ht1, const hit<Tf>& ht2)
//-------------------------------------------------------
{
  return !(ht1 == ht2);
}


//-----------------------------------------
template <class T>
void hit<T>::save_csv (std::ofstream& file)
//-----------------------------------------
{
  std::string delim = ",";
  file << _hit_id << delim << _x << delim << _y << delim << _z << delim << _particle_id << delim << _ID << delim << _hardware << delim << _barrel_endcap << delim << _particle_ID1 << delim << _particle_ID2;
  if (_extra_features)
    file << delim << _layer_disk << delim << _eta_module << delim << _phi_module << delim << _cluster1_x << delim << _cluster1_y << delim << _cluster1_z << delim << _cluster2_x << delim << _cluster2_y << delim << _cluster2_z;
  file << std::endl;
}


//---------------------------------------------------------------------------------
template <class T>
bool hit<T>::read_csv (const std::map<std::string,int>& index, std::ifstream& file)
//---------------------------------------------------------------------------------
{
  //get event_id from filename
  std::string content;
  file >> content;

  if (file.eof()) return true;
  (*this) = hit (index, content, _extra_features);

  return false; // not the eof
}
