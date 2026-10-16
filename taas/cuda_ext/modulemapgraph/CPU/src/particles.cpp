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

#include "MMG/particles.hpp"

//------------------------
template <class T>
particles<T>::particles ()
//------------------------
{
  _size = 0;
  _event_id = -1;
}


//-----------------------------------------------
template <class T>
particles<T>::particles (TTree_particles<T>& TTp)
//-----------------------------------------------
{
  assert (TTp.size());
  particle<T> p;

  for (int i=0; i<TTp.size(); i++)
    { TTp.get(i);
      uint64_t module = TTp.particle_ID();
      p = particle<T> (TTp.event_ID(), TTp.particle_ID(), TTp.subevent(), TTp.barcode(), TTp.pdgID(), TTp.eta(), TTp.pT(), TTp.vx(), TTp.vy(), TTp.vz());
///      _HitsMap.insert (std::pair<uint64_t, hit<T>> (ht.hit_id(), ht));
       _ParticlesMap.insert (std::pair<uint64_t, particle<T>> (p.particle_ID(), p));
//      _moduleID_TTreeHits_map.insert (std::pair<uint64_t, int> (module, i));
//      _particleID_TTreeHits_map.insert (std::pair<uint64_t, int> (TTh.particle_ID(), i));
    }
  _size = TTp.size();
}


//------------------------------------------
template <class T>
particles<T>::operator TTree_particles<T> ()
//------------------------------------------
{
  TTree_particles<T> TTp;

  for (std::_Rb_tree_iterator <std::pair <const uint64_t, particle<T> > > it=_ParticlesMap.begin(); it!=_ParticlesMap.end(); ++it)
    TTp.push_back (it->second);

  return TTp;
}


//------------------------------------------------------------
template <class T>
particles<T>& particles<T>::operator = (const particles<T>& p)
//------------------------------------------------------------
{
  _size = p._size;
  _event_id = p._event_id;
  _ParticlesMap = p._ParticlesMap;

  return *this;
}


//------------------------------------------------------------
template <class T>
particles<T>& particles<T>::operator += (const particle<T>& p)
//------------------------------------------------------------
{
  _size++;
  _ParticlesMap.insert (std::pair<int long, particle<T>> (p.particle_ID(), p));

  return *this;
}


//-----------------------------------------------------------------
template <class Tf>
bool operator == (const particles<Tf>& p1, const particles<Tf>& p2)
//-----------------------------------------------------------------
{
  bool test = (p1._size == p2._size);
//  test *= (p1._event_id == p2._event_id);

/*
  if (test)
    { std::_Rb_tree_const_iterator <std::pair <const uint64_t, particle<Tf> > > it_p2 = p2._ParticlesMap.begin();
      for (std::_Rb_tree_const_iterator <std::pair <const uint64_t, particle<Tf> > > it_p1=p1._ParticlesMap.begin(); it_p1!=p1._ParticlesMap.end(); ++it_p1, ++it_p2)
        { test *= (it_p1->first == it_p2->first);
          // test *= (it_p1->second == it_p2->second);
        }
    }
*/

test = true;
  return test;
}


//-----------------------------------------------------------------
template <class Tf>
bool operator != (const particles<Tf>& p1, const particles<Tf>& p2)
//-----------------------------------------------------------------
{
  return !(p1 == p2);
}


//-----------------------------------------------------------------
template <class Tf>
std::ostream& operator << (std::ostream& s, const particles<Tf>& p)
//-----------------------------------------------------------------
{
  s << "event ID = " << p._event_id << std::endl;

  s << "particle ID = ";
  for (int i=0; i<(*p.particle_id).size(); i++)
    s << (*p.particle_id)[i] << " ";
  std::cout << std::endl;

/*
  s << "particle ID = " << p.particle_id[0] << endl;
  s << "subevent = " << p.subevent[1] << endl;
  s << "barcode = " << p.barcode[1] << endl;
*/
/*
    std::vector<int> N_SP; //on different modules, e.g split SPs = 1
    std::vector<int> pdgID;
    std::vector<float> eta;
    std::vector<float> phi;
    std::vector<float> pT;
    std::vector<float> v_x;
    std::vector<float> v_y;
    std::vector<float> v_z;

    std::vector<pair<double,double>> cut_z0; //two entries per index: 12 and 23
    std::vector<pair<double,double>> cut_dphi;
    std::vector<pair<double,double>> cut_phiSlope;
    std::vector<pair<double,double>> cut_deta;

    std::vector<double> cut_diff_dzdr;
    std::vector<double> cut_diff_dxdy;
*/

  return s;
}


//-------------------------------------------------------
template <class T>
void particles<T>::save_csv (const std::string& filename)
//-------------------------------------------------------
{
  std::ofstream file (filename);
  if (file.fail()) throw std::invalid_argument ("Cannot open file " + filename);

  std::string content = "particle_id,subevent,barcode,px,py,pz,pt,eta,vx,vy,vz,radius,status,charge,pdgId,pass,vProdNIn,vProdNOut,vProdStatus,vProdBarcode,nClusters";
  file << content;

  for (std::_Rb_tree_iterator<std::pair<const long unsigned int, particle<T> > > it = _ParticlesMap.begin(); it != _ParticlesMap.end(); it++)
    (it->second).save_csv(file);

  file.close ();
}


//--------------------------------------------------------------------------------
template <class T>
void particles<T>::read_csv (const std::string& path, const std::string& filename)
//--------------------------------------------------------------------------------
{
  // get event_id from filename
  std::string event_id = extract_event_id_from_filename(filename);
  std::stringstream ID;
  ID << event_id;
  ID >> _event_id;

  std::ifstream file (path+filename);
  if (file.fail()) throw std::invalid_argument ("Cannot open file " + path + filename);

  // delete first line = comments
  std::string content;
  file >> content;

  // store index names
  std::stringstream ssi (content);
  std::vector<std::string> column_id;
  std::string line;
  char delim = ',';
  std::map<std::string,int> index;
  for (int i=0; getline (ssi, line, delim); i++)
      index[line] = i;

  for (bool eof = false; !eof;)
    { particle<T> p;
      eof = p.read_csv (index, file, _event_id);
//      p.event_id() = ;
      if (!eof)
        { _ParticlesMap.insert (std::pair<int long, particle<T>> (p.particle_ID(), p));
          _size++;
        }
    }

  file.close();
}

/*!
  \brief Reads csv file in one shot
  \param filename csv filename
  \n \n
  read the whole csv file as text and send each line to \ref particle::particle(string) contructor
*/

//-------------------------------------------------------------------------------------
template <class T>
void particles<T>::read_csv_once (const std::string& path, const std::string& filename)
//-------------------------------------------------------------------------------------
{

//  std::string event_id = boost::regex_replace (hits_filename, boost::regex("[^0-9]*([0-9]+).*"), std::string("\\1"));
//  if (event_id != boost::regex_replace (particles_filename, boost::regex("[^0-9]*([0-9]+).*"), std::string("\\1")))
//    throw std::invalid_argument ("hits and particules file are not part of the same event");
//  std::stringstream ID;
//  ID << event_id;
//  int eventID = 0;
//  ID >> eventID;

  // get event_id from filename
  std::string event_id = boost::regex_replace (filename, boost::regex("[^0-9]*([0-9]+).*"), std::string("\\1"));
  std::cout << filename << std::endl;
  std::cout << event_id << std::endl;
  std::stringstream ID;
  ID << event_id;
  ID >> _event_id;
  std::cout << "EVENTID = " << _event_id << std::endl;

  std::ostringstream ss;
  std::ifstream file (path+filename);
  if (file.fail()) throw std::invalid_argument ("Cannot open file " + path + filename);

  // block reading
  ss << file.rdbuf();
  file.close ();

  std::istringstream sstream (ss.str());
  std::string content;
  // delete first line = comments
  getline (sstream, content);

  // store index names
  std::stringstream ssi (content);
  std::vector<std::string> column_id;
  std::string line;
  char delim = ',';
  std::map<std::string,int> index;
  for (int i=0;getline (ssi, line, delim);i++)
    index[line] = i;

  for (;getline (sstream, content);)
    { particle<T> p (index, content, _event_id);
      _ParticlesMap.insert (std::pair<int long, particle<T>> (p.particle_ID(), p));
      _size++;
    }
}
