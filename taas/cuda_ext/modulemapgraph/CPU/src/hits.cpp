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

#include "MMG/hits.hpp"


//----------------------------------------------
template <class T>
hits<T>::hits (bool true_graph, bool extra_feat)
//----------------------------------------------
{
  _size = 0;
  _true_graph = true_graph;
  _extra_features = extra_feat;
//  _event_id = -1;
}


//--------------------------------
template <class T>
hits<T>::hits (TTree_hits<T>& TTh)
//--------------------------------
{
  assert (TTh.size());
  _true_graph = TTh.true_graph();
  _extra_features = TTh.extra_features();
  hit<T> ht;

  for (int i=0; i<TTh.size(); i++)
    { TTh.get(i);
      uint64_t module = TTh.module_ID();
      if (_extra_features == TTh.extra_features())
        ht = hit<T> (TTh.hit_id(), TTh.x(), TTh.y(), TTh.z(), TTh.particle_ID(), TTh.module_ID(), TTh.hardware(), TTh.barrel_endcap(), TTh.particle_ID1(), TTh.particle_ID2(),
                   TTh.layer_disk(), TTh.eta_module(), TTh.phi_module(), TTh.cluster1_x(), TTh.cluster1_y(), TTh.cluster1_z(), TTh.cluster2_x(), TTh.cluster2_y(), TTh.cluster2_z());
      else
        ht = hit<T> (TTh.hit_id(), TTh.x(), TTh.y(), TTh.z(), TTh.particle_ID(), TTh.module_ID(), TTh.hardware(), TTh.barrel_endcap(), TTh.particle_ID1(), TTh.particle_ID2());
///      _HitsMap.insert (std::pair<uint64_t, hit<T>> (ht.hit_id(), ht));
      _HitsMap.insert (std::pair<uint64_t, hit<T>> (module, ht));
//      _moduleID_TTreeHits_map.insert (std::pair<uint64_t, int> (module, i));
//      _particleID_TTreeHits_map.insert (std::pair<uint64_t, int> (TTh.particle_ID(), i));
    }
  _size = TTh.size();
}


//--------------------------------
template <class T>
hits<T>::operator TTree_hits<T> ()
//--------------------------------
{
  TTree_hits<T> TTh (_true_graph, _extra_features);

//  std::cout << "cast conversion in hits" << std::endl;
  for (std::_Rb_tree_iterator <std::pair <const uint64_t, hit<T> > > it=_HitsMap.begin(); it!=_HitsMap.end(); ++it)
    TTh.push_back (it->second);

  return TTh;
}


//----------------------------------------------
template <class T>
hits<T>& hits<T>::operator = (const hits<T>& ht)
//----------------------------------------------
{
  _size = ht._size;
  std::cout << "hits.cpp op = " << std::endl;
//  _event_id = ht._event_id;
  _HitsMap = ht._HitsMap;
//  _moduleID_TTreeHits_map = ht._moduleID_TTreeHits_map;
//  _particleID_TTreeHits_map = ht._particleID_TTreeHits_map;

  return *this;
}


//----------------------------------------------
template <class T>
hits<T>& hits<T>::operator += (const hit<T>& ht)
//----------------------------------------------
{
  _size++;
//  _TTreeHits.push_back (ht);
///  _HitsMap.insert (std::pair<uint64_t, hit<T>> (ht.hit_id(), ht));
  _HitsMap.insert (std::pair<uint64_t, hit<T>> (ht.module_ID(), ht));
//  _moduleID_TTreeHits_map.insert (std::pair<uint64_t, int> (ht.module_ID(), _size));
//  _particleID_TTreeHits_map.insert (std::pair<uint64_t, int> (ht.particle_ID(), _size));

  return *this;
}


//---------------------------------------------------------
template <class Tf>
bool operator == (const hits<Tf>& ht1, const hits<Tf>& ht2)
//---------------------------------------------------------
{
  bool test = (ht1._size == ht2._size);
//  test *= (ht1._event_id == ht2._event_id);
//  test *= (ht1._TTreeHits == ht2._TTreeHits);

  if (test)
    { std::_Rb_tree_const_iterator <std::pair <const uint64_t, hit<Tf> > > it_ht2 = ht2._HitsMap.begin();
      for (std::_Rb_tree_const_iterator <std::pair <const uint64_t, hit<Tf> > > it_ht1=ht1._HitsMap.begin(); it_ht1!=ht1._HitsMap.end(); ++it_ht1, ++it_ht2)
        { test *= (it_ht1->first == it_ht2->first);
          test *= (it_ht1->second == it_ht2->second);
        }
    }

  return test;
}


//---------------------------------------------------------
template <class Tf>
bool operator != (const hits<Tf>& ht1, const hits<Tf>& ht2)
//---------------------------------------------------------
{
  return !(ht1 == ht2);
}


//-------------------------------------------------------------
template <class Tf>
std::ostream& operator << (std::ostream& s, const hits<Tf>& ht)
//-------------------------------------------------------------
{
  s << "event ID = " << ht._event_id << std::endl;

  s << "event ID = ";
  for (int i=0; i<(*ht.particle_id).size(); i++)
    s << (*ht.particle_id)[i] << " ";
  std::cout << std::endl;

/*
  s << "particle ID = " << p.particle_id[0] << endl;
  s << "subevent = " << p.subevent[1] << endl;
  s << "barcode = " << p.barcode[1] << endl;
*/
/*
    vector<int> N_SP; //on different modules, e.g split SPs = 1
    vector<int> pdgID;
    vector<float> eta;
    vector<float> phi;
    vector<float> pT;
    vector<float> v_x;
    vector<float> v_y;
    vector<float> v_z;

    vector<pair<double,double>> cut_z0; //two entries per index: 12 and 23
    vector<pair<double,double>> cut_dphi;
    vector<pair<double,double>> cut_phiSlope;
    vector<pair<double,double>> cut_deta;

    vector<double> cut_diff_dzdr;
    vector<double> cut_diff_dxdy;
*/

  return s;
}


//---------------------------------------------------
template <class T>
void hits<T>::save_root (const std::string& filename)
//---------------------------------------------------
{
//  TFile* RootFile = TFile::Open (filename.c_str(), "RECREATE");
//  TTree* treeEventHits = new TTree ("TreeEventHits", "Tree containing the hits of an event' features");
//  treeEventHits -> Branch ("size", &_size, "_size/i");

  std::cout << "hits size = " << _size << std::endl;
  uint64_t hit_id[_size], particle_id[_size], module_ID[_size];
  T x[_size], y[_size], z[_size];

  //for (std::_Rb_tree_iterator <std::pair <const uint64_t, hit<T> > > it=hts.begin(); it!=hts.end(); ++it)

//  std::_Rb_tree_iterator <std::pair <const uint64_t, hit<T> > > it;

  for (int i=0; i<_size; i++)
    { //hit<T> ht = it->second;
//      hit_id[i] = ht.hit_id();
//      particle_id[i] = ht.particle_id();
//      x[i] = ht.x();//it->second.x();
//      y[i] = ht.y();
//      z[i] = ht.z();
//      module_ID[i] = ht.module_ID();
    }

//  treeEventHits -> Branch ("hit_id", &hit_id, "hit_id[_size]/l");
//  treeEventHits -> Branch ("particle_id", &particle_id, "particle_id[_size]/l");
//  treeEventHits -> Branch ("particle_id", &x, "x[_size]/F");
//  treeEventHits -> Branch ("particle_id", &y, "y[_size]/F");
//  treeEventHits -> Branch ("particle_id", &z, "z[_size]/F");

/*
  fTreeBMu->Branch("fNMu", &fNMu, "fNMu/I");
  fTreeBMu->Branch("fEMuMC", fEMuMC, "fEMuMC[fNMu]/F");
*/

//  std::string vector_Tname = "vector<" + boost::typeindex::type_id<T>().pretty_name() + ">";
//  treeHits -> Branch ("x", vector_Tname.c_str(), &x);
//  treeHits -> Branch ("y", vector_Tname.c_str(), &y);
//  treeHits -> Branch ("z", vector_Tname.c_str(), &z);
//  treeHits -> Branch ("ID", "std::vector<uint64_t>", &module_ID);
//  treeEventHits -> Fill();
//  RootFile -> cd ();
//  treeEventHits -> Write ();
 
//  RootFile -> Close ();
}


//---------------------------------------------------
template <class T>
void hits<T>::read_root (const std::string& filename)
//---------------------------------------------------
{
//  _TTreeHits.read(filename);
//
//  for (int i=0; i<_TTreeHits.size(); i++)
//    { _TTreeHits.get (i);
//      _moduleID_TTreeHits_map.insert (std::pair<uint64_t, int> (_TTreeHits.module_ID(), i));
//      _particleID_TTreeHits_map.insert (std::pair<uint64_t, int> (_TTreeHits.particle_id(), i));
//    }
}


//--------------------------------------------------
template <class T>
void hits<T>::save_csv (const std::string& filename)
//--------------------------------------------------
{
  std::ofstream file (filename);
  if (file.fail()) throw std::invalid_argument ("Cannot open file " + filename);

  std::string content = "hit_id,x,y,z,particle_id,ID \n";
  file << content;

  for (std::_Rb_tree_iterator<std::pair<const uint64_t, hit<T> > > it = _HitsMap.begin(); it != _HitsMap.end(); it++)
    (it->second).save_csv(file);

  file.close ();
}


//---------------------------------------------------------------------------
template <class T>
void hits<T>::read_csv (const std::string& path, const std::string& filename)
//---------------------------------------------------------------------------
{
  std::ifstream file (path + filename);
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

  for (bool eof = false; !eof;){
    hit<T> ht (_extra_features);
    eof = ht.read_csv (index, file);
    _HitsMap.insert (std::pair<uint64_t, hit<T>> (ht.module_ID(), ht));
///        { _HitsMap.insert (std::pair<uint64_t, hit<T>> (ht.hit_id(), ht));
//          _TTreeHits.push_back (ht);
    _size++;
//          _moduleID_TTreeHits_map.insert (std::pair<uint64_t, int> (_TTreeHits.module_ID(), _size));
//          _particleID_TTreeHits_map.insert (std::pair<uint64_t, int> (_TTreeHits.particle_ID(), _size));
  }

  file.close();
}

/*!
  \brief Reads csv file in one shot
  \param filename csv filename
  \n \n
  read the whole csv file as text and send each line to \ref particle::particle(string) contructor
*/

//--------------------------------------------------------------------------------
template <class T>
void hits<T>::read_csv_once (const std::string& path, const std::string& filename)
//--------------------------------------------------------------------------------
{
  std::ostringstream ss;
  std::ifstream file (path + filename);
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
  for (int i=0; getline(ssi, line, delim); i++)
    index[line] = i;

  for (;getline (sstream, content);){
    hit<T> ht (index, content, _extra_features);
    _HitsMap.insert (std::pair<uint64_t, hit<T>> (ht.module_ID(), ht));
///    _HitsMap.insert (std::pair<uint64_t, hit<T>> (ht.hit_id(), ht));
    _size++;
  }
}

//--------------------------------------------------------------
template<class Tf>
int define_region (const hit<Tf>& source, const hit<Tf>& target)
//--------------------------------------------------------------
{
  // the two nodes are in the same region
  if ((source.hardware() == target.hardware()) && (source.barrel_endcap() == target.barrel_endcap()))
    { //same region
      if (source.hardware() == "PIXEL" && source.barrel_endcap() == -2) {return 1;}
      if (source.hardware() == "PIXEL" && source.barrel_endcap() == 0) {return 2;}
      if (source.hardware() == "PIXEL" && source.barrel_endcap() == 2) {return 3;}
      if (source.hardware() == "STRIP" && source.barrel_endcap() == -2) {return 4;}
      if (source.hardware() == "STRIP" && source.barrel_endcap() == 0) {return 5;}
      if (source.hardware() == "STRIP" && source.barrel_endcap() == 2) {return 6;}
    } 
    // the two nodes are not in the same region but on the same sub-detector
    else
      if ((source.hardware() == target.hardware()))
        { if (source.hardware() == "PIXEL")
            { if ( (source.barrel_endcap() == -2 && target.barrel_endcap() == 0) || (target.barrel_endcap() == -2 && source.barrel_endcap() == 0) ) {return 7;}
              if ( (source.barrel_endcap() == 2 && target.barrel_endcap() == 0) || (target.barrel_endcap() == 2 && source.barrel_endcap() == 0) ) {return 8;}
              return 0;
            }
          else
            if (source.hardware() == "STRIP")
              { if ((source.barrel_endcap() == -2 && target.barrel_endcap() == 0) || (target.barrel_endcap() == -2 && source.barrel_endcap() == 0)) {return 9;}
                if ( (source.barrel_endcap() == 2 && target.barrel_endcap() == 0) || (target.barrel_endcap() == 2 && source.barrel_endcap() == 0) ) {return 10;}
                return 0;
              }
          else {return 0;}
        }
    //not the same region neither the same sub-detector but the same barrel or endcaps
    else
      if (source.barrel_endcap() == target.barrel_endcap())
        { if (source.barrel_endcap() == 0) {return 11;}
          if (source.barrel_endcap() == 2) {return 12;}
          if (source.barrel_endcap() == -2) {return 13;}
          return 0;
        }
      else
        if ( (source.barrel_endcap() ==0 && source.hardware() == "PIXEL") || (target.barrel_endcap() ==0 && target.hardware() == "PIXEL"))
          { if ((source.barrel_endcap() ==2 && source.hardware() == "STRIP") || (target.barrel_endcap() ==2 && target.hardware() == "STRIP")) {return 14;}
            if ((source.barrel_endcap() ==-2 && source.hardware() == "STRIP") || (target.barrel_endcap() ==-2 && target.hardware() == "STRIP")) {return 15;}
          }
      else {return 0;}
     return 0;
}
