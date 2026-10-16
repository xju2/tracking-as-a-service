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

#include "MMG/TTree_particles.hpp"


//-----------------------------------------
template <class T>
void TTree_particles<T>::allocate_memory ()
//-----------------------------------------
{
  _event_id = new std::vector<int>;
  _particle_ID = new std::vector<uint64_t>;
  _subevent = new std::vector<int>; //PU or HS
  _barcode = new std::vector<int>;
  _pdgID = new std::vector<int>;
  _eta = new std::vector<T>;
//  phi = new std::vector<T>;
  _pT = new std::vector<T>;
  _vx = new std::vector<T>;
  _vy = new std::vector<T>;
  _vz = new std::vector<T>;

  _size = 0;
  _position = -1;
}


//------------------------------------
template <class T>
TTree_particles<T>::TTree_particles ()
//------------------------------------
{
  allocate_memory();
}


//---------------------------------------------------------------
template <class T>
TTree_particles<T>::TTree_particles (const TTree_particles<T>& p)
//---------------------------------------------------------------
{
  allocate_memory ();
  if (p._size)
    { _size = p._size;
      _position = p._position;

      *_event_id = *p._event_id;
      *_particle_ID = *p._particle_ID;
      *_subevent = *p._subevent;
      *_barcode = *p._barcode;
      *_pdgID = *p._pdgID;
      *_eta = *p._eta;
//  *phi = *p.phi;
      *_pT = *p._pT;
      *_vx = *p._vx;
      *_vy = *p._vy;
      *_vz = *p._vz;

      _particleID_TTreeParticles_map = p._particleID_TTreeParticles_map;
    }
}



//---------------------------------------------------------
template <class T>
TTree_particles<T>::TTree_particles(const TTree_dump& dump)
//---------------------------------------------------------
{
  allocate_memory();
  if(dump.nPartEVT){
    //uint width=(dump.event_number==15 || dump.event_number==140)? 8 : 7;
    for(int i(0); i<dump.nPartEVT; ++i){
      
      // Make PID from barcode and subevent (as in acorn)
      uint64_t particle_id = make_particle_id(dump.subevent[i],dump.barcode[i],7);

      (*_event_id).push_back(dump.event_number);
      (*_particle_ID).push_back (particle_id);
      (*_subevent).push_back (dump.subevent[i]);
      (*_barcode).push_back (dump.barcode[i]);
      (*_pdgID).push_back (dump.pdg_id[i]);
      (*_eta).push_back (dump.eta[i]);
      (*_pT).push_back (dump.pt[i]);
      (*_vx).push_back (dump.vx[i]);
      (*_vy).push_back (dump.vy[i]);
      (*_vz).push_back (dump.vz[i]);

      _particleID_TTreeParticles_map.insert (std::pair<uint64_t, int> (particle_id, _size));
      _size++;
    }
  }
}


//-------------------------------------
template <class T>
TTree_particles<T>::~TTree_particles ()
//-------------------------------------
{
  _size = 0;
  _position = -1;
  delete _event_id;
  delete _particle_ID;
  delete _subevent;
  delete _barcode;
  delete _pdgID;
  delete _eta;
//  delete phi;
  delete _pT;
  delete _vx;
  delete _vy;
  delete _vz;
}


//------------------------------------------------------------------------------
template <class T>
TTree_particles<T>& TTree_particles<T>::operator = (const TTree_particles<T>& p)
//------------------------------------------------------------------------------
{
  assert (!_size || _size == p._size);
  assert (p._size);

  if (!(*_event_id).size()) allocate_memory ();
  _size = p._size;
  _position = p._position;

  *_event_id = *p._event_id;
  *_particle_ID = *p._particle_ID;
  *_subevent = *p._subevent;
  *_barcode = *p._barcode;
  *_pdgID = *p._pdgID;
  *_eta = *p._eta;
//  *phi = *p.phi;
  *_pT = *p._pT;
  *_vx = *p._vx;
  *_vy = *p._vy;
  *_vz = *p._vz;

  _particleID_TTreeParticles_map = p._particleID_TTreeParticles_map;

  return *this;
}


//-----------------------------------------------------------------------------
template <class Tf>
bool operator == (const TTree_particles<Tf>& p1, const TTree_particles<Tf>& p2)
//-----------------------------------------------------------------------------
{
  bool test = (p1._size >= 0);
  test *= (p1._size == p2._size);

  test *= (*p1._event_id == *p2._event_id);
  test *= (*p1._particle_ID == *p2._particle_ID);
  test *= (*p1._subevent == *p2._subevent);
  test *= (*p1._barcode == *p2._barcode);
  test *= (*p1._pdgID == *p2._pdgID);
  test *= (*p1._eta == *p2._eta);
  //if (test) test *= ((*p1.phi) == (*p2.phi));
  test *= (*p1._pT == *p2._pT);
  test *= (*p1._vx == *p2._vx);
  test *= (*p1._vy == *p2._vy);
  test *= (*p1._vz == *p2._vz);

  test *= (p1._particleID_TTreeParticles_map == p2._particleID_TTreeParticles_map);

  return test;
}


//-----------------------------------------------------------------------------
template <class Tf>
bool operator != (const TTree_particles<Tf>& p1, const TTree_particles<Tf>& p2)
//-----------------------------------------------------------------------------
{
  return !(p1 == p2);
}


//-------------------------------------------------------------------------------
template <class T>
TTree_particles<T>& TTree_particles<T>::operator += (const TTree_particles<T>& p)
//-------------------------------------------------------------------------------
{
  (*_event_id).push_back(p.event_ID());
  (*_particle_ID).push_back (p.particle_ID());
  (*_subevent).push_back (p.subevent());
  (*_barcode).push_back (p.barcode());
  (*_pdgID).push_back (p.pdgID());
  (*_eta).push_back (p.eta());
//  (*phi).push_back (p.phi());
  (*_pT).push_back (p.pT());
  (*_vx).push_back (p.vx());
  (*_vy).push_back (p.vy());
  (*_vz).push_back (p.vz());

  _particleID_TTreeParticles_map.insert (std::pair<uint64_t, int> (p.particle_ID(), _size));
  _size++;

  return *this;
}


//----------------------------------------
template <class T>
void TTree_particles<T>::get (int i) const
//----------------------------------------
{
  assert (i>=0 && i<_size);
  _position = i;
}


//------------------------------------------------------
template <class T>
void TTree_particles<T>::getID (const uint64_t& i) const
//------------------------------------------------------
{
  _position = -1;
  std::_Rb_tree_const_iterator<std::pair<const uint64_t,int> > ptr = _particleID_TTreeParticles_map.find(i);
  if (ptr != _particleID_TTreeParticles_map.end())
    { _position = ptr->second;
      assert (_position>=0 && _position<_size);
    }
}


//---------------------------------------------
template <class T>
particle<T> TTree_particles<T>::get_particle ()
//---------------------------------------------
{
  return particle<T> ((*_event_id)[_position], (*_particle_ID)[_position], (*_subevent)[_position], (*_barcode)[_position], (*_pdgID)[_position], (*_eta)[_position], (*_pT)[_position], (*_vx)[_position], (*_vy)[_position], (*_vz)[_position]);
}


//--------------------------------------------------------
template <class T>
int TTree_particles<T>::find_particle (const uint64_t& ID)
//--------------------------------------------------------
{
  int p = -1;
  p = _particleID_TTreeParticles_map.find (ID) -> second;
  std::cout << p << " / " << ID << std::endl;

  return p;
}

//----------------------------------------------------
template <class T>
void TTree_particles<T>::branch (TTree* treeParticles)
//----------------------------------------------------
{
  treeParticles -> Branch ("event_id", &_event_id);
  treeParticles -> Branch ("particle_id", &_particle_ID);
  treeParticles -> Branch ("subevent", &_subevent);
  treeParticles -> Branch ("barcode", &_barcode);
  treeParticles -> Branch ("pdgID", &_pdgID);
//  std::string vector_Tname = "vector<" + boost::typeindex::type_id<T>().pretty_name() + ">";
//  treeParticles -> Branch ("eta", vector_Tname.c_str(), &_eta);
//  treeParticles -> Branch ("phi", vector_Tname.c_str(), &phi);
  treeParticles -> Branch ("eta", &_eta);
  treeParticles -> Branch ("pT", &_pT);
  treeParticles -> Branch ("v_x", &_vx);
  treeParticles -> Branch ("v_y", &_vy);
  treeParticles -> Branch ("v_z", &_vz);
}


//----------------------------------------------------------------
template <class T>
void TTree_particles<T>::set_branch_address (TTree* treeParticles)
//----------------------------------------------------------------
{
  treeParticles -> SetBranchAddress ("event_id", &_event_id);
  treeParticles -> SetBranchAddress ("particle_id", &_particle_ID);
  treeParticles -> SetBranchAddress ("subevent", &_subevent);
  treeParticles -> SetBranchAddress ("barcode", &_barcode);
  treeParticles -> SetBranchAddress ("pdgID", &_pdgID);

  treeParticles -> SetBranchAddress ("eta", &_eta);
//  treeParticles -> SetBranchAddress ("phi", &phi);
  treeParticles -> SetBranchAddress ("pT", &_pT);
  treeParticles -> SetBranchAddress ("v_x", &_vx);
  treeParticles -> SetBranchAddress ("v_y", &_vy);
  treeParticles -> SetBranchAddress ("v_z", &_vz);
}


//-------------------------------------------------
template <class T>
void TTree_particles<T>::push_back (particle<T>& p)
//-------------------------------------------------
{
  (*_event_id).push_back(p.event_id());
  (*_particle_ID).push_back (p.particle_ID());
  (*_subevent).push_back (p.subevent());
  (*_barcode).push_back (p.barcode());
  (*_pdgID).push_back (p.pdgID());
  (*_eta).push_back (p.eta());
//  (*phi).push_back (p.phi());
  (*_pT).push_back (p.pT());
  (*_vx).push_back (p.v_x());
  (*_vy).push_back (p.v_y());
  (*_vz).push_back (p.v_z());

  _particleID_TTreeParticles_map.insert (std::pair<uint64_t, int> (p.particle_ID(), _size));
  _size++;
}


/*
//-----------------------------------------------------
template <class T>
void TTree_hits<T>::create_hits_multimap (hits<T>& hts)
//-----------------------------------------------------
{
  for (int i=0; i<hit_id->size(); i++)
    { hit<T> ht ((*hit_id)[i], (*x)[i], (*y)[i], (*z)[i], (*_particle_ID)[i], (*module_ID)[i]);
      hts += HitsMap.insert (std::pair<int long, hit<T>> (ht.module_ID(), ht));
    }
}
*/

//---------------------------------------------------------
template <class T>
void TTree_particles<T>::save (const std::string& filename)
//---------------------------------------------------------
{
  TFile* RootFile = TFile::Open (filename.c_str(), "RECREATE");
  TTree* treeEventParticles = new TTree ("TreeEventParticles", "Tree containing the particles of an event' features");

  // Set up the branches
  branch (treeEventParticles);
  treeEventParticles -> Fill();
  RootFile -> cd ();
  treeEventParticles -> Write ();
  RootFile -> Close (); // also deletes TTree if not already deleted
  std::cout << "particles root file written in " << filename << std::endl;
}


//---------------------------------------------------------
template <class T>
void TTree_particles<T>::read (const std::string& filename)
//---------------------------------------------------------
{
  TFile* RootFile = TFile::Open (filename.c_str(), "READ");
  if (!RootFile) throw std::invalid_argument ("Cannot open file " + filename);

  TTree* treeEventParticles = (TTree*) RootFile -> Get ("TreeEventParticles");
  set_branch_address (treeEventParticles);
  treeEventParticles -> GetEntry(0);

  RootFile -> Close ();

  _size = (*_particle_ID).size();

  for (int i=0; i<_size; i++)
    _particleID_TTreeParticles_map.insert (std::pair<uint64_t, int> ((*_particle_ID)[i], i));
}

//template class TTree_particles<double>;