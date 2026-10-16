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

/*! \class particle
    \brief particle library \n

    \htmlonly 
    <FONT color="#838383">

    insert license
    </FONT>
    \endhtmlonly

    Particle data  \n
    \param event_id event id (extrated from filename)
    \param particle_ID unique tag to recognize a particle (combination of subevent and barcode)

    \authors copyright \htmlonly &#169; \endhtmlonly 2022 Christophe COLLARD \n
             copyright \htmlonly &#169; \endhtmlonly 2022 Charline Rougier \n
             copyright \htmlonly 2022 Centre National de la Recherche Scientifique \endhtmlonly \n
             copyright \htmlonly 2022 Universit&#233; Paul Sabatier, Toulouse 3 \endhtmlonly \n
             copyright \htmlonly &#169; 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007 Laboratoire des 2 infinis de Toulouse (L2ITV) \endhtmlonly \n
             copyright \htmlonly &#169; 2012 Centre d'Elaboration de Mat&#233;riaux et d'Etudes Structurales (CEMES - CNRS) \endhtmlonly \n
    \version 1.1
    \date 2022
    \bug none
    \warning none
*/

#ifndef __cplusplus
#error Must use C++ for the type particle
#endif

#include <iostream>
#include <fstream>
#include <sstream>
#include <assert.h>
#include <vector>
#include <TTree.h>
#include <string>

//===============================
template <class T> class particle
//===============================
{
  private:
    int _event_id;
    uint64_t _particle_ID;
    int _subevent; //PU or HS
    int _barcode;
//    T _p_x, _p_y, _p_z;
    int _N_SP; //on different modules, e.g split SPs = 1
    int _pdgID;
    T _eta;
//    T _phi;
    T _pT;
    // position of the vertex relative to the particle
    T _v_x, _v_y, _v_z;

//    vector<float>* px, py, pz;

  public:
    particle () {}
    particle (const std::map<std::string,int>&, const std::string&, const int&);
    particle (const int&, const uint64_t&, const int&, const int&, const int&, const T&, const T&, const T&, const T&, const T&);
    ~particle () {}

    template <class Tf> friend bool operator == (const particle<Tf>&, const particle<Tf>&);
    template <class Tf> friend bool operator != (const particle<Tf>&, const particle<Tf>&);
    // overload for iostream
    template <class Tf> friend std::ostream& operator << (std::ostream&, const particle<Tf>&);
//    friend istream& operator >> (istream&, const particle&);

    inline const int& event_id () const {return _event_id;}
    inline uint64_t particle_ID () const {return _particle_ID;}
    inline int& subevent () {return _subevent;}
    inline int& barcode () {return _barcode;}
//    inline T& p_x () {return _p_x;}
//    inline T& p_y () {return _p_y;}
//    inline T& p_z () {return _p_z;}
    inline int& n_space_points () {return _N_SP;}
    inline int& pdgID () {return _pdgID;}
    inline T& eta () {return _eta;}
//    inline T& phi () {return _phi;}
    inline T& pT () {return _pT;}
    inline T& v_x () {return _v_x;}
    inline T& v_y () {return _v_y;}
    inline T& v_z () {return _v_z;}

    void save_csv (std::ofstream&);
    bool read_csv (const std::map<std::string,int>&, std::ifstream&, const int&);
};
