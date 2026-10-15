/***********************************************************************
 * Module Map Graph library
 * Track reconstruction for HEP
 * copyright © 2022,2023 ROUGIER Charline
 * copyright © 2022,2025 COLLARD Christophe
 * Laboratoire des 2 Infinis - Toulouse - France (L2IT)
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 **********************************************************************/

/*! \class particles
    \brief particles library \n

    \htmlonly 
    <FONT color="#838383">

    insert license
    </FONT>
    \endhtmlonly

    Particles are a set of \ref particle data. Each particule is identified by a unique \ref particle_id  \n
    Data are stored in root format.

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
#error Must use C++ for the type particles
#endif

#include <iostream>
#include <fstream>
#include <sstream>
#include <assert.h>
#include <vector>
#include <TFile.h>
#include <TTree.h>
#include <boost/type_index.hpp>
#include <boost/regex.hpp>
#include <MMG/particle>
#include <MMG/TTree_particles>
#include <MMG/string_utils>

//================================
template <class T> class particles
//================================
{
  private:
    int _size;
    int _event_id;
    std::multimap<uint64_t, particle<T>> _ParticlesMap;
    // correspondance between eventID and storage position

/*
    std::vector<int> subevent_table;

    // particles properties
    std::vector<int long>* particle_id;
    std::vector<int>* subevent; //PU or HS
    std::vector<int>* barcode;
    std::vector<int>* N_SP; //on different modules, e.g split SPs = 1
    std::vector<int>* pdgID;
    std::vector<T>* eta;
    std::vector<T>* phi;
    std::vector<T>* pT;
    std::vector<T>* v_x;
    std::vector<T>* v_y;
    std::vector<T>* v_z;

    std::vector<std::pair<double,double>>* cut_z0; //two entries per index: 12 and 23
    std::vector<std::pair<double,double>>* cut_dphi;
    std::vector<std::pair<double,double>>* cut_phiSlope;
    std::vector<std::pair<double,double>>* cut_deta;

//    std::vector<double>* cut_diff_dzdr;
//    std::vector<double>* cut_diff_dxdy;

//    vector<float>* px, py, pz;
    std::vector<long int> location;
*/

  public:
    particles ();
    particles (TTree_particles<T>&);  // cast conversion TTree_particles -> particles
    ~particles () {}

    operator TTree_particles<T> (); // cast conversion hits -> TTree_particles
//    particle& operator [] (long int) const;     // returns the address of an element
    particles& operator = (const particles<T>&);
    particles& operator += (const particle<T>&); //////////////////////////////////////////////////// continue here - implement this operator
    template <class Tf> friend bool operator == (const particles<Tf>&, const particles<Tf>&);
    template <class Tf> friend bool operator != (const particles<Tf>&, const particles<Tf>&);

    // overload for iostream
    template <class Tf> friend std::ostream& operator << (std::ostream&, const particles<Tf>&);
//    friend istream& operator >> (istream&, const particle&);

    int size () {return _size;}
    inline std::_Rb_tree_iterator <std::pair <const uint64_t, particle<T> > > begin() {return _ParticlesMap.begin();}
    inline std::_Rb_tree_iterator <std::pair <const uint64_t, particle<T> > > end() {return _ParticlesMap.end();}
    inline std::pair<std::_Rb_tree_iterator<std::pair<const uint64_t, particle<T> > >, std::_Rb_tree_iterator<std::pair<const uint64_t, particle<T> > > > equal_range (const uint64_t& module) {return _ParticlesMap.equal_range(module);}

    inline int& event_id () {return _event_id;}

    void save_csv (const std::string&);
    void read_csv (const std::string&, const std::string&);
    void read_csv_once (const std::string&, const std::string&);


//    struct_ModuleTriplet_particlesInfo(): event_id{},
//    particle_id{}, subevent{},
//    barcode{},
//    N_SP{}, pdgID{},
//    eta{}, phi{}, 
//    pT{}, v_x{},
//    v_y{}, v_z{},
//    cut_z0{}, 
//    cut_dphi{}, 
//    cut_phiSlope{}, 
//    cut_deta{},
//    cut_diff_dzdr{}, cut_diff_dxdy{} {}
};
