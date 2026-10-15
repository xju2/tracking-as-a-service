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

/*! \class hits
    \brief hits library \n

    \htmlonly 
    <FONT color="#838383">

    insert license
    </FONT>
    \endhtmlonly

    Hits are a set of \ref hit data. Each hit belong to a unique \ref particle which is identified by a unique \ref particle_id  \n
    Data are stored in root format.

    \authors copyright \htmlonly &#169; \endhtmlonly 2022 2023, 024 Christophe COLLARD \n
             copyright \htmlonly &#169; \endhtmlonly 2022, 2023 Charline Rougier \n
             copyright \htmlonly 2022, 2023, 2024 Centre National de la Recherche Scientifique \endhtmlonly \n
             copyright \htmlonly 2022, 2023, 2024 Universit&#233; Paul Sabatier, Toulouse 3 \endhtmlonly \n
             copyright \htmlonly &#169; 2022, 2023, 2024 Laboratoire des 2 infinis de Toulouse (L2ITV) \endhtmlonly \n
    \version 0.1
    \date 2022-2024
    \bug none
    \warning none
*/

#ifndef __cplusplus
#error Must use C++ for the type hits
#endif

#include <iostream>
#include <fstream>
#include <assert.h>
#include <vector>
#include <TFile.h>
#include <TTree.h>
#include <boost/type_index.hpp>
#include <MMG/hit>
#include <MMG/TTree_hits>


//===========================
template <class T> class hits
//===========================
{
  private:
    int _size;
    bool _true_graph;
    bool _extra_features;
//    int _event_id;
//    TTree_hits<T> _TTreeHits;
    std::multimap<uint64_t, hit<T>> _HitsMap;
//    std::multimap<uint64_t, int> _moduleID_TTreeHits_map;
//    std::multimap<uint64_t, int> _particleID_TTreeHits_map;

  public:
    hits (bool=false, bool=false);
    hits (TTree_hits<T>&);  // cast conversion TTree_hits -> hits
    ~hits () {}

    operator TTree_hits<T> (); // cast conversion hits -> TTree_hits
    hits<T>& operator = (const hits<T>&);
    hits<T>& operator += (const hit<T>&);
    template <class Tf> friend bool operator == (const hits<Tf>&, const hits<Tf>&);
    template <class Tf> friend bool operator != (const hits<Tf>&, const hits<Tf>&);

    int size () {return _size;}
    inline const std::multimap<uint64_t, hit<T>>& hits_map () const {return _HitsMap;}
    inline bool true_graph () {return _true_graph;}
    inline bool extra_features () {return _extra_features;}
    inline const std::multimap<uint64_t, hit<T>>& get_hits () {return _HitsMap;}
    inline std::_Rb_tree_iterator <std::pair <const uint64_t, hit<T> > > begin() {return _HitsMap.begin();}
    inline std::_Rb_tree_iterator <std::pair <const uint64_t, hit<T> > > end() {return _HitsMap.end();}
    // overload for iostream
    template <class Tf> friend std::ostream& operator << (std::ostream&, const hits<Tf>&);
//    friend istream& operator >> (istream&, const particle&);

//    inline int& event_id () {return _event_id;}

    template <class Tf> friend int define_region (const hit<Tf>& source, const hit<Tf>&);

    void save_root (const std::string&);
    inline void read_root (const std::string&);
//    inline void read_root (const std::string& filename) {_TTreeHits.read(filename);}
    void save_csv (const std::string&);
    void read_csv (const std::string&, const std::string&);
    void read_csv_once (const std::string&, const std::string&);
};
