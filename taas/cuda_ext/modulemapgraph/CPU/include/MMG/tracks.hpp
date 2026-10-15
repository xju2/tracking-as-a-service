/***********************************************************************
 * Module Map Graph library
 * Track reconstruction for HEP
 * copyright © 2025 COLLARD Christophe
 * Laboratoire des 2 Infinis - Toulouse - France (L2IT)
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 **********************************************************************/

#ifndef __cplusplus
#error Must use C++ for the type tracks
#endif

#include <boost/program_options.hpp>
#include <MMG/parameters>
#include <MMG/colors>
#include <MMG/hit>

//==========
class tracks
//==========
{
  private:
    std::string _events_dir, _output_dir;
    int _nb_tracks;
    std::multimap<uint64_t, std::vector<int>> _particleID_EdgeIndices; // edge -> src / dest indices
    std::multimap<uint64_t, std::vector<int>> _particleID_track; // particleID - track
    std::map<int, uint64_t> _position_particleID;
    std::map<uint64_t, int> _particleID_TrackSize;
    std::vector<int> _TrackSize, _TrackSizeSum;
    std::vector<int> _hitID;

  public:
    tracks () {_nb_tracks=0;}
//    tracks (boost::program_options::variables_map&);

    std::vector<int> operator [] (int) const;
    std::vector<int>& nb_hits_by_track () {return _TrackSize;}
    std::vector<int>& hits_by_track_sum () {return _TrackSizeSum;};
    inline int nb_tracks () const {return _nb_tracks;}
    inline std::vector<int>& hitID () {return _hitID;}
    void build (int=1);
    friend std::ostream& operator << (std::ostream&, const tracks&);
    void read_vertices_csv (const std::string&, const std::string&);
    void read_edges_csv (const std::string&, const std::string&);
    void save (int min_hits,  std::string& filename);
};
