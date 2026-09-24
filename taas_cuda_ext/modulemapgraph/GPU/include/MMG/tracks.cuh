/***********************************************************************
 * Module Map Graph library
 * Track reconstruction for HEP
 * copyright © 2024,2025 COLLARD Christophe
 * Laboratoire des 2 Infinis - Toulouse - France (L2IT)
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 **********************************************************************/

#ifndef __cplusplus
#error Must use C++ for the type CUDA tracks
#endif

#include <MMG/CUDA_mask>
//#include <MMG/CUDA_stream_compaction>
#include <MMG/CUDA_edges>

#include <MMG/tracks>

//==================================
template <class T> class CUDA_tracks
//==================================
{
  private:
    int _max_hits_per_track;
    int _max_tracks; // max nb of tracks
    int _size; // _max_hits_per_track * _max_tracks
    int *_nb_hits;
    int *_hits;
    T *_weights;
    int *_last_vertex;
    uint64_t *_PID; // for debug purpose only

  public:
    CUDA_tracks (int, int);
    CUDA_tracks (tracks&);
    ~CUDA_tracks ();

    inline int size () {return _size;}
    inline int nb_tracks () {return _max_tracks;}
    inline int max_hits_per_track () {return _max_hits_per_track;}
    void add (CUDA_edges<T>&, int);
    void add (CUDA_edges<T>&, CUDA_mask&, int);
    void reverse_add (CUDA_edges<T>&, int);
    inline int* nb_hits () {return _nb_hits;}
    inline int* hits () {return _hits;}
    void filter (int, int);
    template <class Tf>
      friend void diff (const CUDA_tracks<Tf>&, const CUDA_tracks<Tf>&, int);
    template <class Tf>
      friend void diff3 (const CUDA_tracks<Tf>&, const CUDA_tracks<Tf>&, const CUDA_tracks<Tf>&, int);
    void read (const std::string&, const std::string&, int);
    void read_real_tracks (const std::string&, const std::string&, int);
    void save (int, const std::string&);
    void save (int, const std::vector<int>&, std::string&);
    void save (int, CUDA_mask&, std::string&);
};
