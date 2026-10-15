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
#error Must use C++ for the type CUDA graph
#endif

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <boost/regex.hpp>
#include <MMG/colors>
#include <MMG/CUDA_edges>

//=================================
template <class T> class CUDA_graph
//=================================
{
  private:
    int _vertices_size;
    bool _memory_allocation;
    int *_vertex;
    CUDA_edges<T> *_edges;

  public:

    CUDA_graph (const std::string&, const std::string&, const std::string&);
    CUDA_graph (int*, int, int*, int*, T*, int); // copy data from CPU to GPU
    CUDA_graph (int, int*, CUDA_edges<T>*); // data are already stored on GPU
    ~CUDA_graph ();

    __device__ inline int* ivertex () {return (*_edges).source();}
    __device__ inline int* overtex () {return (*_edges).target();}

    inline CUDA_edges<T>* edges () const {return _edges;}
    inline int vertices () const {return _vertices_size;}
    inline int size () const {return (*_edges).nb_edges();}
    inline int* vertices_in () const {return (*_edges).source();}
    inline int* vertices_out () const {return (*_edges).target();}
    inline T* weights () const {return (*_edges).weight();}
    inline int* hit_id () const {return _vertex;}
};
