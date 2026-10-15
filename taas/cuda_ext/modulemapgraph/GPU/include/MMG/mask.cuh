/***********************************************************************
 * Module Map Graph library
 * Track reconstruction for HEP
 * copyright © 2024,2025,2026 COLLARD Christophe
 * Laboratoire des 2 Infinis - Toulouse - France (L2IT)
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 **********************************************************************/

#ifndef __cplusplus
#error Must use C++ for the type CUDA mask
#endif

#include <string>
#include <iostream>
#include <MMG/colors>
#include <MMG/CUDA_prescan>
#include <MMG/CUDA_vector_tools>

//=============
class CUDA_mask
//=============
{
  private:
    int _size;
    int _sum;
    bool *_cuda_mask;
    int *_cuda_mask_sum;

  public:
    inline CUDA_mask () {_size = _sum = 0;}
    CUDA_mask (int);
    CUDA_mask (int, bool, int);
    ~CUDA_mask ();

    void init (int);
    void init (int, bool, int);
    int scan (int);
    void reverse (int);

    inline int size () {return _size;}
    inline int sum () {return _sum;}
    inline bool* cuda_mask () {return _cuda_mask;}
    inline int* cuda_mask_sum () {return _cuda_mask_sum;}
};
