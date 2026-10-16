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

#include "MMG/vector_tools.cuh"

//---------------------------------------------------------------------------------------------------------------------
template <class T>
__global__ void init_vector (T *vector, int size){
//---------------------------------------------------------------------------------------------------------------------
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size)
    vector[i] = i;
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
__global__ void init_vector (T *vector, T value, int size){
//---------------------------------------------------------------------------------------------------------------------
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size)
    vector[i] = value;
}

//---------------------------------------------------------------------------------------------------------------------
__global__ void vector_empty_elements (bool* mask, int* vector, int size){
//---------------------------------------------------------------------------------------------------------------------
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i>= size) return;

  mask[i] = (vector[i] != 0);
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
__global__ void compare_vectors (T* v1, T* v2, int size){
//---------------------------------------------------------------------------------------------------------------------
  bool result = true;
  for (int i=0; i<size && result; i++) {
    result *= v1[i] == v2[i];
    if (v1[i] != v2[i]) {
//      printf ("error at %d v1 = %d v2 = %d \n", i-1, v1[i-1], v2[i-1]);
      printf ("error at %d v1 = %d v2 = %d \n", i, (int)v1[i], (int)v2[i]);
//      printf ("error at %d v1 = %d v2 = %d \n", i+1, v1[i+1], v2[i+1]);
      assert(0);
    }
  }
//  printf ("test result = %d \n", result);
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
__global__ void vector_subtraction (T* v1, T* v2, int size){
//---------------------------------------------------------------------------------------------------------------------
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i>= size) return;

  v1[i] -= v2[i];
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
__global__ void print_vector (T* vector, int size){
//---------------------------------------------------------------------------------------------------------------------
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i) return;

  for (int k=0; k<size; k++)
    printf ("%d ", (int)vector[k]);

  printf ("\n");
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
__global__ void print_double_vector (T* vector, int size){
//---------------------------------------------------------------------------------------------------------------------
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i) return;

  for (int k=0; k<size; k++)
    printf ("%f ", vector[k]);

  printf ("\n");
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
__global__ void print_colored_vector (T* vector, int size, int nb){
//---------------------------------------------------------------------------------------------------------------------
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i) return;

  for (int i=0; i<nb; i++)
    printf ("\033[32m%d ", (int)vector[i]);
    for (int i=nb; i<size; i++)
    printf ("\033[31m%d ", (int)vector[i]);

  printf ("\033[0;30m\n");
}

//---------------------------------------------------------------------------------------------------------------------
__global__ void print_colored_double_vector (double* vector, int size, int nb){
//---------------------------------------------------------------------------------------------------------------------
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i) return;

  for (int i=0; i<nb; i++)
    printf ("\033[32m%lf ", vector[i]);
    for (int i=nb; i<size; i++)
    printf ("\033[31m%lf ", vector[i]);

  printf ("\033[0;30m\n");
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
__global__ void print_vector (T* vector, int begin, int end){
//---------------------------------------------------------------------------------------------------------------------
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i) return;

  for (int i=begin; i<end; i++)
    printf ("%d ", (int)vector[i]);

  printf ("\n");
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
__global__ void print_double_vector (T* vector, int begin, int end){
//---------------------------------------------------------------------------------------------------------------------
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i) return;

  for (int i=begin; i<end; i++)
    printf ("%lf ", vector[i]);

  printf ("\n");
}

//---------------------------------------------------------------------------------------------------------------------
__global__ void print_uint64_vector (uint64_t* vector, int begin, int end){
//---------------------------------------------------------------------------------------------------------------------
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i) return;

  for (int i=begin; i<end; i++)
    printf ("%lld ", vector[i]);

  printf ("\n");
}
