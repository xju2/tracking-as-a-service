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

#include "MMG/graph.cuh"

//-----------------------------------------------------------------------
__global__ void printdata (int size, int* overtex)
//-----------------------------------------------------------------------
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size) return;

//  first_vertex[overtex[i]] = 0;
  printf ("i = %d / vertex = %d \n", i, overtex[i]);
}


//=====Private methods for graph=============================================

//=====Public methods for graph==============================================

//--------------------------------------------------------------------------------------------------------------------------------
template <class T>
CUDA_graph<T>::CUDA_graph (const std::string& events_dir, const std::string& vertices_filename, const std::string& edges_filename)
//--------------------------------------------------------------------------------------------------------------------------------
{
  int edges_size = 0;
  _vertices_size = 0;
  char delimiter = ',';

  // read nodes
  std::string path_ReadNodes = events_dir + "/" + vertices_filename;
  std::ifstream file_vertices (path_ReadNodes);
  if (file_vertices.fail()) throw std::invalid_argument ("Cannot open file " + path_ReadNodes);

  std::string csv_content;
  file_vertices >> csv_content;  // delete first line = comments

  std::string v;
  std::vector<int> vertex;
  file_vertices >> csv_content;  // read 1st non comment line
  for (; !file_vertices.eof(); _vertices_size++) {
    std::stringstream ss (csv_content);
    getline (ss, v, delimiter);
    vertex.push_back (std::stoi(v));
    file_vertices >> csv_content;  // delete first line = comments
  }

  file_vertices.close ();

  // read edges
  std::string path_ReadGraphs = events_dir + "/" + edges_filename;
  std::ifstream file_edges (path_ReadGraphs);
  if (file_edges.fail()) throw std::invalid_argument ("Cannot open file " + path_ReadGraphs);

  file_edges >> csv_content;  // delete first line = comments

  std::string v1, v2, w;
  std::vector<int> input_vertex, output_vertex;
  std::vector<T> weight;
  file_edges >> csv_content;  // delete first line = comments

  for (; !file_edges.eof(); edges_size++) {
    std::stringstream ss (csv_content);
    getline (ss, v1, delimiter);
    getline (ss, v2, delimiter);
    getline (ss, w, delimiter);
    input_vertex.push_back (std::stoi(v1));
    output_vertex.push_back (std::stoi(v2));
    weight.push_back (std::stod(w));
    file_edges >> csv_content;  // read first non comment line
  }

  file_edges.close();

  // sort on output vertices
//  std::multimap <int, int> sorted_output_vertices;
//  for (int i=0; i<_edges_size; i++)
//    sorted_output_vertices.insert (std::pair<int,int> (output_vertex[i], i));

//  std::vector<int> sorted_output_vertices_edges;
//  for (std::pair<int,int> vertex :sorted_output_vertices)
//    sorted_output_vertices_edges.push_back(vertex.second);

  // copy data to device
  _memory_allocation = true;
  _edges = new CUDA_edges<T> (edges_size);
  cudaMalloc (&_vertex, _vertices_size * sizeof(int));
//  cudaMalloc (&_sorted_overtex, _edges_size * sizeof(int));
  cudaMemcpy (_vertex, vertex.data(), _vertices_size * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy ((*_edges).source(), input_vertex.data(), edges_size * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy ((*_edges).target(), output_vertex.data(), edges_size * sizeof(int), cudaMemcpyHostToDevice);
//  cudaMemcpy (_sorted_overtex, sorted_output_vertices_edges.data(), _edges_size * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy ((*_edges).weight(), weight.data(), edges_size * sizeof(T), cudaMemcpyHostToDevice);
}

//-----------------------------------------------------------------------------------------------------------------------------
template <class T>
CUDA_graph<T>::CUDA_graph (int *vertices, int nb_vertices, int *edges_source, int *edges_target, T *edges_weight, int nb_edges)
//-----------------------------------------------------------------------------------------------------------------------------
{
  _vertices_size = nb_vertices;

  _memory_allocation = true;
  _edges = new CUDA_edges<T> (nb_edges);
  cudaMalloc (&_vertex, _vertices_size * sizeof(int));
  cudaMemcpy (_vertex, vertices, _vertices_size * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy ((*_edges).source(), edges_source, nb_edges * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy ((*_edges).target(), edges_target, nb_edges * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy ((*_edges).weight(), edges_weight, nb_edges * sizeof(T), cudaMemcpyHostToDevice);
}

//-----------------------------------------------------------------------------------------------------------------------------
template <class T>
CUDA_graph<T>::CUDA_graph (int nb_vertices, int *vertices, CUDA_edges<T> *edges)
//-----------------------------------------------------------------------------------------------------------------------------
{
  _memory_allocation = false;
  _vertices_size = nb_vertices;
  _vertex = vertices;
  _edges = edges;
}

//---------------------------
template <class T>
CUDA_graph<T>::~CUDA_graph ()
//---------------------------
{
  if (_memory_allocation) {
    cudaFree (_vertex);
    delete _edges;
  }
}
