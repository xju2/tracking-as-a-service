/***********************************************************************
 * Module Map Graph library
 * Track reconstruction for HEP
 * copyright © 2024,2025 COLLARD Christophe
 * Laboratoire des 2 Infinis de Toulouse - France (L2IT)
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 **********************************************************************/

#include <time.h>
#include <iostream>
#include <string>
#include <boost/program_options.hpp>
#include <MMG/common_options>
#include <MMG/parameters>
#include <MMG/graph>
#include <MMG/hit>
#include <MMG/nodes>
#include <MMG/edges>
#include <MMG/CUDA_graph_creator>
#include <MMG/CUDA_nodes>
#include <MMG/colors>
#include <MMG/CUDA_error_check>

__global__ void null_kernel(){}

//-------------------------------
int main (int argc, char* argv[])
//-------------------------------
{
  // warm up GPU card
  cudaDeviceSynchronize();
  std::chrono::system_clock::time_point start_null = std::chrono::system_clock::now();
  null_kernel<<<1,1>>> ();
  cudaDeviceSynchronize();
  std::chrono::system_clock::time_point end_null = std::chrono::system_clock::now();
  std::cout << red << "init card cpu time : " << std::chrono::duration_cast<std::chrono::microseconds>(end_null - start_null).count() * 0.001 << " ms" << reset;

  clock_t start, end;

  start = clock();

  boost::program_options::options_description desc;
  common_options my_common_options (desc);
  my_common_options.add_hardware_options (desc);
  my_common_options.add_sequencer_options (desc);
  // Options::addGeometryOptions (desc);
  //Options::addMaterialOptions (desc);
  my_common_options.add_input_options (desc);
  //  Options::addGraphCreationOptions (desc);
  my_common_options.add_output_options (desc);
  my_common_options.add_graph_options<data_type1> (desc);

  boost::program_options::variables_map vm = my_common_options.parse (desc, argc, argv);
  if (vm.empty()) return EXIT_FAILURE;

  /////////////////
  // build graph //
  /////////////////

  start = clock();

  std::cout << red << "-------------------" << reset;
  std::cout << red << "Build Graphs on GPU" << reset;
  std::cout << red << "-------------------" << reset;

/*
// --test wrapper-----
  int size = 0;
  uint64_t *hit_id, *module_id;
  data_type1 *x, *y, *z;

  std::ifstream file ("1event/event000000001-truth.csv");
  if (file.fail()) throw std::invalid_argument ("Cannot open file 1event/event000000001-truth.csv");
  // delete first line = comments
  std::string content;
  file >> content;
  // store index names
  std::stringstream ssi (content);
  std::vector<std::string> column_id;
  std::string line;
  char delim = ',';
  std::map<std::string,int> index;

  for (int i=0; getline (ssi, line, delim); i++)
      index[line] = i;

  int max_size = 267828;
  hit_id = new uint64_t [max_size];
  module_id = new uint64_t [max_size];
  x = new data_type1 [max_size];
  y = new data_type1 [max_size];
  z = new data_type1 [max_size];

  for (bool eof = false; !eof;)
    { hit<data_type1> ht;
      eof = ht.read_csv (index, file);
      if (eof) break;
      hit_id[size] = ht.hit_id();
      module_id[size] = ht.module_ID();
      x[size] = ht.x();
      y[size] = ht.y();
      z[size] = ht.z();
      size++;
      assert (size <= max_size);
    }
  file.close();
  std::cout << "nb_hits = " << size << std::endl;
  assert (size==max_size);
  int max_events = 1;
  CUDA_graph_creator<data_type1> my_graph_interactive (vm, false);
  std::cout << yellow << "nb events = " << my_graph_interactive.nb_events() << reset;
  CUDA_nodes<data_type1> nodes_interactive;
  CUDA_edges<data_type1> edges_interactive;
  my_graph_interactive.build (nodes_interactive, edges_interactive, "000000001", hit_id, module_id, x, y, z, size);
  std::cout << "nb nodes = " << nodes_interactive.nb_nodes() << std::endl;
  std::cout << "nb edges = " << edges_interactive.nb_edges() << std::endl;
  std::cout << "print vector on CPU" << std::endl;
  for (int i=50; i<60; i++)
    std::cout << module_id[i] << " ";
  std::cout << std::endl;
  my_graph_interactive.save (nodes_interactive, edges_interactive, "000000001");
  cudaDeviceSynchronize();
// --test wrapper-end----
*/

//  CUDA_graph_creator<data_type1> my_graph (vm);
  CUDA_graph_creator<data_type1,data_type2> my_graph (vm);
  nodes<data_type1> *graph_nodes = new nodes<data_type1> [my_graph.nb_events()];
  edges<data_type1> *graph_edges = new edges<data_type1> [my_graph.nb_events()];

  CUDA_nodes<data_type1> node [my_graph.nb_events()];
  CUDA_edges<data_type1> edge [my_graph.nb_events()];

  clock_t start_dev, end_dev;
  cudaDeviceSynchronize();
  start_dev = clock ();

  for (int i=0; i<my_graph.nb_events(); i++) {
    my_graph.build (node[i], edge[i], i);
  }

  cudaDeviceSynchronize();
  end_dev = clock();
  std::cout << red << "average cpu/gpu time = " << (double) (end_dev - start_dev) / CLOCKS_PER_SEC / my_graph.nb_events() << reset;
  std::cout << yellow << "NB EVENTS = " << my_graph.nb_events() << reset;
  for (int i=0; i<my_graph.nb_events(); i++) {
    std::cout << "saving event ID " << my_graph.eventID()[i] << std::endl;
    node[i].DeviceToHost (graph_nodes[i]);
    edge[i].DeviceToHost (graph_edges[i]);
    graph<data_type1> G (graph_nodes[i], graph_edges[i]);
    G.save_csv (my_graph.eventID()[i], vm["output-dir"].as<std::string>(), false);
  }

  delete [] graph_nodes;
  delete [] graph_edges;

  end = clock();
  std::cout << "graph build cpu time : " << (double)(end-start)/CLOCKS_PER_SEC << std::endl;
  std::cout << "end read file" << std::endl;
  int time = (double)(end-start)/CLOCKS_PER_SEC;
  int min = time / 60;
  int sec = time - 60 * min;
  std::cout << green << "graph build cpu time : " << min << "'" << sec << "\"" << reset;

  return 0;
};
