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

#include <time.h>
#include <iostream>
#include <string>
#include <boost/program_options.hpp>
#include <MMG/common_options>
#include <MMG/parameters>
#include <MMG/colors>
#include <MMG/CUDA_edge_layer_connector>
#include <cuda_profiler_api.h>
#include <MMG/CUDA_vector_tools>
#include <MMG/CUDA_error_check>

//-------------------------------
int main (int argc, char* argv[])
//-------------------------------
{
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
  my_common_options.add_edge_layer_connector_options<data_type1> (desc);

  boost::program_options::variables_map vm = my_common_options.parse (desc, argc, argv);
  if (vm.empty()) return EXIT_FAILURE;

  /////////////////
  // build graph //
  /////////////////

  std::cout << red << "---------------------------" << reset;
  std::cout << red << "Edge layer connector on GPU" << reset;
  std::cout << red << "---------------------------" << reset;

  start = clock();
  CUDA_edge_layer_connector<data_type1> MM_graph (vm);
  end = clock();

  std::cout << "tracks reconstruction cpu time : " << (double)(end-start)/CLOCKS_PER_SEC << std::endl;
  int time = (double)(end-start)/CLOCKS_PER_SEC;
  int min = time / 60;
  int sec = time - 60 * min;
  std::cout << green << "tracks reconstruction cpu time : " << min << "'" << sec << "\"" << reset;
cudaDeviceSynchronize();
exit(0);

  // same but with arguments
  std::string event_dir, vertices_filename, edges_filename;
  event_dir = "/scratch_ssd/collard/1dump_pyg/";
  vertices_filename = "000001150-hit.csv";
  edges_filename = "000001150-edge.csv";
  CUDA_graph<data_type1> G_from_file (event_dir, vertices_filename, edges_filename);
  int nb_vertices, nb_edges;
  int *vertices, *edges_source, *edges_target;
  data_type1 *edges_weight;

  //cp data from MM_graph
  nb_vertices = G_from_file.vertices();
  nb_edges = G_from_file.size();
  vertices = G_from_file.hit_id();
  edges_source = G_from_file.vertices_in();
  edges_target = G_from_file.vertices_out();
  edges_weight = G_from_file.weights();

  // create graph (from GNN scored edges)
  bool ACTS = true;
  CUDA_edges edges (nb_edges, edges_source, edges_target, edges_weight, ACTS);
  CUDA_graph G (nb_vertices, vertices, &edges);

  data_type1 weights_cut = 0.01;
  int min_hits = 4;
  int blocks = 512;

  CUDA_edge_layer_connector WT (&G, weights_cut, min_hits, blocks);
  WT.build_tracks ();
  WT.filter();

  int nb_tracks = WT.cuda_tracks()->nb_tracks();
  int max_hits_per_track = WT.cuda_tracks()->max_hits_per_track();
  int tracks_size = WT.cuda_tracks()->size();
  std::vector<int> nb_hits(nb_tracks), hits(tracks_size);
  cudaMemcpy (nb_hits.data(), WT.cuda_tracks()->nb_hits(), nb_tracks * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy (hits.data(), WT.cuda_tracks()->hits(), tracks_size * sizeof(int), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  for (int i=0; i<35; i++) {
//    if (nb_hits[i] > min_hits) {
      std::cout << "nb hits = " << nb_hits[i] << std::endl;
      for (int j=0; j<nb_hits[i]; j++) {
        int indice = hits[i*max_hits_per_track+j];
        std::cout << indice << " ";
      }
      std::cout << std::endl;
    }

  cudaDeviceSynchronize();
  GPU_ERROR_CHECK (cudaGetLastError());

  return 0;
};
