/***********************************************************************
 * Module Map Graph library
 * Track reconstruction for HEP
 * copyright © 2025 COLLARD Christophe
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
#include <MMG/module_map_triplet>
#include <MMG/graph>
#include <MMG/hit>
#include <MMG/nodes>
#include <MMG/edges>
#include <MMG/CUDA_graph_creator>
#include <MMG/CUDA_nodes>
#include <MMG/colors>
#include <MMG/CUDA_error_check>
#include <thread>
#include <chrono>

/*
void task (int nb)
{
  std::this_thread::sleep_for(std::chrono::seconds(10-nb));
  std::cout << "task #: " << nb << std::endl;
  int addr;
  std::cout << "addr = " << &addr << std::endl;
}
*/

//---------------------------------------------------------------------------------------------------------------------
int main (int argc, char* argv[]){
//---------------------------------------------------------------------------------------------------------------------
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

  std::cout << red << "-------------------------------------" << reset;
  std::cout << red << "Build Graphs on GPU (multi-threading)" << reset;
  std::cout << red << "-------------------------------------" << reset;

  std::chrono::system_clock::time_point start = std::chrono::system_clock::now();

  std::string event_filename_pattern = vm["input-filename-pattern"].as<std::string>();
  std::string events_dir = vm["input-dir"].as<std::string>() + "/";

  //------------------------
  // read Module Map Triplet
  //------------------------

  std::string module_map_dir = vm["input-module-map"].template as<std::string>();
  if (module_map_dir.empty())
    throw std::invalid_argument ("Missing Module Map");

  module_map_triplet<data_type2> module_map_triplet;
  module_map_triplet.read_TTree(module_map_dir.c_str());
  if (!module_map_triplet)
    throw std::runtime_error("Cannot retrieve ModuleMap from " + module_map_dir);

  //------------------------------
  // log all hits events filenames
  //------------------------------

  std::string log_hits_filenames = "events-hits.txt";
  std::string cmd_hits = "ls " + events_dir + "|grep " + event_filename_pattern + " |grep hits.* > " + log_hits_filenames;
  system(cmd_hits.c_str());
  cmd_hits = "ls " + events_dir + "|grep " +  event_filename_pattern + " |grep truth.* >> " + log_hits_filenames;
  system(cmd_hits.c_str());

  //---------------------------------------
  // open files containing events filenames
  //---------------------------------------

  std::ifstream h_file(log_hits_filenames);
  if (h_file.fail())
    throw std::invalid_argument("Cannot open file " + log_hits_filenames);

  int nb_events = 2;
  int blocks = 64;
  int max_blocks = 1024; // device related
  int device = 0;
  std::vector<cudaStream_t> streams (nb_events);

  std::string hits_filename [nb_events];
  for (int i=0; !h_file.eof() && i<nb_events; i++)
    h_file >> hits_filename[i];

std::cout << "filename = " << hits_filename << std::endl;

//  CUDA_graph_creator<data_type1> my_graph (vm);
//  CUDA_graph_creator<data_type1,data_type2> my_graph (vm);
  nodes<data_type1> *graph_nodes = new nodes<data_type1> [nb_events];
  edges<data_type1> *graph_edges = new edges<data_type1> [nb_events];

  CUDA_nodes<data_type1> nodes [nb_events];
  CUDA_edges<data_type1> edges [nb_events];


  cudaDeviceSynchronize();
  std::chrono::system_clock::time_point start_time = std::chrono::system_clock::now();


  //------------------------
  // read events in parallel
  //------------------------

  int nb_gpu = 0;
  cudaGetDeviceCount (&nb_gpu);
  std::cout << blue << "CUDA info: " << nb_gpu <<  " GPU found" << reset;

  std::vector<CUDA_graph_creator<data_type1,data_type2>> graphs (nb_events);
  std::vector<std::thread> events;

/*
  std::vector<std::thread> worker;
  // Constructs the new thread and runs it. Does not block execution.
  for (int i=0; i<10; i++)
    worker.push_back (
      std::thread (task, i));

    // Do other things...

    // Makes the main thread wait for the new thread to finish execution, therefore blocks its own execution.
  for (int i=0; i<10; i++)
    worker[i].join();
*/

  for (int i=0; i<nb_events; i++){
    std::string event_id = extract_event_id (hits_filename[i], event_filename_pattern);
    events.push_back(
      std::thread (&CUDA_graph_creator<data_type1,data_type2>::init, &(graphs[i]),
                   std::ref(vm), std::ref(module_map_triplet), event_id, std::ref(hits_filename[i]), blocks, max_blocks, device, std::ref(streams[i])));
//    device = (device + 1) % nb_gpu;)));
  }

  for (int i=0; i<nb_events; i++)
    events[i].join();

  //-----------------------------------
  // run graph construction in parallel
  //-----------------------------------

//  CUDA_graph_creator<data_type1,data_type2> graph (vm, module_map_triplet, blocks, device);

  cudaDeviceSynchronize();
  std::chrono::system_clock::time_point start_build = std::chrono::system_clock::now();

  std::vector<std::thread> workers;

///* reprendre ici          reprendre ici          reprendre ici          reprendre ici          reprendre ici          reprendre ici          reprendre ici          reprendre ici
  for (int i=0; i<nb_events; i++){
    std::string event_id = extract_event_id (hits_filename[i], event_filename_pattern);
    workers.push_back(
      std::thread (&CUDA_graph_creator<data_type1,data_type2>::multithreads_build, &(graphs[i]),
                   std::ref(nodes[i]), std::ref(edges[i]), std::ref(streams[i])));
  }

  for (int i=0; i<nb_events; i++)
    workers[i].join();
//*/


//  for (int i=0; i<nb_events; i++)
//    graphs[i].multithreads_build (nodes[i], edges[i], streams[i]);

//  for (int i=0; i<nb_events; i++){
//    std::thread (&CUDA_graph_creator<data_type1,data_type2>::multithreads_build, &(graphs[i]),
//                 std::ref(nodes[i]), std::ref(edges[i])).detach();
//  }


  cudaDeviceSynchronize();
  std::chrono::system_clock::time_point end_build = std::chrono::system_clock::now();

  cudaDeviceSynchronize();
  std::chrono::system_clock::time_point end_time = std::chrono::system_clock::now();
  std::cout << green << "real cpu/gpu time = " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << " ms"<< reset;
  std::cout << green << "real cpu/gpu time = " << std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count() << " s"<< reset;
  std::cout << yellow << "real cpu/gpu time to build graphs = " << std::chrono::duration_cast<std::chrono::milliseconds>(end_build - start_build).count() << " ms" << reset;
  std::cout << yellow << "average cpu/gpu time to build graphs = " << std::chrono::duration_cast<std::chrono::milliseconds>(end_build - start_build).count() / nb_events << " ms" << reset;

/*
//my_graph.build (nodes_interactive[0], edges_interactive[0], 0, hit_id, module_id, x, y, z, size, 512); /// delete this line after debug
  std::cout << yellow << "NB EVENTS = " << my_graph.nb_events() << reset;
  for (int i=0; i<1; i++){ //my_graph.nb_events(); i++) {
    std::cout << "saving event ID " << my_graph.eventID()[i] << std::endl;
    node[i].DeviceToHost (graph_nodes[i]);
    edge[i].DeviceToHost (graph_edges[i]);
    graph<data_type1> G (graph_nodes[i], graph_edges[i]);
    G.save_csv (my_graph.eventID()[i], vm["output-dir"].as<std::string>(), false);
  }
*/

  delete [] graph_nodes;
  delete [] graph_edges;

  cudaDeviceSynchronize();
  std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
  std::cout << green << "graph build full cpu/gpu time = " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << " s" << reset;

  return 0;
};
