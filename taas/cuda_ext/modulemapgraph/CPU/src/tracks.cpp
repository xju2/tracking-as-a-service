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

#include "MMG/tracks.hpp"

/*
//--------------------------------------------------------------
template <class T>
tracks<T>::tracks (boost::program_options::variables_map& po_vm)
//--------------------------------------------------------------
{
  _events_dir = po_vm["input-dir"].as<std::string>() + "/";
  _output_dir = po_vm["output-dir"].as<std::string>();

  std::string event_filename_pattern = po_vm["input-filename-pattern"].as<std::string>();
  // log all hits events filenames
  std::string log_track_filenames = "tracks.txt";
  std::string cmd_tracks = "ls " + _events_dir + "|grep " + event_filename_pattern + " > " + log_track_filenames;
  system(cmd_tracks.c_str());

  // open files containing events filenames
  std::ifstream t_file(log_track_filenames);
  if (t_file.fail())
    throw std::invalid_argument("Cannot open file " + log_track_filenames);

  std::string track_filename;

  std::string log_nb_events = "events_nb.txt";
  std::string cmd_nb_events = "wc -l " + log_track_filenames + " > " + log_nb_events;
  system (cmd_nb_events.c_str());
  std::ifstream e_file (log_nb_events);
  e_file >> _nb_events;
  std::cout << red << "nb events = " << _nb_events << reset;
  e_file.close();

//  read_events (t_file, event_filename_pattern);
}
*/

//------------------------------------------------
std::vector<int> tracks::operator [] (int i) const
//------------------------------------------------
{
  assert (i>=0 && i< _nb_tracks);

  return _particleID_track.find (_position_particleID.find(i)->second) -> second;
}

//-------------------------------
void tracks::build (int min_hits)
//-------------------------------
{
  assert (min_hits >= 1);

  for (std::_Rb_tree_iterator<std::pair<const uint64_t, std::vector<int>>> it = _particleID_EdgeIndices.begin(); it != _particleID_EdgeIndices.end();) {
    // sort edges by ID
    uint64_t module = it->first;

    // extract edges by ID
    std::pair<std::_Rb_tree_iterator<std::pair<const uint64_t, std::vector<int>>>, std::_Rb_tree_iterator<std::pair<const uint64_t, std::vector<int>>>> edges = _particleID_EdgeIndices.equal_range (module);

    // store edge input and edge ouput and count the number of edges (assuming edges are classified by ID)
    std::vector<int> edge_input, edge_output;
    int nb=0;
    for (std::_Rb_tree_iterator<std::pair<const uint64_t, std::vector<int>>> edge = edges.first; edge != edges.second; edge++) {
      edge_input.push_back (edge->second[0]);
      edge_output.push_back (edge->second[1]);
      it++;
      nb++;
    }

    if (nb >= min_hits) {
    // look for first edge -> start track
    bool first_edge = true;
    int nb_first = 0;
    std::vector<int> track;
    // find first edge and store it in track
    for (int k=0; k<nb; k++, first_edge=true) {
      for (int l=0; l<nb && first_edge; l++)
        if (edge_input[k] == edge_output[l]) first_edge = false;
      if (first_edge) {
        track.push_back (edge_input[k]);
        track.push_back (edge_output[k]);
        nb_first++;

        // sort and store other edges with the same module ID in track
        int nb_others = nb + nb_first - 1;

        for (int k=track.size()-1; k<nb_others; k++)
          for (int l=0; l<nb_others; l++)
            if (track[k] == edge_input[l]) {
              track.push_back (edge_output[l]);
              break;
            }

        _particleID_track.insert (std::pair<uint64_t, std::vector<int>> (module, track));
        _position_particleID.insert (std::pair<int, uint64_t> (_nb_tracks, module));
        _particleID_TrackSize.insert (std::pair<uint64_t, int> (module, track.size()));
        _nb_tracks++;
        }
      }

    if (nb_first > 1) {
      std::cout << "error: more than 1 first edge" << std::endl;
      for (int k=0; k<2*nb_first; k++)
        std::cout << track[k] << " ";
      std::cout << std::endl;
      for (int k=0; k<nb; k++)
        std::cout << edge_input[k] << " ";
      std::cout << std::endl;
      for (int k=0; k<nb; k++)
        std::cout << edge_output[k] << " ";
      std::cout << std::endl;
    }

    // sort and store other edges with the same module ID in track
    int nb_others = nb + nb_first - 1;

/*
    for (int k=track.size()-1; k<nb_others; k++)
      for (int l=0; l<nb_others; l++)
        if (track[k] == edge_input[l]) {
          track.push_back (edge_output[l]);
          break;
        }
*/

    if (nb_first != 1) {
      std::cout << red << "error" << reset;
      std::cout << "module ID = " << module << std::endl;
      std::cout << "nb = " << nb << std::endl;
      for (int k=0; k<nb; k++)
        std::cout << edge_output[k] << " ";
      std::cout << std::endl;
      for (int k=0; k<nb; k++)
        std::cout << edge_input[k] << " ";
      std::cout << std::endl;
      for (int k=0; k<=nb_others; k++)
        std::cout << track[k] << " ";
      std::cout << std::endl;
    }

//    if (module == 6080010723) std::cout << blue << "PARTICLE FOUND !!!!!!!!!!!!!!!!" << " / track nb = " << _nb_tracks+1 << reset;
    if (module == 6120010103) {
      std::cout << blue << "PARTICLE FOUND !!!!!!!!!!!!!!!!" << " / track nb = " << _nb_tracks+1 << reset;
      for (int n=0; n<track.size(); n++)
        std::cout << track[n] << " ";
      std::cout << std::endl;
    }
    }
  }

  _TrackSizeSum.push_back(0);
  for (int i=0; i<_nb_tracks; i++) {
    _TrackSize.push_back (_particleID_TrackSize.find (_position_particleID.find(i)->second) -> second);
    _TrackSizeSum.push_back (_TrackSize[i] + _TrackSizeSum[i]);
  }
}

//--------------------------------------------------------------
std::ostream& operator << (std::ostream& s, const tracks& trcks)
//--------------------------------------------------------------
{
//  std::cout << red << trcks._particleID_EdgeIndices.size() << reset;
int i = 0;
//  for (std::pair<uint64_t, std::vector<int>>::iteratormodule_doublet<T>> ModuleDoublet : _module_map_doublet) {
  for (std::pair<uint64_t, std::vector<int>> it : trcks._particleID_EdgeIndices)
    s << i++ << ": " << it.first << ", " << (it.second)[0] << ", " << (it.second)[1] << std::endl;
  s << red << trcks._particleID_EdgeIndices.size() << reset;

  return s;
}

//-----------------------------------------------------------------------------------
void tracks::read_vertices_csv (const std::string& path, const std::string& filename)
//-----------------------------------------------------------------------------------
{
  std::ifstream file (path + "/" + filename);
  if (file.fail()) throw std::invalid_argument ("Cannot open file " + path + filename);

  // delete first line = comments
  std::string content;
  file >> content;
  std::cout << content << std::endl;

  // store index names
  std::stringstream ssi (content);
  std::vector<std::string> column_id;
  std::string line;
  char delim = ',';
  std::map<std::string,int> index;
  for (int i=0; getline (ssi, line, delim); i++)
      index[line] = i;

  file >> content;

  for (; !file.eof(); file>>content) {
    std::stringstream ss (content);
    std::vector<std::string> column_id;
    std::string line;
    for (;getline (ss, line, delim);)
      column_id.push_back (line);

    int hitID;
    std::istringstream (column_id[index.find("hit_id")->second]) >> hitID;
    _hitID.push_back (hitID);
  }

  file.close();
}

//--------------------------------------------------------------------------------
void tracks::read_edges_csv (const std::string& path, const std::string& filename)
//--------------------------------------------------------------------------------
{
  std::ifstream file (path + "/" + filename);
  if (file.fail()) throw std::invalid_argument ("Cannot open file " + path + filename);

  // delete first line = comments
  std::string content;
  file >> content;
  std::cout << content << std::endl;

  // store index names
  std::stringstream ssi (content);
  std::vector<std::string> column_id;
  std::string line;
  char delim = ',';
  std::map<std::string,int> index;
  for (int i=0; getline (ssi, line, delim); i++)
      index[line] = i;

  file >> content;

  for (; !file.eof(); file>>content) {
    std::stringstream ss (content);
    std::vector<std::string> column_id;
    std::string line;
    for (;getline (ss, line, delim);)
      column_id.push_back (line);

    std::vector<int> edge (2);
    uint64_t particle_id;
    std::string split_edge;
    std::istringstream (column_id[index.find("track_edges_src")->second]) >> edge[0];
    std::istringstream (column_id[index.find("track_edges_dst")->second]) >> edge[1];
    std::istringstream (column_id[index.find("track_particle_id")->second]) >> particle_id;
    std::istringstream (column_id[index.find("track_redundant_split_edges")->second]) >> split_edge;
    
    if (split_edge == "False")
      _particleID_EdgeIndices.insert (std::pair<uint64_t, std::vector<int>> (particle_id, edge));
  }

  file.close();
}

//------------------------------------------------------
void tracks::save (int min_hits,  std::string& filename)
//------------------------------------------------------
{
  std::cout << "removing file " << filename << std::endl;
  std::string cmd = "rm " + filename;
  system (cmd.c_str());

  std::cout << "saving data ...." << reset;
  int maxID = _hitID.size();
  std::ofstream outfile;
  outfile.open(filename.c_str(), std::ios_base::app);

  for (std::_Rb_tree_iterator<std::pair<const uint64_t, std::vector<int>>> it = _particleID_track.begin(); it != _particleID_track.end(); it++) {
    uint64_t pID = it->first;
    std::vector<int> track = it->second;
    for (int i=0; i<track.size(); i++)
      outfile << track[i] << " ";
    outfile << std::endl;
  }

/*
  std::map<int, uint64_t>::iterator it = _position_particleID.begin();
  for (int pos=0; pos<_nb_tracks; pos++, it++) {
    int nb_hits = it->second;
    uint64_t particleID = _position_particleID.find (pos) -> second;
    std::Rb_tree_iterator<std::pair <const uint64_t, hit<T> > >td::multimap<uint64_t, std::vector<int>>
    std::vector<int> track = 
    if (nb_hits >= min_hits) {
      for (int j=0; nb_hits; j++) {
        assert (j < maxID);
//        outfile << _particleID_track.find (particleID) -> second << " ";
      }
      outfile << std::endl;
    }
  }
*/
  outfile.close();

//  std::cout << "max tracks = " << _max_tracks << std::endl;
}
