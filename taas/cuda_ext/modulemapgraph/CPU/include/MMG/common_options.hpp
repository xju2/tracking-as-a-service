/***********************************************************************
 * Module Map Graph library
 * Track reconstruction for HEP
 * copyright © 2022,2023 ROUGIER Charline
 * copyright © 2022,2023,2024,2025 COLLARD Christophe
 * Laboratoire des 2 Infinis - Toulouse - France (L2IT)
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 **********************************************************************/

#ifndef __cplusplus
#error Must use C++ for the type common options
#endif

#include <string>
#include <iostream>
#include <fstream>
#include <exception>
#include <regex>
#include <system_error>
#include <boost/program_options.hpp>


enum class output_format : uint8_t {
  DirectoryOnly = 0,
  Root = 1,
  Csv = 2,
  Obj = 4,
  Json = 8,
  Txt = 16,
  All = std::numeric_limits<uint8_t>::max()
};

//==================
class common_options
//==================
{
  public:
    common_options (boost::program_options::options_description&);
    ~common_options () {}

    /// Add hardware options
    void add_hardware_options (boost::program_options::options_description&);

    /// Add sequencer options, e.g. number of events.
    void add_sequencer_options (boost::program_options::options_description&);

    /// Add common input-related options.
    void add_input_options (boost::program_options::options_description&);

    void add_output_options (boost::program_options::options_description&);

    /// Add module map options
    void add_module_map_options (boost::program_options::options_description&);

    /// Add random number options such as the global seed.
    void add_random_numbers_options (boost::program_options::options_description&);

    /// Add common geometry-related options.
    void add_geometry_options (boost::program_options::options_description&);

    /// Add common material-related options.
    void add_material_options (boost::program_options::options_description&);

    /// Add common graph-related options.
    template <class T> void add_graph_options (boost::program_options::options_description&);

    template <class T> void add_edge_layer_connector_options (boost::program_options::options_description&);

    /// Parse options and return the resulting variables map.
    ///
    /// Automatically prints the help text if requested.
    ///
    /// @returns Empty variables map if help text was shown.
    boost::program_options::variables_map parse (const boost::program_options::options_description&, int, char**) noexcept(false);
    /// Read the log level.
//    Acts::Logging::Level read_log_level (const boost::program_options::variables_map&);
    /// Read the sequencer config.
//    ActsExamples::Sequencer::Config read_sequencer_config (const boost::program_options::variables_map&);
};
