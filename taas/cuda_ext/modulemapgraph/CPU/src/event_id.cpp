/***********************************************************************
 * Module Map Graph library
 * Track reconstruction for HEP
 * copyright © 2023,2024 COLLARD Christophe
 * Laboratoire des 2 Infinis - Toulouse - France (L2IT)
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 **********************************************************************/

 #include "MMG/event_id.hpp"

//-------------------------------------------------------
std::string extract_event_id(const std::string& filename)
//-------------------------------------------------------
{
  // Remove first version number string like vX and vXY (if more than two digits it will break)
  std::string filename_no_version = boost::regex_replace(filename, boost::regex("v[0-9]{0,2}"), std::string(""));

  return boost::regex_replace(filename_no_version, boost::regex("[^0-9]*([0-9]+).*"), std::string("\\1"));
}

//-----------------------------------------------------------------------------------
std::string extract_event_id(const std::string& filename, const std::string& pattern)
//-----------------------------------------------------------------------------------
{
  std::string name = filename.substr (pattern.length());
  // Remove first version number string like vX and vXY (if more than two digits it will break)
  std::string filename_no_version = boost::regex_replace(name, boost::regex("v[0-9]{0,2}"), std::string(""));

  return boost::regex_replace(filename_no_version, boost::regex("[^0-9]*([0-9]+).*"), std::string("\\1"));
}
