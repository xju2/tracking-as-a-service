/***********************************************************************
 * Module Map Graph library
 * Track reconstruction for HEP
 * copyright © 2022,2023 ROUGIER Charline
 * copyright © 2022, 2023, 2024, 2025 COLLARD Christophe
 * copyright © 2025 VALLIER Alexis
 * Laboratoire des 2 Infinis - Toulouse - France (L2IT)
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 **********************************************************************/

#include "MMG/string_utils.hpp"
#include <regex>
#include <vector>

//---------------------------------------------------------------------------------------------------------------------
std::string extract_event_id_from_filename(const std::string& filename){
//---------------------------------------------------------------------------------------------------------------------
  std::regex numberRegex(R"(\d+)");
  std::sregex_iterator it(filename.begin(), filename.end(), numberRegex);
  std::sregex_iterator end;
  std::vector<std::string> numbers;

  while(it!=end){
    numbers.push_back(it->str());
    ++it;
  }

  if(numbers.size()>0)
    return numbers.back();
  else
    return "";
}
