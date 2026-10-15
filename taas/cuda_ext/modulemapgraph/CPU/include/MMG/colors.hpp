/***********************************************************************
 * Module Map Graph library
 * Track reconstruction for HEP
 * copyright © 2022,2023 COLLARD Christophe
 * Laboratoire des 2 Infinis - Toulouse - France (L2IT)
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 **********************************************************************/

#ifndef __cplusplus
#error Must use C++ for colors
#endif

#ifndef __colors_hpp
#define __colors_hpp

#ifndef __iostream
#include <iostream>
#endif


//------------------------------------------
inline std::ostream& reset (std::ostream& s)
//------------------------------------------
{
  s << std::endl << "\033[0m";
  return s;
}


//------------------------------------------
inline std::ostream& black (std::ostream& s)
//------------------------------------------
{
  s << "\033[1;30m";
  return s;
}


//----------------------------------------
inline std::ostream& red (std::ostream& s)
//----------------------------------------
{
  s << "\033[1;31m";
  return s;
}


//------------------------------------------
inline std::ostream& green (std::ostream& s)
//------------------------------------------
{
  s << "\033[1;32m";
  return s;
}


//-------------------------------------------
inline std::ostream& yellow (std::ostream& s)
//-------------------------------------------
{
  s << "\033[1;33m";
  return s;
}


//-----------------------------------------
inline std::ostream& blue (std::ostream& s)
//-----------------------------------------
{
  s << "\033[1;34m";
  return s;
}

 
//--------------------------------------------
inline std::ostream& magenta (std::ostream& s)
//--------------------------------------------
{
  s << "\033[1;35m";
  return s;
}


//-----------------------------------------
inline std::ostream& cyan (std::ostream& s)
//-----------------------------------------
{
  s << "\033[1;36m";
  return s;
}


//------------------------------------------
inline std::ostream& white (std::ostream& s)
//------------------------------------------
{
  s << "\033[1;37m";
  return s;
}


//---------------------------------------------
inline std::ostream& bg_black (std::ostream& s)
//---------------------------------------------
{
  s << "\033[1;40m";
  return s;
}


//-------------------------------------------
inline std::ostream& bg_red (std::ostream& s)
//-------------------------------------------
{
  s << "\033[1;41m";
  return s;
}


//---------------------------------------------
inline std::ostream& bg_green (std::ostream& s)
//---------------------------------------------
{
  s << "\033[1;42m";
  return s;
}


//----------------------------------------------
inline std::ostream& bg_yellow (std::ostream& s)
//----------------------------------------------
{
  s << "\033[1;43m";
  return s;
}


//--------------------------------------------
inline std::ostream& bg_blue (std::ostream& s)
//--------------------------------------------
{
  s << "\033[1;44m";
  return s;
}


//-----------------------------------------------
inline std::ostream& bg_magenta (std::ostream& s)
//-----------------------------------------------
{
  s << "\033[1;45m";
  return s;
}


//--------------------------------------------
inline std::ostream& bg_cyan (std::ostream& s)
//--------------------------------------------
{
  s << "\033[1;46m";
  return s;
}


//---------------------------------------------
inline std::ostream& bg_white (std::ostream& s)
//---------------------------------------------
{
  s << "\033[1;47m";
  return s;
}


//------------------------------------------
inline std::ostream& clear (std::ostream& s)
//------------------------------------------
{
  s << "\033[2J" << "\033[H";
  return s;
}


#endif
