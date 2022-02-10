
//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
//
// Adapted for Low Level Parallel Programming 2017
//
#include "ped_model.h"
#include "ped_waypoint.h"
#include "ped_model.h"
#include <iostream>
#include <stack>
#include <algorithm>
#include "cuda_testkernel.h"
#include <omp.h>
#include <thread>
#include <emmintrin.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <algorithm>
#include <iterator>
#include <smmintrin.h>
void Ped::Model::setup(std::vector<Ped::Tagent*> agentsInScenario, std::vector<Twaypoint*> destinationsInScenario, IMPLEMENTATION implementation)
{
	// Convenience test: does CUDA work on this machine?
	cuda_test();
	// Set 
	agents = std::vector<Ped::Tagent*>(agentsInScenario.begin(), agentsInScenario.end());
	
	int size = agents.size();
	
	X = (float *) _mm_malloc((size + size % 4) * sizeof(float), 16);

	Y = (float *) _mm_malloc((size + size % 4) * sizeof(float), 16);
	
	int i = 0;
	for(auto agent : agents) {
		X[i] = (float)agent->getX();
		Y[i] = (float)agent->getY();
		agent->destination = agent->waypoints.front();
		agent->waypoints.pop_front();
		i++;
	}
	
	// Set up destinations
	
	destinations = std::vector<Ped::Twaypoint*>(destinationsInScenario.begin(), destinationsInScenario.end());
	int destSize = destinations.size();
	
	destX = (float *)_mm_malloc((size+ size % 4) * sizeof(float),16);
	destY = (float *)_mm_malloc((size+ size % 4) * sizeof(float),16);
	destR = (float *)_mm_malloc((size + size % 4) * sizeof(float),16);
	destXNext = (float *)_mm_malloc((size+ size % 4) * sizeof(float),16);
	destYNext = (float *)_mm_malloc((size+ size % 4) * sizeof(float),16);	
	destRNext = (float *)_mm_malloc((size+ size % 4) * sizeof(float),16);
		
	for(int i = 0; i < (size + size % 4); i++) {
		
		destX[i] = destXNext[i] = (float)destinations[i%destSize]->getx();
		destY[i] = destYNext[i] = (float)destinations[i%destSize]->gety();
		destR[i] = destRNext[i] = (float)destinations[i%destSize]->getr();
	}
	
	// Sets the chosen implemenation. Standard in the given code is SEQ
	this->implementation = implementation;

	// Set up heatmap (relevant for Assignment 4)
	setupHeatmapSeq();
}

void Ped::Model::thread_func(int val, int work){
  for(int i = 0; i < work; i++){
    int index = val + i;
    agents[index]->computeNextDesiredPosition();
    agents[index]->setX(agents[index]->getDesiredX());
    agents[index]->setY(agents[index]->getDesiredY());
  }
}

void print128_num(__m128 var)
{
    float val[4];
    memcpy(val, &var, sizeof(val));
    printf("Numerical: %f %f %f %f \n", 
           val[0], val[1], val[2], val[3]);
}

void Ped::Model::tick()
{
  if(this->implementation == Ped::SEQ){
      for (auto agent: agents){
	agent->computeNextDesiredPosition();
	agent->setX(agent->getDesiredX());
	agent->setY(agent->getDesiredY());
      }
    }
  else if(this->implementation == Ped::OMP){
    int i;
#pragma omp parallel for private(i)
    for (i = 0; i < agents.size(); i++){
      agents[i]->computeNextDesiredPosition();
      agents[i]->setX(agents[i]->getDesiredX());
      agents[i]->setY(agents[i]->getDesiredY());
    } 
  }
  else if(this->implementation == Ped::PTHREAD){
  
     int num_threads = 1;
     if(num_threads > agents.size()){
       num_threads = agents.size();
     }
     std::thread threads[num_threads];
     int num_agents = agents.size();
     int work = num_agents / num_threads;
     int remainder = num_agents % num_threads;
     for(int i = 0; i < num_threads && num_agents; i++){
       int index = i * work;
       threads[i] = std::thread(&Ped::Model::thread_func,this, index, work);
     }
     thread_func(num_agents-remainder,remainder);
     for(int i = 0; i < num_threads; i++){
       threads[i].join();
     }
   }
   else if(this->implementation == Ped::VECTOR){
	   __m128 Xd,Yd, Xs,Ys, len, mask_rad, mask_zero, corr, Rd, Xn, Yn, Xnd, Ynd, Xds, Yds;
	   __m128 zeros = _mm_setzero_ps();
	   __m128 ones = _mm_set1_ps(1);

	   for (int i = 0; i < agents.size(); i+=4)
	   {	

		   	Xs = _mm_load_ps(&X[i]);
		   	Ys = _mm_load_ps(&Y[i]);
			Xds = _mm_load_ps(&destX[i]);
			Yds = _mm_load_ps(&destY[i]);
			Rd = _mm_load_ps(&destR[i]);
				
			Xd = _mm_sub_ps(Xds, Xs);
			Yd = _mm_sub_ps(Yds, Ys);
		        
		       
			len = _mm_sqrt_ps(_mm_add_ps(_mm_mul_ps(Xd, Xd),_mm_mul_ps(Yd, Yd)));
			// If mask==1 len < rad
			mask_rad = _mm_cmplt_ps(len,Rd);
			//cout << "-----------\n";
			//print128_num(mask_rad);
			//print128_num(len);			
			mask_zero = _mm_cmpeq_ps(len,zeros);
			corr = _mm_blendv_ps(zeros,ones,mask_zero);
			len = _mm_add_ps(len,corr);

			Xd = _mm_div_ps(Xd,len);
			Yd = _mm_div_ps(Yd,len);
			Xd = _mm_add_ps(Xd,Xs);
			Yd = _mm_add_ps(Yd,Ys);
			
		    							//mask!=1,mask==1,mask	
				
			Xn = _mm_blendv_ps(Xd,Xs,mask_rad);
			Yn = _mm_blendv_ps(Yd,Ys,mask_rad);

			Xn = _mm_round_ps(Xn, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
			Yn = _mm_round_ps(Yn, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
			_mm_store_ps(&X[i], Xn);
			_mm_store_ps(&Y[i], Yn);
			

			Xnd = _mm_load_ps(&destXNext[i]);	
			Ynd = _mm_load_ps(&destYNext[i]);
			//print128_num(Xds);
			//print128_num(Xnd);
			Xn = _mm_blendv_ps(Xds,Xnd,mask_rad);
			Yn = _mm_blendv_ps(Yds,Ynd,mask_rad);
			
			_mm_store_ps(&destX[i], Xn);
			_mm_store_ps(&destY[i], Yn);
			
	   }
	   int j = 0; 
	   for (auto agent : agents){
		   agent->setX((int)round(X[j]));
		   agent->setY((int)round(Y[j]));
		   if(destX[j] == destXNext[j] && destY[j] == destYNext[j]){
			Twaypoint *dest = agent->destination;
			agent->waypoints.push_back(dest);
			agent->destination = dest = agent->waypoints.front();
			agent->waypoints.pop_front();
			//cout << "dest :" << dest->getx() << "\n";
			destYNext[j] = dest->gety();
			destXNext[j] = dest->getx();
			destRNext[j] = dest->getr();			
	   		}
		   j++;
	   }
   }
}



////////////
/// Everything below here relevant for Assignment 3.
/// Don't use this for Assignment 1!
///////////////////////////////////////////////

// Moves the agent to the next desired position. If already taken, it will
// be moved to a location close to it.
void Ped::Model::move(Ped::Tagent *agent)
{
	// Search for neighboring agents
	set<const Ped::Tagent *> neighbors = getNeighbors(agent->getX(), agent->getY(), 2);

	// Retrieve their positions
	std::vector<std::pair<int, int> > takenPositions;
	for (std::set<const Ped::Tagent*>::iterator neighborIt = neighbors.begin(); neighborIt != neighbors.end(); ++neighborIt) {
		std::pair<int, int> position((*neighborIt)->getX(), (*neighborIt)->getY());
		takenPositions.push_back(position);
	}

	// Compute the three alternative positions that would bring the agent
	// closer to his desiredPosition, starting with the desiredPosition itself
	std::vector<std::pair<int, int> > prioritizedAlternatives;
	std::pair<int, int> pDesired(agent->getDesiredX(), agent->getDesiredY());
	prioritizedAlternatives.push_back(pDesired);

	int diffX = pDesired.first - agent->getX();
	int diffY = pDesired.second - agent->getY();
	std::pair<int, int> p1, p2;
	if (diffX == 0 || diffY == 0)
	{
		// Agent wants to walk straight to North, South, West or East
		p1 = std::make_pair(pDesired.first + diffY, pDesired.second + diffX);
		p2 = std::make_pair(pDesired.first - diffY, pDesired.second - diffX);
	}
	else {
		// Agent wants to walk diagonally
		p1 = std::make_pair(pDesired.first, agent->getY());
		p2 = std::make_pair(agent->getX(), pDesired.second);
	}
	prioritizedAlternatives.push_back(p1);
	prioritizedAlternatives.push_back(p2);

	// Find the first empty alternative position
	for (std::vector<pair<int, int> >::iterator it = prioritizedAlternatives.begin(); it != prioritizedAlternatives.end(); ++it) {

		// If the current position is not yet taken by any neighbor
		if (std::find(takenPositions.begin(), takenPositions.end(), *it) == takenPositions.end()) {

			// Set the agent's position 
			agent->setX((*it).first);
			agent->setY((*it).second);

			break;
		}
	}
}

/// Returns the list of neighbors within dist of the point x/y. This
/// can be the position of an agent, but it is not limited to this.
/// \date    2012-01-29
/// \return  The list of neighbors
/// \param   x the x coordinate
/// \param   y the y coordinate
/// \param   dist the distance around x/y that will be searched for agents (search field is a square in the current implementation)
set<const Ped::Tagent*> Ped::Model::getNeighbors(int x, int y, int dist) const {

	// create the output list
	// ( It would be better to include only the agents close by, but this programmer is lazy.)	
	return set<const Ped::Tagent*>(agents.begin(), agents.end());
}

void Ped::Model::cleanup() {
	// Nothing to do here right now. 
}

Ped::Model::~Model()
{
	std::for_each(agents.begin(), agents.end(), [](Ped::Tagent *agent){delete agent;});
	std::for_each(destinations.begin(), destinations.end(), [](Ped::Twaypoint *destination){delete destination; });
}
