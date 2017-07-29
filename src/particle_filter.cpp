/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;


void ParticleFilter::init(double x, double y, double theta, double std[]) 
{
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	
	num_particles = 100;

	default_random_engine gen;

	normal_distribution<double> N_x(x, std[0]);
	normal_distribution<double> N_y(y, std[1]);
	normal_distribution<double> N_theta(theta, std[2]);

	for (int i = 0; i < num_particles; i++) 
	{
		Particle particle;
		particle.x = N_x(gen);
		particle.y = N_y(gen);
		particle.theta = N_theta(gen);
		particle.weight = 1;

		particles.push_back(particle);
		weights.push_back(1);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) 
{
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find normal_distribution and default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	double x_new = 0;
	double y_new = 0;
	double theta_new = 0;

	default_random_engine gen;

	normal_distribution<double> N_x(0, std_pos[0]);
	normal_distribution<double> N_y(0, std_pos[1]);
	normal_distribution<double> N_theta(0, std_pos[2]);


	for (int i = 0; i < num_particles; i++) 
	{
		Particle particle = particles[i];
		if (yaw_rate != 0) 
		{
			x_new = particle.x + velocity * (1 / yaw_rate) * (-sin(particle.theta) \
				                                             + sin(particle.theta + yaw_rate * delta_t));

			y_new = particle.y + velocity * (1 / yaw_rate) * (cos(particle.theta) \
				                                            - cos(particle.theta + yaw_rate * delta_t));

			theta_new = particle.theta + yaw_rate * delta_t;
		} 
		else 
		{
			x_new = particle.x + velocity * delta_t * cos(particle.theta);
			y_new = particle.y + velocity * delta_t * sin(particle.theta);
			theta_new = particle.theta;
		}


		particles[i].x = x_new + N_x(gen);
		particles[i].y = y_new + N_y(gen);
		particles[i].theta = theta_new + N_theta(gen);

	}
	
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, \
	                                 vector<LandmarkObs>& observations) 
{
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	int num_observations;
	num_observations = observations.size();

	int num_maplandmarks;
	num_maplandmarks = predicted.size();

	for (int i = 0; i < num_observations; i++) 
	{
		double x = observations[i].x;
		double y = observations[i].y;

		int id;
		double min_distance = 50;

		for (int ii = 0; ii < num_maplandmarks; ii++) 
		{
			double xm = predicted[ii].x;
			double ym = predicted[ii].y;
			double temp_dist = dist(x, y, xm, ym);

			if (temp_dist < min_distance) 
			{
				id = predicted[ii].id;
				min_distance = temp_dist;
			}
		}

		observations[i].id = id;
	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], \
	                              vector<LandmarkObs> observations, Map map_landmarks) 
{
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html


	double sigma_x = std_landmark[0];
	double sigma_y = std_landmark[1];

	int num_maplandmarks = map_landmarks.landmark_list.size();


	for (int i = 0; i < num_particles; i++) 
	{
		vector<LandmarkObs> obs_abs;
		int num_obs = observations.size();

		Particle particle = particles[i];

		vector<LandmarkObs> obs_pred;

		int nn = 1;
		for (int ii = 0; ii < num_maplandmarks; ii++) 
		{
			LandmarkObs obs;
			obs.x = map_landmarks.landmark_list[ii].x_f;
			obs.y = map_landmarks.landmark_list[ii].y_f;

			if (dist(obs.x, obs.y, particle.x, particle.y) <= sensor_range) 
			{
				obs.id = nn;
				nn++;
				obs_pred.push_back(obs);
			}
		}


		for (int ii = 0; ii < num_obs; ii++) 
		{
			double xr = observations[ii].x;
			double yr = observations[ii].y;

			LandmarkObs measurement;

			measurement.x = xr * cos(particle.theta) - yr * sin(particle.theta) + particle.x;
			measurement.y = xr * sin(particle.theta) + yr * cos(particle.theta) + particle.y;

			obs_abs.push_back(measurement);
		}

		dataAssociation(obs_pred, obs_abs);

		double w = 1;

		for (int ii = 0; ii < num_obs; ii++) 
		{
			double x = obs_abs[ii].x;
			double y = obs_abs[ii].y;
			int id = obs_abs[ii].id;

			if (id > 0) 
			{

				double xm = obs_pred[id-1].x;
				double ym = obs_pred[id-1].y;
				
				double p = (1 / (2 * M_PI * sigma_x * sigma_y)) * \
				    exp(-(pow(x - xm, 2) / (2 * pow(sigma_x, 2)) + pow(y-ym, 2) / (2 * pow(sigma_y, 2))));
				w *= p;
			}
		}
		weights[i] = w;
	}
}

void ParticleFilter::resample() 
{
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    default_random_engine gen;
	discrete_distribution<int> distribution(weights.begin(), weights.end());

	vector<Particle> resample_particles;

	for (int i = 0; i < num_particles; i++) 
	{
		resample_particles.push_back(particles[distribution(gen)]);
	}	

	particles = resample_particles;
}


Particle ParticleFilter::SetAssociations(Particle particle, vector<int> associations, \
	                                     vector<double> sense_x, vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
