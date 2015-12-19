import csv
import numpy as np
from random import shuffle, randint, random
from math import sqrt
import matplotlib.pyplot as pl

def euclidean_distance(x1, y1, x2, y2):
	return sqrt((x2 - x1)**2 + (y2 - y1)**2)


def generate_distance_matrix(coords):
	n = len(coords)
	ret = np.zeros([n, n])

	for i in range(len(coords)):
		for j in range(len(coords)):
			x = coords[i]
			y = coords[j]
			ret[i][j] = euclidean_distance(x[0], x[1], y[0], y[1])

	return ret


def energy(path, D):
	E = 0.0
	for i in range(len(path)-1):
		E += D[path[i]][path[i+1]]

	return E


def pi_ratio(E_old, E_new, T):
	return np.exp(-1.0 / T * (E_new - E_old))


def metropolis_algorithm(D, T, start, end, iterations):
	path_0 = range(len(D))
	path_0.remove(start)

	if start != end:
		path_0.remove(end)
	shuffle(path_0)
	path_0 = [start] + path_0 + [end]

	energy_0 = energy(path_0, D)

	for i in range(iterations):
		path_new = path_0[:]

		idx1 = randint(1, len(path_0) - 3)
		idx2 = randint(1, len(path_0) - 3)
		while idx1 == idx2:
			idx1 = randint(1, len(path_0) - 3)
			idx2 = randint(1, len(path_0) - 3)

		if idx1 > idx2:
			temp = idx2
			idx2 = idx1
			idx1 = temp

		path_new = path_new[:idx1] + list(reversed(path_new[idx1:idx2+1])) \
								   + path_new[idx2+1:]

		energy_new = energy(path_new, D)

		check = min(1.0, pi_ratio(energy_0, energy_new, T))

		u = random()

		if u < check:
			T = 0.9999 * T
			path_0 = path_new
			energy_0 = energy_new

	return path_0, energy_0


# header: x_corr, y_corr, town, county, state, county number

names = {}
numbers = []
locations = {}
county_names = {}
counties = {}

with open('CAtowns.csv', 'rb') as csvfile:
	csvread = csv.reader(csvfile, delimiter=',', quotechar='"')
	i = 0
	for row in csvread:
		if i != 0:
			locs = row[0:2]
			locs = [float(j) for j in locs]
			num = int(row[5])-1
			numbers.append(num)

			if num in counties:
				counties[num].append(locs)
				names[num].append(row[2])
				locations[num].append(locs)
			else:
				counties[num] = [locs]
				names[num] = [row[2]]
				county_names[num] = row[3]
				locations[num] = [locs]


		i += 1

all_counties = []

num_counties = max(numbers) + 1

county_path_matrix = np.zeros([num_counties, num_counties], dtype=(np.int32, (2,)))
for i in range(num_counties):
	coords = counties[i]
	all_counties.append(generate_distance_matrix(coords))

	for j in range(len(coords)):
		town = coords[j]
		for k in range(i + 1, num_counties):
			other_coords = counties[k]
			
			for m in range(len(other_coords)):
				other_town = other_coords[m]

				dist = euclidean_distance(town[0], town[1],
										  other_town[0], other_town[1])

				idx = county_path_matrix[i][k][0]
				other_idx = county_path_matrix[i][k][1]
				min_dist = euclidean_distance(coords[idx][0],
								coords[idx][1],
				            	other_coords[other_idx][0],
				            	other_coords[other_idx][1])

				if dist < min_dist:
					county_path_matrix[i][k] = [j, m]
					county_path_matrix[k][i] = [m, j]

dist_county_matrix = np.zeros([num_counties, num_counties])
for i in range(county_path_matrix.shape[0]):
	for j in range(i+1, county_path_matrix.shape[1]):
		two_towns = county_path_matrix[i][j]
		town1 = counties[i][two_towns[0]]
		town2 = counties[j][two_towns[1]]

		dist = euclidean_distance(town1[0], town1[1], town2[0], town2[1])

		dist_county_matrix[i][j] = dist
		dist_county_matrix[j][i] = dist

T = 50.0
num_iters = 1000000

county_path, county_cost = metropolis_algorithm(dist_county_matrix, T, 18, 18, num_iters)
cpath = [county_names[i] for i in county_path]
print cpath
print len(cpath)
print county_cost

# start_county = counties[county_path[0]]
# start_city = 0
# for i in range(len(start_county)):
# 	name = names[county_path[0]][i]
# 	if 'Pasadena' in name:
# 		start_city = i
# 		print 'found pasadena'

# total_cost = county_cost
# overall_x = []
# overall_y = []
# start = start_city
# for i in range(len(county_path)-1):
# 	county1 = county_path[i]
# 	county2 = county_path[i+1]
# 	two_towns = county_path_matrix[county1][county2]
# 	end = two_towns[0]

# 	town_path, town_cost = metropolis_algorithm(all_counties[county1], 2*T, start, end, num_iters)
# 	for town in town_path:
# 		loc = locations[county1][town]
# 		overall_x.append(loc[0])
# 		overall_y.append(loc[1])
# 	total_cost += town_cost
# 	print total_cost

# 	start = two_towns[1]

# 	transition_loc = locations[county2][start]
# 	overall_x.append(loc[0])
# 	overall_y.append(loc[1])

# pl.plot(overall_x, overall_y, color="blue", linestyle="-")
# pl.savefig("tour.png")
