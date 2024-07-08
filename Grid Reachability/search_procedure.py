# MIN_BOUND = 10
# MAX_BOUND = 100
import numpy as np

from queries import Queries
from collections import deque

prec = 11

class SearchProcedure:

	def __init__(self):
		# mapping of key (4-tuple of pairs) SOURCE to list of 4-tuples (in 1 step) TARGETS
		self.dict_mapping_source_to_target = {}
		# mapping of key (4-tuple of pairs) TARGET to list of 4-tuples (in 1 step) SOURCES
		self.dict_mapping_target_to_source = {}
	'''
	parameters:
	patch_length: float, grid square length on both x and y
	input_region: tuple ((float,float),(float,float)), input region to grid up. The two tuples represent bound on the
	X and Y locations respectively.
	velocity_bounds: (float,float), velocity range to consider

	return:
	list of tuples ((float,float),(float,float),(float,float),(float,float))
	for bounds on each grid square
	'''
	def generate_grid_bounds_with_bounds(self, input_region, x_bound, y_bound, v_x_bound, v_y_bound):
		assert len(input_region) == 4
		#assert 10 < patch_length <= 100

		generated_bounds = []

		x_input_region, y_input_region, v_x_input_region, v_y_input_region = input_region
		
		for x_lb in np.arange(x_input_region[0], x_input_region[1], x_bound):
			x_ub = min(x_lb + x_bound, x_input_region[1])
			#x_ub = x_lb + x_bound
			for y_lb in np.arange(y_input_region[0], y_input_region[1], y_bound):
				y_ub = min(y_lb + y_bound, y_input_region[1])
				#y_ub = y_lb + y_bound
				for v_x_lb in np.arange(v_x_input_region[0], v_x_input_region[1], v_x_bound):
					v_x_ub = min(v_x_lb + v_x_bound, v_x_input_region[1])
					#v_x_ub = v_x_lb + v_x_bound
					for v_y_lb in np.arange(v_y_input_region[0], v_y_input_region[1], v_y_bound):
						v_y_ub = min(v_y_lb + v_y_bound, v_y_input_region[1])
						#v_y_ub = v_y_lb + v_y_bound

						single_patch = ((x_lb, x_ub), (y_lb, y_ub), (v_x_lb, v_x_ub), (v_y_lb, v_y_ub))
						generated_bounds.append(single_patch)

		return generated_bounds

	def generate_grid_bounds_in_range(self, input_region, x_bound, y_bound, v_x_bound, v_y_bound, range_region):
		assert len(input_region) == 4
		# assert 10 < patch_length <= 100

		generated_bounds = []

		x_input_region, y_input_region, v_x_input_region, v_y_input_region = input_region
		x_range_region, y_range_region, v_x_range_region, v_y_range_region = range_region

		for x_lb in np.arange(x_input_region[0], x_input_region[1], x_bound):
			if x_lb >= x_range_region[1]:
				continue
			x_ub = min(x_lb + x_bound, x_input_region[1])
			# x_ub = x_lb + x_bound
			for y_lb in np.arange(y_input_region[0], y_input_region[1], y_bound):
				if y_lb >= y_range_region[1]:
					continue
				y_ub = min(y_lb + y_bound, y_input_region[1])
				# y_ub = y_lb + y_bound
				for v_x_lb in np.arange(v_x_input_region[0], v_x_input_region[1], v_x_bound):
					if v_x_lb >= v_x_range_region[1]:
						continue
					v_x_ub = min(v_x_lb + v_x_bound, v_x_input_region[1])
					# v_x_ub = v_x_lb + v_x_bound
					for v_y_lb in np.arange(v_y_input_region[0], v_y_input_region[1], v_y_bound):
						if v_y_lb >= v_y_range_region[1]:
							continue
						v_y_ub = min(v_y_lb + v_y_bound, v_y_input_region[1])
						# v_y_ub = v_y_lb + v_y_bound

						single_patch = ((x_lb, x_ub), (y_lb, y_ub), (v_x_lb, v_x_ub), (v_y_lb, v_y_ub))
						generated_bounds.append(single_patch)

		return generated_bounds


	def generate_grid_bounds_with_search(self, input_region, coordinate, precision):
		#note we must show beforehand via induction we stay in the velocity bounds for this process to be correct
		assert len(input_region) == 4
		#assert  0 < patch_length <= 500

		queries = Queries()

		position = None
		#generate patch length for x,y,vx,vy.
		if coordinate == 'x':
			position = 0
		elif coordinate == 'y':
			position = 1
		elif coordinate == 'v_x':
			position = 2
		elif coordinate == 'v_y':
			position = 3

		range = input_region[position][1] - input_region[position][0]
		lower = 0
		upper = range

		mid = round((upper + lower) / 2,prec-1)

		while (upper - lower > precision):
			print(upper, lower)
			print(upper-lower)
			mid = round((upper + lower) / 2,prec-1)

			#ensure no timeout
			is_unsat = queries.query_bounds(input_region,coordinate,mid,200,1)

			if is_unsat == 1: # unsat
				upper = mid
			elif is_unsat == 0: # sat
				lower = mid
			else: # if T.O./M.O./error -> crash
				print("error")
				assert False
		return upper

	def generate_grid_fine_velocity(self, input_region, precision):
		#note we must show beforehand via induction we stay in the velocity bounds for this process to be correct
		assert len(input_region) == 4
		#assert  0 < patch_length <= 500

		queries = Queries()

		position = None

		#take v_x bound
		range = input_region[2][1]
		lower = 0
		upper = range

		mid = round((upper + lower) / 2, prec-1)

		while (upper - lower > precision):
			print(upper, lower)
			print(upper-lower)
			mid = round((upper + lower) / 2, prec-1)

			#find
			is_unsat = queries.check_fine_velocity(input_region, mid, 200, 1)

			if is_unsat == 1: # unsat
				lower = mid
			elif is_unsat == 0: # sat
				upper = mid
			else: # if T.O./M.O./error -> crash
				print("error")
				assert False
		return lower


	def find_adjacent_cells(self, list_of_bounds_for_current_cell, x_bound, y_bound, v_x_bound, v_y_bound, input_region):
		"""
		list_of_bounds_for_current_cell: a list of 4 X (float, float) tuples undicating the bounds
		of the current cell
		patch_length: float, grid square length on both x and y
		input_region: tuple ((float,float),(float,float)), input region to grid up. The two tuples represent bound on the
		X and Y locations respectively.
		velocity_bounds: (float,float), velocity range to consider

		return:
		list of inner-lists of tuples ((float,float),(float,float),(float,float),(float,float))
		with each inner-list representing bounds on each grid square
		"""
		assert len(input_region) == 4
		#assert 0 < patch_length <= 500

		x_lb_of_cell, x_ub_of_cell = list_of_bounds_for_current_cell[0]
		y_lb_of_cell, y_ub_of_cell = list_of_bounds_for_current_cell[1]
		v_x_lb_of_cell, v_x_ub_of_cell = list_of_bounds_for_current_cell[2]
		v_y_lb_of_cell, v_y_ub_of_cell = list_of_bounds_for_current_cell[3]

		list_of_adjacent_cells = []
		x_input_region, y_input_region, v_x_input_region, v_y_input_region = input_region
		for candidate_x_lb in np.arange(x_input_region[0], x_input_region[1], x_bound):
			candidate_x_ub = min(candidate_x_lb + x_bound, x_input_region[1])
			#candidate_x_ub = candidate_x_lb + x_bound
			for candidate_y_lb in np.arange(y_input_region[0], y_input_region[1], y_bound):
				candidate_y_ub = min(candidate_y_lb + y_bound, y_input_region[1])
				#candidate_y_ub = candidate_y_lb + y_bound
				for candidate_v_x_lb in np.arange(v_x_input_region[0], v_x_input_region[1], v_x_bound):
					candidate_v_x_ub = min(candidate_v_x_lb + v_x_bound, v_x_input_region[1])
					#candidate_v_x_ub = candidate_v_x_lb + v_x_bound
					for candidate_v_y_lb in np.arange(v_y_input_region[0], v_y_input_region[1], v_y_bound):
						candidate_v_y_ub = min(candidate_v_y_lb + v_y_bound, v_y_input_region[1])
						#candidate_v_y_ub = candidate_v_y_lb + v_y_bound

						candidate_cell_bounds = ((candidate_x_lb, candidate_x_ub), (candidate_y_lb, candidate_y_ub), (candidate_v_x_lb, candidate_v_x_ub), (candidate_v_y_lb, candidate_v_y_ub))

						if candidate_cell_bounds == list_of_bounds_for_current_cell:
							continue

						adjacent_in_x = x_lb_of_cell <= candidate_x_lb <= x_ub_of_cell or x_lb_of_cell <= candidate_x_ub <= x_ub_of_cell
						adjacent_in_y = y_lb_of_cell <= candidate_y_lb <= y_ub_of_cell or y_lb_of_cell <= candidate_y_ub <= y_ub_of_cell
						adjacent_in_v_x = v_x_lb_of_cell <= candidate_v_x_lb <= v_x_ub_of_cell or v_x_lb_of_cell <= candidate_v_x_ub <= v_x_ub_of_cell
						adjacent_in_v_y = v_y_lb_of_cell <= candidate_v_y_lb <= v_y_ub_of_cell or v_y_lb_of_cell <= candidate_v_y_ub <= v_y_ub_of_cell

						found_adjacent_cell = adjacent_in_x and adjacent_in_y and adjacent_in_v_x and adjacent_in_v_y
						if found_adjacent_cell:
							list_of_adjacent_cells.append(candidate_cell_bounds)

		return list_of_adjacent_cells


	def check_possible_transition_between_cells(self, current_cell, patch_length, input_region, velocity_bounds):
		list_of_adjacent_cells = self.find_adjacent_cells(list_of_bounds_for_current_cell=current_cell, patch_length=patch_length, input_region=input_region, velocity_bounds=velocity_bounds)
		for adjacent_cell_target in list_of_adjacent_cells:
			queries = Queries()
			if queries.transitioncheck(current_cell,adjacent_cell_target,200,1):
				# TODO - implement can_transition() - return TRUE if transition possible, FALSE otherwise
				if tuple(current_cell) in self.dict_mapping_source_to_target:
					self.dict_mapping_source_to_target[tuple(current_cell)].append(tuple(adjacent_cell_target))
				else:
					self.dict_mapping_source_to_target[tuple(current_cell)] = [tuple(adjacent_cell_target)]

				if tuple(adjacent_cell_target) in self.dict_mapping_target_to_source:
					self.dict_mapping_target_to_source[tuple(adjacent_cell_target)].append(tuple(current_cell))
				else:
					self.dict_mapping_target_to_source[tuple(adjacent_cell_target)] = [tuple(current_cell)]

	def generate_grid(self, input_region, precision=0.01):
		x_bound = self.generate_grid_bounds_with_search(input_region, 'x', precision)

		y_bound = self.generate_grid_bounds_with_search(input_region, 'y', precision)

		v_x_bound = self.generate_grid_bounds_with_search(input_region, 'v_x', precision)

		v_y_bound = self.generate_grid_bounds_with_search(input_region, 'v_y', precision)

		return x_bound, y_bound, v_x_bound, v_y_bound

	def generate_reachable_from_zero(self, input_region, x_bound, y_bound, v_x_bound, v_y_bound):

		assert len(input_region) == 4
		# assert 10 < patch_length <= 100

		generated_bounds = []

		x_input_region, y_input_region, v_x_input_region, v_y_input_region = input_region

		for x_lb in np.arange(x_input_region[0], x_input_region[1], x_bound):
			x_ub = min(x_lb + x_bound, x_input_region[1])
			# x_ub = x_lb + x_bound
			for y_lb in np.arange(y_input_region[0], y_input_region[1], y_bound):
				y_ub = min(y_lb + y_bound, y_input_region[1])
				# y_ub = y_lb + y_bound

				single_patch = ((x_lb, x_ub), (y_lb, y_ub), (0,0), (0,0))
				generated_bounds.append(single_patch)

		#grid cells with zero velocity
		zero_cells = set(generated_bounds)

		#bound on velocity containing zero,
		v_x_bounds_containing_zero, v_y_bounds_containing_zero = None, None
		for v_x_lb in np.arange(v_x_input_region[0], v_x_input_region[1], v_x_bound):
			v_x_ub = min(v_x_lb + v_x_bound, v_x_input_region[1])
			if v_x_lb <= 0 <= v_x_ub:
				v_x_bounds_containing_zero = (v_x_lb, v_x_ub)


		for v_y_lb in np.arange(v_y_input_region[0], v_y_input_region[1], v_y_bound):
			v_y_ub = min(v_y_lb + v_y_bound, v_y_input_region[1])
			if v_y_lb <= 0 <= v_y_ub:
				v_y_bounds_containing_zero = (v_y_lb, v_y_ub)

		assert v_x_bounds_containing_zero != None and v_y_bounds_containing_zero != None


		queries = Queries()
		#generate cells reachable from zero velocity in one step
		reachable = set()
		for cell in zero_cells:
			containing_cell = (cell[0],cell[1],v_x_bounds_containing_zero,v_y_bounds_containing_zero)
			candidate_cells = set(self.find_adjacent_cells(containing_cell, x_bound, y_bound, v_x_bound, v_y_bound, input_region))
			candidate_cells.add(containing_cell)
			for candidate_cell in candidate_cells:
				is_sat = queries.transitioncheck(cell, candidate_cell, 100, 1)
				if is_sat:
					reachable.add(candidate_cell)

		#initially everything reachable in 1 step from 0 velocity
		f = open("reachableMappings.txt", "w")
		candidates = reachable.copy()
		while True:
			added_cells = set()
			for reachable_cell in candidates:
				for adj_cell in self.find_adjacent_cells(reachable_cell, x_bound, y_bound, v_x_bound, v_y_bound, input_region):
					if adj_cell in reachable:
						continue
					is_sat = queries.transitioncheck(reachable_cell, adj_cell, 100, 1)
					if is_sat:
						f.write(str(reachable_cell) + " -> " + str(adj_cell) + "\n")
						added_cells.add(adj_cell)
						reachable.add(adj_cell)

			if len(added_cells) == 0:
				break

			candidates = added_cells

		f.close()
		print(reachable)
		print(len(reachable))
		return reachable

	def check_from_zero_one_step(self, input_region, x_bound, y_bound, v_x_bound, v_y_bound):

		assert len(input_region) == 4
		# assert 10 < patch_length <= 100

		right_patch = None

		x_input_region, y_input_region, v_x_input_region, v_y_input_region = input_region

		for x_lb in np.arange(x_input_region[0], x_input_region[1], x_bound):
			x_ub = min(x_lb + x_bound, x_input_region[1])
			# x_ub = x_lb + x_bound
			for y_lb in np.arange(y_input_region[0], y_input_region[1], y_bound):
				y_ub = min(y_lb + y_bound, y_input_region[1])
				# y_ub = y_lb + y_bound

				single_patch = ((x_lb, x_ub), (y_lb, y_ub), (0,0), (0,0))
				if x_ub == 5 and y_ub == 5:
					right_patch = single_patch

		#bound on velocity containing zero,
		v_x_bounds_containing_zero, v_y_bounds_containing_zero = None, None
		for v_x_lb in np.arange(v_x_input_region[0], v_x_input_region[1], v_x_bound):
			v_x_ub = min(v_x_lb + v_x_bound, v_x_input_region[1])
			if v_x_lb <= 0 <= v_x_ub:
				v_x_bounds_containing_zero = (v_x_lb, v_x_ub)


		for v_y_lb in np.arange(v_y_input_region[0], v_y_input_region[1], v_y_bound):
			v_y_ub = min(v_y_lb + v_y_bound, v_y_input_region[1])
			if v_y_lb <= 0 <= v_y_ub:
				v_y_bounds_containing_zero = (v_y_lb, v_y_ub)

		assert v_x_bounds_containing_zero != None and v_y_bounds_containing_zero != None


		queries = Queries()
		#generate cells reachable from zero velocity in one step
		file = open("reachableFromZero.txt", "w")
		all_cells = deque()
		visited_cells = set()
		containing_cell = (right_patch[0], right_patch[1], v_x_bounds_containing_zero, v_y_bounds_containing_zero)
		is_sat = queries.transitioncheck(right_patch, containing_cell, 100, 1)
		one_step_reach = set()
		if is_sat:
			all_cells.append(containing_cell)
			one_step_reach.add(containing_cell)
			visited_cells.add(containing_cell)

		checking_cells = self.find_adjacent_cells(containing_cell, x_bound, y_bound, v_x_bound, v_y_bound, input_region)
		checking_cells.append(containing_cell)
		for cell in checking_cells:
			is_sat = queries.transitioncheck(right_patch, cell, 100, 1)
			if is_sat:
				one_step_reach.add(cell)
				if cell not in visited_cells:
					all_cells.append(cell)
					visited_cells.add(cell)
		file.write(str(right_patch) + " -> " + str(one_step_reach) + "\n\n")

		while len(all_cells) != 0:
			cur_cell = all_cells.popleft()
			one_step_reach = set()
			candidate_cells = self.find_adjacent_cells(cur_cell, x_bound, y_bound, v_x_bound, v_y_bound, input_region)
			candidate_cells.append(cur_cell)
			for candidate_cell in candidate_cells:
				is_sat = queries.transitioncheck(cur_cell, candidate_cell, 100, 1)
				if is_sat:
					one_step_reach.add(candidate_cell)
					if candidate_cell not in visited_cells:
						all_cells.append(candidate_cell)
						visited_cells.add(candidate_cell)
			file.write(str(cur_cell) + " -> " + str(one_step_reach) + "\n\n")
		file.close()
		print(len(visited_cells))

	def do_docking_search(self, input_region):
		#divide docking region to put into our d set
		x_bound, y_bound, v_x_bound, v_y_bound = self.generate_grid(input_region)

		docking_region = [(0, 0.5), (0, 0.5), (-1.6, 1.6), (-1.6, 1.6)]

		d = set(self.generate_grid_bounds_in_range(docking_region, x_bound, y_bound, v_x_bound, v_y_bound, docking_region))

		s = deque()
		for element in d:
			s.append(element)

		queries = Queries()

		while len(s) != 0:
			e = s.popleft()
			for adj in self.find_adjacent_cells(e, x_bound, y_bound, v_x_bound, v_y_bound, input_region):
				if adj in d:
					continue
				is_unsat = queries.always_reaches(e, adj, 100, 1)
				if is_unsat:
					d.add(adj)
					#s.append(adj)

		return d


	def create_mapping(self, start_point, input_region):
		x_bound, y_bound, v_x_bound, v_y_bound = self.generate_grid(input_region)
		small_v_bound = self.generate_grid_fine_velocity(input_region, 0.0001)

		#assert that x_bound, y_bound, v_x_bound, v_y_bound, small_v_bound are all less than input region

		visited = set()
		expand = deque()

		transition_outside = False

		#nested loop here to loop over two positions for x and y (lower,upper) and velocities (lower,upper,middle)
		x_options = [round(start_point[0] - x_bound/2 - x_bound, prec), round(start_point[0] - x_bound/2, prec), round(start_point[0] + x_bound/2,prec)]
		y_options = [round(start_point[1] - y_bound/2 - y_bound,prec), round(start_point[1] - y_bound/2,prec), round(start_point[1] + y_bound/2,prec)]
		v_x_options = [round(-v_x_bound-small_v_bound,prec),-small_v_bound,small_v_bound]
		v_y_options = [round(-v_y_bound-small_v_bound,prec),-small_v_bound,small_v_bound]

		'''
		print(x_bound,y_bound,v_x_bound,v_y_bound,small_v_bound)
		print(x_options)
		print(y_options)
		print(v_x_options)
		print(v_y_options)
		quit()
		'''


		for x_option in x_options:
			x_bounds = (x_option,round(x_option+x_bound,prec))
			for y_option in y_options:
				y_bounds = (y_option,round(y_option+y_bound,prec))
				for v_x_option in v_x_options:
					if v_x_option == -small_v_bound:
						v_x_bounds = (v_x_option, small_v_bound)
					else:
						v_x_bounds = (v_x_option, round(v_x_option + v_x_bound,prec))
					for v_y_option in v_y_options:
						if v_y_option == -small_v_bound:
							v_y_bounds = (v_y_option, small_v_bound)
						else:
							v_y_bounds = (v_y_option, round(v_y_option + v_y_bound,prec))
						expand.append((x_bounds,y_bounds,v_x_bounds,v_y_bounds))

		#first step, see what we can visit
		queries = Queries()
		#turn starting point into cell (assumption, none of our initial candidates are outside the region)
		starting_cell = ((start_point[0],start_point[0]),(start_point[1],start_point[1]),(start_point[2],start_point[2]),(start_point[3],start_point[3]))
		starting_cells = deque()
		for candidate in expand:
			can_transition = queries.transitioncheck(starting_cell, candidate, 100, 1)
			if can_transition:
				starting_cells.append(candidate)
				visited.add(candidate)

		file = open("reachableFromZero.txt", "w")
		file.write("checking docking for:" + str(start_point) + "\n\n")
		file.write("initial set of reachable cells:" + str(starting_cells) + "\n\n")

		expand = starting_cells
		while len(expand) != 0:
			cur_cell = expand.popleft()
			one_step_reach = set()
			adjacent_cells = self.find_next_cells(x_bound,y_bound,v_x_bound, v_y_bound,small_v_bound,cur_cell,input_region)

			#file.write("adjacent cells:" + str(adjacent_cells) + "\n\n")
			for adj_cell in adjacent_cells:
				can_transition = queries.transitioncheck(cur_cell, adj_cell, 100, 1)
				if can_transition:
					one_step_reach.add(adj_cell)
					if (adj_cell[0][0] < input_region[0][0] or adj_cell[0][1] > input_region[0][1] or adj_cell[1][0] < input_region[1][0] or adj_cell[1][1] > input_region[1][1]):
						if not transition_outside:
							transition_outside = True
					else:
						if adj_cell not in visited and (adj_cell[0][1] < -0.5 or adj_cell[0][0] > 0.5 or adj_cell[1][1] < -0.5 or adj_cell[1][0] > 0.5):
							visited.add(adj_cell)
							expand.append(adj_cell)
			file.write(str(cur_cell) + " -> " + str(one_step_reach) + "\n\n")

		file.write("Did go outside input region of " + str(input_region) + " :" + str(transition_outside) + "\n\n")
		file.close()

	def find_next_cells(self,x_bound,y_bound,v_x_bound,v_y_bound,small_v_bound,cur_cell,input_region):
		x_options = [round(cur_cell[0][0] - x_bound,prec), cur_cell[0][0], cur_cell[0][1]]
		y_options = [round(cur_cell[1][0] - y_bound,prec), cur_cell[1][0], cur_cell[1][1]]


		if cur_cell[2][0] - v_x_bound < 0 and cur_cell[2][1] + v_x_bound > 0:
			v_x_options = [round(-small_v_bound - v_x_bound,prec), -small_v_bound, small_v_bound]

			if cur_cell[2][0] != -small_v_bound or cur_cell[2][1] != small_v_bound:
				if cur_cell[2][1] == -small_v_bound:
					v_x_options.append(round(cur_cell[2][0] - v_x_bound,prec))
				else:
					v_x_options.append(cur_cell[2][1])
		else:
			v_x_options = [cur_cell[2][0]]

			if cur_cell[2][0] != input_region[2][0]:
				v_x_options.append(max(round(cur_cell[2][0]-v_x_bound,prec), input_region[2][0]))

			if cur_cell[2][1] != input_region[2][1]:
				v_x_options.append(cur_cell[2][1])

		if cur_cell[3][0] - v_y_bound < 0 and cur_cell[3][1] + v_y_bound > 0:
			v_y_options = [round(-small_v_bound- v_y_bound,prec), -small_v_bound, small_v_bound]

			if cur_cell[3][0] != -small_v_bound or cur_cell[3][1] != small_v_bound:
				if cur_cell[3][1] == -small_v_bound:
					v_y_options.append(round(cur_cell[3][0] - v_y_bound,prec))
				else:
					v_y_options.append(cur_cell[3][1])
		else:
			v_y_options = [cur_cell[3][0]]

			if cur_cell[3][0] != input_region[3][0]:
				v_y_options.append(max(round(cur_cell[3][0]-v_y_bound,prec), input_region[3][0]))

			if cur_cell[3][1] != input_region[3][1]:
				v_y_options.append(cur_cell[3][1])

		'''
		print(x_bound,y_bound,v_x_bound,v_y_bound,small_v_bound)
		print(x_options)
		print(y_options)
		print(v_x_options)
		print(v_y_options)
		'''

		expand = set()
		for x_option in x_options:
			x_bounds = (x_option, round(x_option + x_bound,prec))
			for y_option in y_options:
				y_bounds = (y_option, round(y_option + y_bound,prec))
				for v_x_option in v_x_options:
					if v_x_option == -small_v_bound:
						v_x_bounds = (v_x_option, small_v_bound)
					else:
						if v_x_option == input_region[2][0]:
							if v_x_option == cur_cell[2][0]:
								v_x_bounds = (v_x_option, cur_cell[2][1])
							else:
								v_x_bounds = (v_x_option, cur_cell[2][0])
						else:
							v_x_bounds = (v_x_option, min(round(v_x_option + v_x_bound,prec), input_region[2][1]))
					for v_y_option in v_y_options:
						if v_y_option == -small_v_bound:
							v_y_bounds = (v_y_option, small_v_bound)
						else:
							if v_y_option == input_region[3][0]:
								if v_y_option == cur_cell[3][0]:
									v_y_bounds = (v_y_option, cur_cell[3][1])
								else:
									v_y_bounds = (v_y_option, cur_cell[3][0])
							else:
								v_y_bounds = (v_y_option, min(round(v_y_option + v_y_bound,prec), input_region[3][1]))
						expand.add((x_bounds, y_bounds, v_x_bounds, v_y_bounds))

		return expand


if __name__ == '__main__':
	input_region = [(-10, 10), (-10, 10), (-1.6,  1.6), (-1.6, 1.6)]
	searchProcedure = SearchProcedure()

	searchProcedure.create_mapping([5,5,0,0], input_region)
	#x_bound, y_bound, v_x_bound, v_y_bound = searchProcedure.generate_grid(input_region)
	#print(x_bound, y_bound, v_x_bound, v_y_bound)

	#searchProcedure.check_from_zero_one_step(input_region, x_bound, y_bound, v_x_bound, v_y_bound)

	#v_bound = searchProcedure.generate_grid_fine_velocity(input_region, 0.001)
	#print(x_bound, y_bound, v_bound)
	#queries = Queries()
	#queries.check_cycle(input_region, 'positive', v_bound, 100)


	'''
	searchProcedure = SearchProcedure()
	x_bound = searchProcedure.generate_grid_bounds_with_search(input_region,'x',0.01)

	y_bound = searchProcedure.generate_grid_bounds_with_search(input_region, 'y', 0.01)

	v_x_bound = searchProcedure.generate_grid_bounds_with_search(input_region, 'v_x', 0.01)

	v_y_bound = searchProcedure.generate_grid_bounds_with_search(input_region, 'v_y', 0.01)
	print(x_bound)
	print(y_bound)
	print(v_x_bound)
	print(v_y_bound)

	#searchProcedure.generate_grid_bounds_with_bounds(input_region, x_bound, y_bound, v_x_bound, v_y_bound)


	print(generated_grid_bounds)

	cell_to_check = [(30, 40), (50, 60), (-1.6,  1.6), (-1.6, 1.6)]
	adjacent_cells = searchProcedure.find_adjacent_cells(
		list_of_bounds_for_current_cell=cell_to_check,
		patch_length=10,
		input_region=[(10, 100), (10, 100)],
		velocity_bounds=velocity_bounds)

	print(f"the cells adjacent to\n {cell_to_check}\n -> are:")
	print("*"*20)
	for res in adjacent_cells:
		print(res)

	cell_to_check = [(30, 40), (50, 60), (0,0), (0,0)]
	searchProcedure.check_possible_transition_between_cells(cell_to_check,patch_length,input_region,velocity_bounds)
	#print(tuple(cell_to_check))
	for res in searchProcedure.dict_mapping_source_to_target.get(tuple(cell_to_check)):
		print(res)
	'''
