import json


def sudoku_reference():
	"""Gets a tuple of reference objects that are useful for describing the Sudoku grid."""

	def cross(vector_a, vector_b):
		"""Cross product of two vectors A and B, concatenating strings."""
		return [a + b for a in vector_a for b in vector_b]

	all_rows = 'ABCDEFGHI'
	all_cols = '123456789'

	# Build up list of all cell positions on the grid
	coords = cross(all_rows, all_cols)
	# print(len(coords))  # 81

	row_units = [cross(row, all_cols) for row in all_rows]

	col_units = [cross(all_rows, col) for col in all_cols]

	box_units = [cross(row_square, col_square) for row_square in ['ABC', 'DEF', 'GHI'] for col_square in ['123', '456', '789']]

	all_units = row_units + col_units + box_units  # Add units together
	groups = {}

	groups['units'] = {pos: [unit for unit in all_units if pos in unit] for pos in coords}

	groups['peers'] = {pos: set(sum(groups['units'][pos], [])) - {pos} for pos in coords}

	return coords, groups, all_units


def parse_puzzle(puzzle):
	"""
	Parses a string describing a Sudoku puzzle board into a dictionary with each cell mapped to its relevant
	coordinate, i.e. A1, A2, A3...
	"""

	if len(puzzle) != 81:
		raise ValueError('Input puzzle has %s grid positions specified, must be 81. Specify a position using any '
						 'digit from 1-9 and 0 or . for empty positions.' % len(puzzle))

	coords, groups, all_units = sudoku_reference()

	# Turn the list into a dictionary using the coordinates as the keys
	parsed_puzzle =  dict(zip(coords, puzzle))

	
	return parsed_puzzle



def validate_sudoku(puzzle):
	"""Checks if a completed Sudoku puzzle has a valid solution."""
	if puzzle is None:
		return False

	coords, groups, all_units = sudoku_reference()
	full = [str(x) for x in range(1, 10)]

	return all([sorted([puzzle[cell] for cell in unit]) == full for unit in all_units])


def solve_puzzle(puzzle):
	"""Solves a Sudoku puzzle from a string input."""
	digits = '123456789'  

	coords, groups, all_units = sudoku_reference()
	input_grid = parse_puzzle(puzzle)
	# print(input_grid)
	input_grid = {k: v for k, v in input_grid.items() if v != '.'}  
	output_grid = {cell: digits for cell in coords} 

	def confirm_value(grid, pos, val):
		"""Confirms a value by eliminating all other remaining possibilities."""
		remaining_values = grid[pos].replace(val, '')  
		for val in remaining_values:
			grid = eliminate(grid, pos, val)
		return grid

	def eliminate(grid, pos, val):
		"""Eliminates `val` as a possibility from all peers of `pos`."""

		if grid is None:  # Exit if grid has already found a contradiction
			return None

		if val not in grid[pos]:  # If we have already eliminated this value we can exit
			return grid

		grid[pos] = grid[pos].replace(val, '')  # Remove the possibility from the given cell

		if len(grid[pos]) == 0:  # If there are no remaining possibilities, we have made the wrong decision
			return None
		elif len(grid[pos]) == 1:  # We have confirmed the digit and so can remove that value from all peers now
			for peer in groups['peers'][pos]:
				grid = eliminate(grid, peer, grid[pos])  # Recurses, propagating the constraint
				if grid is None:  # Exit if grid has already found a contradiction
					return None

		# Check for the number of remaining places the eliminated digit could possibly occupy
		for unit in groups['units'][pos]:
			possibilities = [p for p in unit if val in grid[p]]

			if len(possibilities) == 0:  # If there are no possible locations for the digit, we have made a mistake
				return None
			# If there is only one possible position and that still has multiple possibilities, confirm the digit
			elif len(possibilities) == 1 and len(grid[possibilities[0]]) > 1:
				if confirm_value(grid, possibilities[0], val) is None:
					return None

		return grid

	# First pass of constraint propagation
	for position, value in input_grid.items():  # For each value we're given, confirm the value
		output_grid = confirm_value(output_grid, position, value)

	if validate_sudoku(output_grid):  # If successful, we can finish here
		return output_grid

	def guess_digit(grid):
		"""Guesses a digit from the cell with the fewest unconfirmed possibilities and propagates the constraints."""

		if grid is None:  # Exit if grid already compromised
			return None

		# Reached a valid solution, can end
		if all([len(possibilities) == 1 for cell, possibilities in grid.items()]):
			return grid

		# Gets the coordinate and number of possibilities for the cell with the fewest remaining possibilities
		n, pos = min([(len(possibilities), cell) for cell, possibilities in grid.items() if len(possibilities) > 1])

		for val in grid[pos]:
			# Run the constraint propagation, but copy the grid as we will try many adn throw the bad ones away.
			# Recursively guess digits until its complete and there's a valid solution
			solution = guess_digit(confirm_value(grid.copy(), pos, val))
			if solution is not None:
				return solution

	output_grid = guess_digit(output_grid)
	return output_grid

