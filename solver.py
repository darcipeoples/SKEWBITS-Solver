import argparse
from copy import deepcopy
from enum import Enum
import json
import time
from typing import Any, Dict, List, Tuple
from PIL import Image, ImageDraw

EMPTY_DOT = 'O'
EMPTY_CELL = 'X'
HORIZ_DIV = '-'
VERT_DIV = '|'

OUTLINE_FILL = (0, 0, 0)
BACKGROUND_FILL = (255, 255, 255)
BLANK_CELL_FILL = (240,)*3

class Direction(Enum):
  UP = (0,-1)
  LEFT = (-1,0)
  DOWN = (0,1)
  RIGHT = (1,0)

  TOP_LEFT = (-1,-1)
  TOP_RIGHT = (1,-1)
  BOTTOM_RIGHT = (1,1)
  BOTTOM_LEFT = (-1,1)

DIAGONALS_CLOCKWISE = [Direction.TOP_LEFT, Direction.TOP_RIGHT, Direction.BOTTOM_RIGHT, Direction.BOTTOM_LEFT]
ADJACENT_CLOCKWISE = [Direction.UP, Direction.LEFT, Direction.DOWN, Direction.RIGHT]

Grid = List[List[str]]
Coord = Tuple[int, int]

class Piece(Enum):
  RED = 'red'
  BLUE = 'blue'
  YELLOW = 'yellow'
  GREEN = 'green'
  EMPTY = 'empty'

class Bit(Enum):
  ONE = '1'
  TWO_A = '2a'
  TWO_B = '2b'

PIECE_INFO = {
  Piece.RED: {
    'bit_symbol': 'R',
    'dot_symbol': 'r',
    'ansi_color': 160,
    'image_color': (250, 80, 70),
    'bits': [
      (Bit.ONE, (None,0)),
      (Bit.TWO_A, (0,1)),
      (Bit.TWO_A, (0,1)),
      (Bit.ONE, (0,None)),
    ],
  },
  Piece.BLUE: {
    'bit_symbol': 'B',
    'dot_symbol': 'b',
    'ansi_color': 21,
    'image_color': (20, 130, 200),
    'bits': [
      (Bit.ONE, (None, 0)),
      (Bit.TWO_B, (0,2)),
      (Bit.TWO_B, (0,2)),
      (Bit.TWO_B, (0,2)),
      (Bit.ONE, (0, None)),
    ],
  },
  Piece.YELLOW: {
    'bit_symbol': 'Y',
    'dot_symbol': 'y',
    'ansi_color': 220,
    'image_color': (250, 210, 80),
    'bits': [
      (Bit.ONE, (None, 0)),
      (Bit.TWO_B, (0,2)),
      (Bit.TWO_B, (0,2)),
      (Bit.ONE, (0, None)),
    ],
  },
  Piece.GREEN: {
    'bit_symbol': 'G',
    'dot_symbol': 'g',
    'ansi_color': 34,
    'image_color': (110, 220, 130),
    'bits': [
      (Bit.ONE, (None, 0)),
      (Bit.TWO_A, (1,0)),
      (Bit.TWO_B, (0,2)),
      (Bit.TWO_A, (0,1)),
      (Bit.ONE, (0,None)),
    ],
  },
  Piece.EMPTY: {
    'bit_symbol': EMPTY_CELL,
    'dot_symbol': EMPTY_DOT,
    'ansi_color': 8,
    'image_color': BLANK_CELL_FILL,
  },
}

######## GRID UTILS ########

def get_open_cells(grid: Grid) -> List[Coord]: 
  return [(x, y) for y, line in enumerate(grid) for x, cell in enumerate(line) if cell == EMPTY_CELL]

def get_open_dots(grid: Grid) -> List[Coord]: 
  return [(x, y) for y, line in enumerate(grid) for x, cell in enumerate(line) if cell == EMPTY_DOT]

def get_grid_str(grid: Grid) -> str: 
  return '\n'.join([' '.join(x) for x in grid])


######## DIRECTION UTILS ########

def get_direction_from_offset(offset: Coord) -> Direction:
  for member in Direction:
      if member.value == offset:
          return member
  raise Exception("Unable to cast offset to Direction enum")

def get_inverse_direction(direction: Direction) -> Direction: 
  dir_idx = DIAGONALS_CLOCKWISE.index(direction)
  inverse_dir_idx = (dir_idx + (len(DIAGONALS_CLOCKWISE) // 2)) % len(DIAGONALS_CLOCKWISE)
  return DIAGONALS_CLOCKWISE[inverse_dir_idx]

def get_direction_between(from_pos: Coord, to_pos: Coord) -> Direction:
  x1, y1 = from_pos
  x2, y2 = to_pos
  dx, dy = x2 - x1, y2 - y1
  return get_direction_from_offset((dx, dy))

def get_coord_in_direction(coord: Coord, direction: Direction) -> Coord:
  return (coord[0] + direction.value[0], coord[1] + direction.value[1])

def rotate_direction(direction: Direction, clockwise_rotations: int) -> Direction:
  target_dir_idx = (DIAGONALS_CLOCKWISE.index(direction) + clockwise_rotations) % len(DIAGONALS_CLOCKWISE)
  return DIAGONALS_CLOCKWISE[target_dir_idx]


######## BIT TYPE HELPERS ########

# Find out info about the end dot, based on the bit and start dot
def get_end_dot_info(curr_bit_pos: Coord, start_dot_num: int, start_dot_dir: Direction, end_dot_num: int) -> Tuple[int, Direction]:
  if start_dot_num is None:
    raise Exception("get_end_dot_info can't handle bits without a start dot")
  
  # No end dot (end of a piece)
  if end_dot_num is None:
    return None, None
  
  clockwise_rotations = end_dot_num - start_dot_num
  end_dot_dir = rotate_direction(start_dot_dir, clockwise_rotations)
  end_dot_pos = get_coord_in_direction(curr_bit_pos, end_dot_dir)

  return end_dot_pos, end_dot_dir


######## DEBUGGING FUNCTIONS #########

def get_set_bit_summary(sb: Dict[str, Any], n=20) -> str:
  return f"{sb['bit_idx']:<{n}}" \
    f"{sb['bit_type'].value:<{n}}" \
    f"{sb['cell_pos']:<{n}}" \
    f"{str(sb['start_dot_num']):<{n}}" \
    f"{str(sb['start_dot_pos']):<{n}}" \
    f"{str(sb['start_dot_dir']):<{n}}" \
    f"{str(sb['end_dot_num']):<{n}}" \
    f"{str(sb['end_dot_pos']):<{n}}" \
    f"{str(sb['end_dot_dir']):<{n}}" \
    f"{bit_str(sb['piece_type']):<{n}}"

def get_set_bits_summary(sbs: List[Dict[str, Any]], n=20) -> str:
  headers = ['bit_idx','bit_type','cell_pos','start_dot_num','start_dot_pos','start_dot_dir','end_dot_num','end_dot_pos','end_dot_dir','piece_type']
  header = ''.join([f"{header:<{n}}" for header in headers])
  bit_strs = []
  for sb in sbs:
    bit_strs.append(get_set_bit_summary(sb, n))
  return f"{header}\n" + '\n'.join(bit_strs)

def get_remaining_bit_summary(rb: Dict[str, Any], n=20) -> str:
  return f"{rb['bit_idx']:<{n}}" \
    f"{rb['bit_type'].value:<{n}}" \
    f"{str(rb['start_dot_num']):<{n}}" \
    f"{str(rb['end_dot_num']):<{n}}" \
    f"{bit_str(rb['piece_type']):<{n}}"

def get_remaining_bits_summary(rbs: List[Dict[str, Any]], n=20) -> str:
  headers = ['bit_idx','bit_type','start_dot_num','end_dot_num','piece_type']
  header = ''.join([f"{header:<{n}}" for header in headers])
  bit_strs = []
  for rb in rbs:
    bit_strs.append(get_remaining_bit_summary(rb, n))
  return f"{header}\n" + '\n'.join(bit_strs)

def print_partial_solution(set_piece_bits, remaining_piece_bits, open_cells: List[Coord], open_dots: List[Coord], verbose: bool = False):
  M = max(
    [max([x[0] for x in [bit['cell_pos'], bit['start_dot_pos'], bit['end_dot_pos']] if x is not None]) for bit in set_piece_bits] + 
    [cell[0] for cell in open_cells] + 
    [dot[0] for dot in open_dots]
  ) + 1
  N = max(
    [max([x[1] for x in [bit['cell_pos'], bit['start_dot_pos'], bit['end_dot_pos']] if x is not None]) for bit in set_piece_bits] + 
    [cell[1] for cell in open_cells] + 
    [dot[1] for dot in open_dots]
  ) + 1

  if verbose:
    print(M, N)
    print(f'\nSet bits ({len(set_piece_bits)}):')
    print(get_set_bits_summary(set_piece_bits))
    print(f'\nRemaining bits ({len(remaining_piece_bits)}):')
    print(get_remaining_bits_summary(remaining_piece_bits))
    print(f'\nOpen cells ({len(open_cells)}):\t', '\t'.join([str(x) for x in open_cells]))
    print(f'Open dots: ({len(open_dots)})\t', '\t'.join([str(x) for x in open_dots]))

  grid = [[' ' for x in range(M)] for y in range(N)]

  for bit in set_piece_bits:
    bit_x, bit_y = bit['cell_pos']
    piece_type = bit['piece_type']
    grid[bit_y][bit_x] = bit_str(piece_type)

    start_dot_pos = bit['start_dot_pos']
    if start_dot_pos is not None:
      start_dot_x, start_dot_y = start_dot_pos
      grid[start_dot_y][start_dot_x] = dot_str(piece_type)

    end_dot_pos = bit['end_dot_pos']
    if end_dot_pos is not None:
      end_dot_x, end_dot_y = end_dot_pos
      grid[end_dot_y][end_dot_x] = dot_str(piece_type)

  for cell_x, cell_y in open_cells:
    grid[cell_y][cell_x] = bit_str(Piece.EMPTY)

  for dot_x, dot_y in open_dots:
    grid[dot_y][dot_x] = dot_str(Piece.EMPTY)

  print(get_grid_str(grid))


######## TEXT DISPLAY FUNCTIONS ########

def bit_str(piece_type: Piece) -> str:
  piece_info = PIECE_INFO[piece_type]
  ansi_color = piece_info['ansi_color']
  bit_symbol = piece_info['bit_symbol']
  return u"\u001b[48;5;" + str(ansi_color) + "m" + bit_symbol + u"\u001b[0m"

def dot_str(piece_type: Piece) -> str:
  piece_info = PIECE_INFO[piece_type]
  ansi_color = piece_info['ansi_color']
  dot_symbol = piece_info['dot_symbol']
  return u"\u001b[38;5;" + str(ansi_color) + "m" + dot_symbol + u"\u001b[0m"

def get_soln_str(grid: Grid, solution) -> str:
  grid = deepcopy(grid)

  for bit in solution:
    bit_x, bit_y = bit['cell_pos']
    piece_type = bit['piece_type']
    grid[bit_y][bit_x] = bit_str(piece_type)

    start_dot_pos = bit['start_dot_pos']
    if start_dot_pos is not None:
      start_dot_x, start_dot_y = start_dot_pos
      grid[start_dot_y][start_dot_x] = dot_str(piece_type)

    end_dot_pos = bit['end_dot_pos']
    if end_dot_pos is not None:
      end_dot_x, end_dot_y = end_dot_pos
      grid[end_dot_y][end_dot_x] = dot_str(piece_type)
  
  return get_grid_str(grid)

######## IMAGE GENERATION UTILS ########

def get_cell_center(i, j, cd):
  return (i//2+1.5)*cd, (j//2+1.5)*cd

def get_dot_center(i, j, cd):
  return (i//2+1)*cd, (j//2+1)*cd

def get_scaled_coords(coord, cd, scale):
  cx, cy = coord
  return ((cx - cd*scale, cy - cd*scale), (cx + cd*scale, cy + cd*scale))

# Generate an image showing the puzzle outline
def get_outline_image(grid: Grid, cd: int = 20) -> Image:
  M, N = max([len(line) for line in grid]) // 2, len(grid) // 2

  image = Image.new("RGB", (cd*(M+2), cd*(N+2)), BACKGROUND_FILL)
  draw = ImageDraw.Draw(image)

  open_cells = get_open_cells(grid)
  open_dots = get_open_dots(grid)

  # Draw an MxN grid of blank cells
  for i in range(0, M*2, 2):
    for j in range(0, N*2, 2):
      color = BLANK_CELL_FILL
      center = get_cell_center(i, j, cd)
      coords = get_scaled_coords(center, cd, 0.4)
      draw.rectangle(coords, outline=color, fill=color)

  # Overfill all available cells with the outline color
  for (i, j) in open_cells:
    color = OUTLINE_FILL
    center = get_cell_center(i, j, cd)
    coords = get_scaled_coords(center, cd, 0.6)
    draw.rectangle(coords, outline=color, fill=color)
  
  # Overfill all available dots with outline color
  for (i, j) in open_dots:
    color = OUTLINE_FILL
    center = get_dot_center(i, j, cd)
    coords = get_scaled_coords(center, cd, 0.3)
    draw.ellipse(coords, outline=color, fill=color)

  # Fill the outline's cells with the background color
  for (i, j) in open_cells:
    color = BACKGROUND_FILL
    center = get_cell_center(i, j, cd)
    coords = get_scaled_coords(center, cd, 0.55)
    draw.rectangle(coords, outline=color, fill=color)

  # Fill the outline's dots with the background color
  for (i, j) in open_dots:
    color = BACKGROUND_FILL
    center = get_dot_center(i, j, cd)
    coords = get_scaled_coords(center, cd, 0.25)
    draw.ellipse(coords, outline=color, fill=color)

  # Underfill the outline's cells with the empty cell color
  for (i, j) in open_cells:
    color = BLANK_CELL_FILL
    center = get_cell_center(i, j, cd)
    coords = get_scaled_coords(center, cd, 0.4)
    draw.rectangle(coords, outline=color, fill=color)

  return image

# Generate an image of the solution on the grid
def get_solution_image(grid: Grid, solution, cd: int = 20) -> Image:
  image = get_outline_image(grid, cd)
  draw = ImageDraw.Draw(image)

  # Fill each solution cell with the bit's color
  for bit in solution:
    i, j = bit['cell_pos']
    piece_type = bit['piece_type']

    color = PIECE_INFO[piece_type]['image_color']
    center = get_cell_center(i, j, cd)
    coords = get_scaled_coords(center, cd, 0.45)
    draw.rectangle(coords, outline=color, fill=color)

  # Overfill all solution dots with the background color
  for bit in solution:
    bit_x, bit_y = bit['cell_pos']
    end_dot_pos = bit['end_dot_pos']
    if end_dot_pos is not None:
      i, j = end_dot_pos
      color = BACKGROUND_FILL
      center = get_dot_center(i, j, cd)
      coords = get_scaled_coords(center, cd, 0.25)
      draw.ellipse(coords, outline=color, fill=color)

  # Fill each solution dot with the bit's color
  # Connect each dot to the bit it came from
  for bit in solution:
    bit_pos = bit['cell_pos']
    piece_type = bit['piece_type']
    piece_color = PIECE_INFO[piece_type]['image_color']

    end_dot_pos = bit['end_dot_pos']
    end_dot_dir = bit['end_dot_dir']
    if end_dot_pos is None or end_dot_dir is None:
      continue

    i, j = end_dot_pos
    dot_center = get_dot_center(i, j, cd)
    i, j = bit_pos
    bit_center = get_cell_center(i, j, cd)

    # Connect the dot and bit with a line
    color = piece_color
    coords = [dot_center, bit_center]
    draw.line(coords, width=int(cd*.3), fill=color)

    # Fill the dot with the bit's color
    color = piece_color
    coords = get_scaled_coords(dot_center, cd, 0.15)
    draw.ellipse(coords, outline=color, fill=color)

  return image


######## SOLVER FUNCTIONS #######

def find_connected_components(coords):
    coords = coords.copy()
    
    def is_valid(x, y):
        return (x, y) in coords

    def dfs(x, y, component):
        if is_valid(x, y):
            component.append((x, y))
            coords.remove((x, y))  # Mark the cell as visited by removing it

            # Check all 8 neighbors
            for dx in [-2, 0, 2]:
                for dy in [-2, 0, 2]:
                    if dx == 0 and dy == 0:
                        continue
                    dfs(x + dx, y + dy, component)

    connected_components = []
    while coords:
        x, y = coords[0]  # Start with the first remaining coordinate
        component = []
        dfs(x, y, component)
        connected_components.append(component)

    return connected_components

def has_impossible_cell_chains(set_piece_bits, remaining_piece_bits, open_cells):
  # If more open cells than remaining bits, hard to say if impossible
  if len(remaining_piece_bits) < len(open_cells):
    return False

  # Find out how many bits left in the remaining pieces
  set_pieces = set([bit['piece_idx'] for bit in set_piece_bits])
  remaining_bit_pieces = [bit['piece_idx'] for bit in remaining_piece_bits]
  
  # If some pieces partially done, hard to say
  if set_pieces.intersection(remaining_bit_pieces):
    return False
  
  remaining_piece_lens = {}
  for piece in remaining_bit_pieces:
    if piece not in remaining_piece_lens:
      remaining_piece_lens[piece] = 0
    remaining_piece_lens[piece] += 1
  min_piece_length = min(remaining_piece_lens.values())
  
  open_cell_chains = find_connected_components(open_cells)

  # If more isolated components than pieces, will have empty spaces
  if len(open_cell_chains) > len(remaining_piece_lens):
    return True

  for open_cell_chain in open_cell_chains:
    if len(open_cell_chain) < min_piece_length:
      return True
    
  return False

# set_piece_bits:         the bits we've set in place
# remaining_piece_bits:   the bits we haven't set yet
# open_cells:             cells that aren't set yet
# open_dots:              dots that aren't set yet
def solve(set_piece_bits, remaining_piece_bits, open_cells: List[Tuple[int, int]], open_dots: List[Tuple[int, int]], solve_all):

  # If no more remaining piece bits, we found a solution
  if len(remaining_piece_bits) == 0:
    return [set_piece_bits]
  
  # Get the next bit to set & remove it from the remaining bits
  curr_bit = remaining_piece_bits[0]
  new_remaining_piece_bits = remaining_piece_bits[1:]

  starting_a_new_piece = False
  # If no bits set yet, we're starting a new piece
  if len(set_piece_bits) == 0:
    starting_a_new_piece = True
  # If current bit's piece is different from the previous, we're starting a new piece
  else:
    prev_bit = set_piece_bits[-1]
    if prev_bit['piece_idx'] != curr_bit['piece_idx']:
      starting_a_new_piece = True

  all_solutions = []
  
  # If starting a new piece, we have to try all of the starting positions and orientations
  if starting_a_new_piece:
    has_impossible_chains = has_impossible_cell_chains(set_piece_bits, remaining_piece_bits, open_cells)
    if has_impossible_chains:
      return []
  
    print_partial_solution(set_piece_bits, remaining_piece_bits, open_cells, open_dots)

    # Put the first bit in any location
    for curr_bit_pos in open_cells:
      # Give the first bit any orientation (aka put its end dot anywhere)
      for end_dot_dir in DIAGONALS_CLOCKWISE:
        end_dot_num = 0
        end_dot_pos = get_coord_in_direction(curr_bit_pos, end_dot_dir)

        start_dot_num, start_dot_pos, start_dot_dir = None, None, None

        # Set this bit and its end dot
        new_set_piece_bits = set_piece_bits.copy()
        new_set_piece_bits.append({
          'piece_idx': curr_bit['piece_idx'],
          'piece_type': curr_bit['piece_type'],
          
          'bit_idx': curr_bit['bit_idx'],
          'bit_type': curr_bit['bit_type'],

          'cell_pos': curr_bit_pos,

          'start_dot_num': start_dot_num,
          'start_dot_pos': start_dot_pos,
          'start_dot_dir': start_dot_dir,

          'end_dot_num': end_dot_num,
          'end_dot_pos': end_dot_pos,
          'end_dot_dir': end_dot_dir,
        })

        new_open_cells = open_cells.copy()
        new_open_cells.remove(curr_bit_pos)
        new_open_dots = open_dots.copy()
        if end_dot_pos is not None and end_dot_pos not in open_dots:
          continue
        new_open_dots.remove(end_dot_pos)

        child_solutions = solve(
          new_set_piece_bits,
          new_remaining_piece_bits,
          new_open_cells,
          new_open_dots,
          solve_all
        )

        if child_solutions == []:
          continue
        if not solve_all:
          return child_solutions
        all_solutions.extend(child_solutions)
    return all_solutions

  # If continuing a piece, try rotating all around the start dot
  for dir_from_start_dot in DIAGONALS_CLOCKWISE:
    # If continuing a piece, fetch where our last dot was
    prev_bit = set_piece_bits[-1]
    
    start_dot_pos = prev_bit['end_dot_pos']

    curr_bit_pos = get_coord_in_direction(start_dot_pos, dir_from_start_dot)
    start_dot_dir = get_inverse_direction(dir_from_start_dot)
    assert start_dot_dir == get_direction_between(curr_bit_pos, start_dot_pos)

    # Skip this possible coord if already occupied
    if curr_bit_pos not in open_cells:
      continue
    # Find out where the end dot has to go
    end_dot_pos, end_dot_dir = get_end_dot_info(
      curr_bit_pos, 
      curr_bit['start_dot_num'], 
      start_dot_dir,
      curr_bit['end_dot_num']
    )
    # Skip this possible coord if its end dot's pos is already taken
    if end_dot_pos is not None and end_dot_pos not in open_dots:
      continue

    # Set tbe bit & its end dot
    new_set_piece_bits = set_piece_bits.copy()
    new_set_piece_bits.append({
      'piece_idx': curr_bit['piece_idx'],
      'piece_type': curr_bit['piece_type'],
      
      'bit_idx': curr_bit['bit_idx'],
      'bit_type': curr_bit['bit_type'],

      'cell_pos': curr_bit_pos,

      'start_dot_num': curr_bit['start_dot_num'], 
      'start_dot_pos': start_dot_pos,
      'start_dot_dir': start_dot_dir,

      'end_dot_num': curr_bit['end_dot_num'],
      'end_dot_pos': end_dot_pos,
      'end_dot_dir': end_dot_dir,
    })

    new_open_cells = open_cells.copy()
    new_open_cells.remove(curr_bit_pos)
    new_open_dots = open_dots.copy()
    # Skip setting if no end dot (e.g. type '1' & end of piece)
    if end_dot_pos is not None:
      new_open_dots.remove(end_dot_pos)

    child_solutions = solve(
      new_set_piece_bits,
      new_remaining_piece_bits,
      new_open_cells,
      new_open_dots,
      solve_all
    )

    if child_solutions == []:
      continue
    if not solve_all:
      return child_solutions
    all_solutions.extend(child_solutions)
  return all_solutions


######## INPUT PARSING FUNCTIONS ########

def parse_solution_json_obj(solution_obj):
  solution_obj = deepcopy(solution_obj)
  for bit in solution_obj:
    bit['piece_type'] = Piece[bit['piece_type'].split('Piece.')[1]]
    bit['bit_type'] = Bit[bit['bit_type'].split('Bit.')[1]]
    if bit['start_dot_dir'] is not None:
      bit['start_dot_dir'] = Direction[bit['start_dot_dir'].split('Direction.')[1]]
    if bit['end_dot_dir'] is not None:
      bit['end_dot_dir'] = Direction[bit['end_dot_dir'].split('Direction.')[1]]
  return solution_obj
  
def load_solution_from_file(filename):
  with open(f'solutions/json/{filename}') as f:
    solution = json.load(f)['solution']
    solution = parse_solution_json_obj(solution)
  return solution

def load_solutions_from_file(filename):
  parsed_solutions = []
  with open(f'solutions/json/{filename}') as f:
    solutions = json.load(f)['solutions']
    for solution in solutions:
      parsed_solutions.append(parse_solution_json_obj(solution))
  return parsed_solutions

def load_grid_from_file(filename):
  grid = [list(x.rstrip().upper()) for x in open(f'puzzles/{filename}', 'r').readlines()]
  grid = get_grid_mark_empty_dots(grid)
  return grid

# Replace CELLS surrounded by dividers with EMPTY_CELL
def get_grid_mark_empty_cells(grid):
  grid = deepcopy(grid)
  for y, line in enumerate(grid):
    for x, cell in enumerate(line):
      if y % 2 != 1 or x % 2 != 1:
        continue
      neighbor_symbols = []
      for direction in ADJACENT_CLOCKWISE:
        nx, ny = get_coord_in_direction((x, y), direction)
        if ny >= len(grid) or ny < 0 or nx >= len(grid[ny]) or nx < 0:
          neighbor_symbols.append(None)
          continue
        neighbor_symbols.append(grid[ny][nx])
      if neighbor_symbols == [HORIZ_DIV, VERT_DIV]*2:
        grid[y][x] = EMPTY_CELL
  return grid

# Replace dots surrounded by EMPTY_CELL with EMPTY_DOT
def get_grid_mark_empty_dots(grid):
  grid = deepcopy(grid)
  for y, line in enumerate(grid):
    for x, cell in enumerate(line):
      if y % 2 != 0 or x % 2 != 0:
        continue
      neighbor_symbols = []
      for direction in DIAGONALS_CLOCKWISE:
        nx, ny = get_coord_in_direction((x, y), direction)
        if ny >= len(grid) or ny < 0 or nx >= len(grid[ny]) or nx < 0:
          neighbor_symbols.append(None)
          continue
        neighbor_symbols.append(grid[ny][nx])
      if set(neighbor_symbols) == {EMPTY_CELL}:
        grid[y][x] = EMPTY_DOT
  return grid

# Dedupe solutions based just on the color and location of the bits and dots (ignore hinge direction)
def dedupe_solutions(solutions):
  solution_map = {}
  for solution in solutions:
    dot_coords_to_piece_type = {}
    bit_coords_to_piece_type = {}
    for bit in solution:
      piece_type = bit['piece_type']
      cell_pos = bit['cell_pos']
      start_dot_pos = bit['start_dot_pos']
      end_dot_pos = bit['end_dot_pos']
      bit_coords_to_piece_type[tuple(cell_pos)] = piece_type
      if start_dot_pos is not None:
        dot_coords_to_piece_type[tuple(start_dot_pos)] = piece_type
      if end_dot_pos is not None:
        dot_coords_to_piece_type[tuple(end_dot_pos)] = piece_type
    key = str(sorted(bit_coords_to_piece_type.items())) + str(sorted(dot_coords_to_piece_type.items())) 
    print(key)
    solution_map[key] = solution
  return list(solution_map.values())


######### MAIN FUNCTION ########

def main():
  # Define and parse command line arguments
  parser = argparse.ArgumentParser(description="Solver for Make Anything's SKEWBITS puzzle.")
  parser.add_argument('filename', help='The puzzle filename to read from.')
  parser.add_argument('--solve-all', action='store_true', help='Whether to calculate all solutions or just the first found.')
  args = parser.parse_args()
  puzzle_filename = args.filename
  solve_all = args.solve_all

  puzzle_name = puzzle_filename.lower().split('.txt')[0]

  # Get a list of all of the piece bits which we'll need to place
  all_pieces = [Piece.RED, Piece.YELLOW, Piece.GREEN, Piece.BLUE]
  all_piece_bits = []
  for piece_idx, piece_type in enumerate(all_pieces):
      piece_info = PIECE_INFO[piece_type]
      for bit_idx, bit_item in enumerate(piece_info['bits']):
        bit_type, dot_nums = bit_item
        start_dot_num, end_dot_num = dot_nums
        all_piece_bits.append({
          'piece_idx': piece_idx,
          'piece_type': piece_type,

          'bit_idx': bit_idx,
          'bit_type': bit_type,

          'start_dot_num': start_dot_num,

          'end_dot_num': end_dot_num,
        })

  # Load the puzzle grid from file
  grid = load_grid_from_file(puzzle_filename)
  print(get_grid_str(grid))

  # Save an image of the puzzle outline
  image = get_outline_image(grid)
  image.save(f"outlines/{puzzle_name}.png")

  # Solve the puzzle
  all_open_cells = get_open_cells(grid)
  all_open_dots = get_open_dots(grid)

  start_time = time.time()
  solutions = solve([], all_piece_bits, all_open_cells, all_open_dots, solve_all)

  # Load the puzzle from file
  # solutions = load_solutions_from_file(f'{puzzle_name}-all.json')

  # Print info about the found soltuion(s)
  if solutions == []:
    print('Did not find any solutions. The puzzle is likely impossible')
    return
  
  total_solutions = len(solutions)
  solutions = dedupe_solutions(solutions)
  print(solutions)
  print(f"{len(solutions)} solutions (deduped from {total_solutions})")

  # Save the solution image(s)
  for i, solution in enumerate(solutions):
    print(get_soln_str(grid, solution))

    image = get_solution_image(grid, solution, 20)
    image.save(f"solutions/images/{puzzle_name}{f'-{i}' if solve_all else ''}.png")

  # Save the solution(s) json
  with open(f"solutions/json/{puzzle_name}{'-all' if solve_all else ''}.json", 'w+') as f:
    result = {
      'solutions': solutions,
      'duration': int(time.time() - start_time),
    }
    json.dump(result, f, default=str, indent=2)
  
if __name__ == '__main__':
  main()