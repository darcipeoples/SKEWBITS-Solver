from copy import deepcopy
from enum import Enum
import json
import time
from typing import Any, Dict, List, Tuple
from PIL import Image, ImageDraw

class Direction(Enum):
  BR = (1,1)
  BL = (-1,1)
  TR = (1,-1)
  TL = (-1,-1)

class Bit(Enum):
  ONE = '1'
  TWO_A = '2a'
  TWO_B = '2b'

class Piece(Enum):
  RED = 'red'
  BLUE = 'blue'
  YELLOW = 'yellow'
  GREEN = 'green'
  EMPTY = 'empty'

Grid = List[List[str]]

Coord = Tuple[int, int]

DOTS_CLOCKWISE = [Direction.TL, Direction.TR, Direction.BR, Direction.BL]

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
    'bit_symbol': 'X',
    'dot_symbol': 'O',
    'ansi_color': 8,
    'image_color': (240, 240, 240),
  },
}

EMPTY_DOT = PIECE_INFO[Piece.EMPTY]['dot_symbol']
EMPTY_CELL = PIECE_INFO[Piece.EMPTY]['bit_symbol']

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
  dir_idx = DOTS_CLOCKWISE.index(direction)
  inverse_dir_idx = (dir_idx + (len(DOTS_CLOCKWISE) // 2)) % len(DOTS_CLOCKWISE)
  return DOTS_CLOCKWISE[inverse_dir_idx]

def get_direction_between(from_pos: Coord, to_pos: Coord) -> Direction:
  x1, y1 = from_pos
  x2, y2 = to_pos
  dx, dy = x2 - x1, y2 - y1
  return get_direction_from_offset((dx, dy))

def get_coord_in_direction(coord: Coord, direction: Direction) -> Coord:
  return (coord[0] + direction.value[0], coord[1] + direction.value[1])

def rotate_direction(direction: Direction, clockwise_rotations: int) -> Direction:
  target_dir_idx = (DOTS_CLOCKWISE.index(direction) + clockwise_rotations) % len(DOTS_CLOCKWISE)
  return DOTS_CLOCKWISE[target_dir_idx]


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


######## DISPLAY FUNCTIONS ########

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
  M, N = max([len(line) for line in grid]) // 2, len(grid) // 2

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

def get_solution_image(grid: Grid, solution, cd: int = 20):
  M, N = max([len(line) for line in grid]) // 2, len(grid) // 2

  bg_color = (255, 255, 255)
  blank_color = PIECE_INFO[Piece.EMPTY]['image_color']
  outline_color = "black"

  # Create a new image
  image = Image.new("RGB", (cd*(M+2), cd*(N+2)), bg_color)
  draw = ImageDraw.Draw(image)

  original_open_cells = get_open_cells(grid)
  original_open_dots = get_open_dots(grid)

  # Draw a grid of empty cells
  for i in range(M):
    for j in range(N):
      cx, cy = (i+1.5)*cd, (j+1.5)*cd
      color = blank_color
      coords = [(cx - cd*0.4, cy - cd*0.4), (cx + cd*0.4, cy + cd*0.4)]
      draw.rectangle(coords, outline=color, width=2, fill=color)

  # Fill original empty cells with outline color
  for (i, j) in original_open_cells:
    cx, cy = (i//2+1.5)*cd, (j//2+1.5)*cd
    color = outline_color
    coords = [(cx - cd*0.60, cy - cd*0.60), (cx + cd*0.60, cy + cd*0.60)]
    draw.rectangle(coords, outline=color, width=5, fill=color)
  
  # Fill original available dots with outline color
  for (i, j) in original_open_dots:
    cx, cy = (i//2+1)*cd, (j//2+1)*cd
    color = outline_color
    coords = [(cx - cd*.3, cy - cd*.3), (cx + cd*.3, cy + cd*.3)]
    draw.ellipse(coords, outline=color, width=5, fill=color)

  # White-out the outline cells
  for (i, j) in original_open_cells:
    cx, cy = (i//2+1.5)*cd, (j//2+1.5)*cd
    color = bg_color
    coords = [(cx - cd*0.55, cy - cd*0.55), (cx + cd*0.55, cy + cd*0.55)]
    draw.rectangle(coords, outline=color, width=5, fill=color)

  # White-out the outline circles
  for (i, j) in original_open_dots:
    cx, cy = (i//2+1)*cd, (j//2+1)*cd
    color = bg_color
    coords = [(cx - cd*0.25, cy - cd*0.25), (cx + cd*0.25, cy + cd*0.25)]
    draw.ellipse(coords, outline=color, width=2, fill=color)

  # Draw rectangles for each bit
  for bit in solution:
    bit_x, bit_y = bit['cell_pos']
    piece_type = bit['piece_type']

    cx, cy = (bit_x//2+1.5)*cd, (bit_y//2+1.5)*cd
    color = PIECE_INFO[piece_type]['image_color']
    coords = [(cx - cd*0.45, cy - cd*0.45), (cx + cd*0.45, cy + cd*0.45)]
    draw.rectangle(coords, outline=color, width=5, fill=color)

  # White-out end dots
  for bit in solution:
    bit_x, bit_y = bit['cell_pos']
    end_dot_pos = bit['end_dot_pos']
    if end_dot_pos is not None:
      edx, edy = end_dot_pos
      cx, cy = (edx//2+1)*cd, (edy//2+1)*cd
      color = bg_color
      coords = [(cx - cd*0.25, cy - cd*0.25), (cx + cd*0.25, cy + cd*0.25)]
      draw.ellipse(coords, outline=color, width=2, fill=color)

  for bit in solution:
    bit_x, bit_y = bit['cell_pos']
    piece_type = bit['piece_type']

    end_dot_pos = bit['end_dot_pos']
    if end_dot_pos is not None:
      end_dot_x, end_dot_y = end_dot_pos

      # Draw hinge at center of dot
      edx, edy = (end_dot_x//2+1)*cd, (end_dot_y//2+1)*cd
      color = PIECE_INFO[piece_type]['image_color']
      coords = [(edx - cd*0.15, edy - cd*0.15), (edx + cd*0.15, edy + cd*0.15)]
      draw.ellipse(coords, outline=color, width=5, fill=color)

      # Connect hinge and cell with a linesquare
      bcx, bcy = (bit_x//2+1.5)*cd, (bit_y//2+1.5)*cd
      color = PIECE_INFO[piece_type]['image_color']
      coords = [(edx, edy), (bcx, bcy)]
      draw.line(coords, width=int(cd*.33), fill=color)

  return image
  

######## SOLVER FUNCTIONS #######
  
# set_piece_bits:         the bits we've set in place
# remaining_piece_bits:   the bits we haven't set yet
# open_cells:             cells that aren't set yet
# open_dots:              dots that aren't set yet
def solve(set_piece_bits, remaining_piece_bits, open_cells: List[Tuple[int, int]], open_dots: List[Tuple[int, int]]):
  print_partial_solution(set_piece_bits, remaining_piece_bits, open_cells, open_dots)

  # If no more remaining piece bits, we found the solution
  if len(remaining_piece_bits) == 0:
    return set_piece_bits
  
  # TODO: See if we can get rid of extra copying
  # Get the next bit to set & remove it from the remaining bits
  curr_bit = remaining_piece_bits[0]
  new_remaining_piece_bits = deepcopy(remaining_piece_bits[1:])

  starting_a_new_piece = False
  # If no bits set yet, we're starting a new piece
  if len(set_piece_bits) == 0:
    starting_a_new_piece = True
  # If current bit's piece is different from the previous, we're starting a new piece
  else:
    prev_bit = set_piece_bits[-1]
    if prev_bit['piece_idx'] != curr_bit['piece_idx']:
      starting_a_new_piece = True
  # If starting a new piece, we have to try all of the starting positions and orientations
  if starting_a_new_piece:
    # Put the first bit in any location
    for curr_bit_pos in open_cells:
      # Give the first bit any orientation (aka put its end dot anywhere)
      for end_dot_dir in DOTS_CLOCKWISE:
        end_dot_num = 0
        end_dot_pos = get_coord_in_direction(curr_bit_pos, end_dot_dir)

        start_dot_num, start_dot_pos, start_dot_dir = None, None, None

        # Set this bit and its end dot
        new_set_piece_bits = deepcopy(set_piece_bits)
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

        new_open_cells = deepcopy(open_cells)
        new_open_cells.remove(curr_bit_pos)
        new_open_dots = deepcopy(open_dots)
        if end_dot_pos is not None and end_dot_pos not in open_dots:
          continue
        new_open_dots.remove(end_dot_pos)

        solution = solve(
          new_set_piece_bits,
          new_remaining_piece_bits,
          new_open_cells,
          new_open_dots
        )

        if solution is None:
          continue
        return solution
    return None

  # If continuing a piece, try rotating all around the start dot
  for dir_from_start_dot in DOTS_CLOCKWISE:
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
    new_set_piece_bits = deepcopy(set_piece_bits)
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

    new_open_cells = deepcopy(open_cells)
    new_open_cells.remove(curr_bit_pos)
    new_open_dots = deepcopy(open_dots)
    # Skip setting if no end dot (e.g. type '1' & end of piece)
    if end_dot_pos is not None:
      new_open_dots.remove(end_dot_pos)

    solution = solve(
      new_set_piece_bits,
      new_remaining_piece_bits,
      new_open_cells,
      new_open_dots
    )

    if solution is None:
      continue
    return solution
  return None

######### MAIN FUNCTION ########

def main():
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

  # print(get_remaining_bits_summary(all_piece_bits))

  puzzle = '024.txt'
  puzzle_name = puzzle.split('.txt')[0]

  grid = [list(x.rstrip()) for x in open(f'puzzles/{puzzle}', 'r').readlines()]

  # print(get_grid_str(grid))

  all_open_cells = get_open_cells(grid)
  all_open_dots = get_open_dots(grid)

  start_time = time.time()
  solution = solve([], all_piece_bits, all_open_cells, all_open_dots)

  if solution is None:
    print('Did not find a solution. The puzzle is likely impossible')

  print(solution)
  
  image = get_solution_image(grid, solution)
  image.save(f"solutions/images/{puzzle_name}.png")

  with open(f"solutions/json/{puzzle_name}.json", 'w+') as f:
    result = {
      'solution': solution,
      'duration': time.time() - start_time,
    }
    json.dump(result, f, default=str)
  
if __name__ == '__main__':
  main()