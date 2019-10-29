import argparse
import logging

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger('sudoku-solver')


parser = argparse.ArgumentParser(description='Sudoku solver')
parser.add_argument('--p', type=str, required = True, help='path to image to evaluate')

image_path = parser.parse_args().p
logger.info(image_path)