import argparse

# python2.7 arpgparse.py --integers 1 --strings s

parser = argparse.ArgumentParser(description='Name.')
parser.add_argument('--integers', default='2', metavar='N', type=int, nargs='*',
                    help='int key')
parser.add_argument('--strings', default='d', metavar='K', type=str, nargs='?',
                    help='str key')
args = parser.parse_args()
print(args.integers)
print(args.strings[0])
