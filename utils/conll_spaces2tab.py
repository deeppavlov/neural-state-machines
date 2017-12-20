import sys
import re

import os

if __name__ == '__main__':
    assert len(sys.argv) == 3

    infilename = sys.argv[1]
    assert os.path.isfile(infilename)
    outfilename = sys.argv[2]
    assert os.path.isfile(outfilename)

    errors_count = 0

    with open(infilename) as fin, open(outfilename, 'w') as fout:
        sentence = []
        for line in fin:
            line = line.strip()
            if not line or line.startswith('#'):
                print(line, file=fout)

            try:
                groups = re.match(r'(\d+.?\d*)' + r'\s+([^\s]+)' * 9, line).groups()
                print(*groups, file=fout, sep='\t')
            except Exception as e:
                print("'{}' happened for '{}'".format(e, line), file=sys.stderr)
                errors_count += 1
                if errors_count > 10:
                    raise
