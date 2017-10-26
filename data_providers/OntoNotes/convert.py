import os
import subprocess
import sys
from collections import namedtuple
import shutil

sys.path.append(os.path.expanduser('~/reps/py-clearnlp-converter/'))
from clearnlp.converter import SubprocessConverter

sys.path.append(os.path.expanduser('~/reps/py-clearnlp-converter/'))


ONTONOTES_FOLDER = os.path.expanduser('~/data/OntoNotes')
CONVERTED_FOLDER = os.path.expanduser('~/data/OntoNotesConll')

print('Using folder "{}"'.format(ONTONOTES_FOLDER), file=sys.stderr)

shutil.copytree(ONTONOTES_FOLDER, CONVERTED_FOLDER)
for root, dirs, files in os.walk(CONVERTED_FOLDER):
    for fn in files:
        if not fn.endswith('.parse'):
            os.remove(os.path.join(root, fn))

print('A new folder "{}" has just created. Converting...'.format(CONVERTED_FOLDER), file=sys.stderr)

c = SubprocessConverter()

command = [c.java_command,
           '-ea',
           '-cp', ':'.join(c.classpath),
           c.class_name,
           '-h', c.head_rule_path,
           '-r',
           '-i', CONVERTED_FOLDER]
proc = subprocess.run(command, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    universal_newlines=True)


print('Done!', file=sys.stderr)
print('Return Code: {}'.format(proc.returncode), file=sys.stderr)
