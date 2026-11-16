"""One entry point for the utility workbench."""
import argparse
import importlib
import sys
import os

COMMANDS = dict(setup='setup_project',strategies='strategies',strategy='strategies',quickstart='quickstart',doctor='doctor',recommend='recommend',
                run='run_project',evaluate='evaluate_project',audit='audit_project',
                predict='predict_project',surrogate_demo='surrogate_demo')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,usage='%(prog)s COMMAND [options]')
    parser.add_argument('command',choices=COMMANDS)
    values = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(values[:1])
    remaining = values[1:]
    # A local CLI default, respecting an explicit user choice. Avoid excessive
    # OpenMP parallelism on small references and fragile Windows core discovery.
    os.environ.setdefault('OMP_NUM_THREADS','1')
    module = importlib.import_module('QML_utility.'+COMMANDS[args.command])
    return (module.run_main if args.command=='strategy' else module.main)(remaining)


if __name__ == '__main__':
    main()
