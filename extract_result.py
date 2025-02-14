import os
import argparse
import json
from shutil import copyfile, copytree
from distutils.util import strtobool

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, required=True)
    parser.add_argument('--dest', type=str, required=True)
    args = parser.parse_args()

    failed_cases = []
    for item in os.walk(args.src):
        # item[0]: root, item[1]: dirs, item[2]: files
        if any(file_name == 'params.log' for file_name in item[2]):
            rounds, metric, do_ft = None, None, False
            with open(os.path.join(item[0], 'params.log'), 'r') as f:
                for line in f.readlines():
                    if line.startswith('rounds'):
                        rounds = int(line.split(' : ')[1])
                    if line.startswith('metric'):
                        metric = line.split(' : ')[1].strip()
                    if line.startswith('do_ft'):
                        do_ft = strtobool(line.split(' : ')[1].strip())

            if not os.path.exists(os.path.join(item[0], 'summary.json')):
                failed_cases.append(item[0])
                continue
            with open(os.path.join(item[0], 'summary.json'), 'r') as f:
                summary = json.load(f)
                if not do_ft and metric == 'bmta' and len(summary['test_loss'][0]) != rounds:
                    failed_cases.append(item[0])
                    continue
                if not do_ft and metric == 'last' and len(summary['test_loss'][0]) != 1:
                    failed_cases.append(item[0])
                    continue
                if do_ft and metric == 'bmta' and len(summary['test_loss'][0]) != rounds * 2:
                    failed_cases.append(item[0])
                    continue
                if do_ft and metric == 'last' and len(summary['test_loss'][0]) != 2:
                    failed_cases.append(item[0])
                    continue

            dest_root = os.path.join(args.dest, item[0].split(args.src)[1].lstrip(os.sep))
            if not os.path.exists(dest_root):
                os.makedirs(dest_root)
                for file_name in item[2]:
                    if file_name != 'train.log':
                        copyfile(os.path.join(item[0], file_name), os.path.join(dest_root, file_name))
        elif any(file_name == 'params.json' for file_name in item[2]):
            with open(os.path.join(item[0], 'params.json'), 'r') as f:
                params = json.load(f)
                rounds = params['rounds']
                metric = params['metric']
                do_ft = params.get('do_ft', False)

            summary_dir = os.path.join(item[0], 'summary')
            if not any(file_name.startswith('events.out.tfevents') for file_name in os.listdir(summary_dir)):
                failed_cases.append(item[0])
                continue

            dest_root = os.path.join(args.dest, item[0].split(args.src)[1].lstrip(os.sep))
            if not os.path.exists(dest_root):
                os.makedirs(dest_root)
                for file_name in item[2]:
                    if file_name != 'train.log':
                        copyfile(os.path.join(item[0], file_name), os.path.join(dest_root, file_name))
                copytree(summary_dir, os.path.join(dest_root, 'summary'))

    for failed_case in failed_cases:
        params = failed_case.split(args.src)[1].lstrip(os.sep).split(os.sep)[:-1]
        print(f"{' '.join(params)} failed!")
