#!/usr/bin/env python3
import argparse
import json
import logging
import os

from libs.executor.defs import TaskDefinition
from libs.executor.executor import Executor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)-15s | %(lineno)-5d | %(levelname).4s | %(message)s",
)

parser = argparse.ArgumentParser(description='Fetch task and run it locally')

parser.add_argument('task_id',
                    help='Task identifier, mandatory.')
parser.add_argument('-e',
                    '--env',
                    dest='env',
                    help='Target environment.',
                    choices=['dev', 'staging', 'prod'],
                    default='dev')
parser.add_argument('-t',
                    '--token',
                    dest='habx_token',
                    help='habx token to use',
                    default=os.environ.get('HABX_TOKEN', 'ZBnV3nXJhdqehDmtWT'),
                    required=False)

args = parser.parse_args()

task_id: str = args.task_id
env: str = args.env
habx_token: str = args.habx_token

task_dir = 'tasks/%s' % task_id

completion_file = '%s/.done' % task_dir

if not os.path.exists(completion_file):
    os.makedirs(task_dir, exist_ok=True)
    cmd = "opt-replay {task_id} --env {env} --token {token} -o {task_dir}".format(
        task_id=task_id,
        env=env,
        token=habx_token,
        task_dir=task_dir,
    )
    logging.info("Executing %s", cmd)
    assert os.system(cmd) == 0
    with open(completion_file, 'w') as fp:
        pass

executor = Executor()
td = TaskDefinition()
td.local_context.output_dir = task_dir

with open('%s/blueprint.json' % task_dir, 'r') as blueprint_fp:
    td.blueprint = json.load(blueprint_fp)

with open('%s/setup.json' % task_dir, 'r') as setup_fp:
    td.setup = json.load(setup_fp)

with open('%s/params.json' % task_dir, 'r') as params_fp:
    td.params = json.load(params_fp)

response = executor.run(td)

meta = {
    'elapsed_times': response.elapsed_times,
}

with open(os.path.join(task_dir, "meta.json"), 'w') as response_fp:
    json.dump(meta, response_fp, indent=2, sort_keys=True)

for i, solution in enumerate(response.solutions):
    solution_path = os.path.join(task_dir, "solution_%d.json" % i)
    with open(solution_path, 'w') as solution_fp:
        json.dump(solution, solution_fp, indent=2, sort_keys=True)
