import glob
import logging
import os
import json
from typing import List

import boto3

from libs.executor.defs import ExecWrapper, TaskDefinition
import libs.optimizer as opt


class S3Upload(ExecWrapper):
    """
    Upload of the results and generation of a manifest file
    """
    def __init__(self, output_dir: str, s3_repository: str, task_id: str):
        super().__init__()
        self.s3_client = boto3.client('s3')
        self.output_dir = output_dir
        self.s3_repository = s3_repository
        self.task_id = task_id

    def _output_files(self) -> List[str]:
        return glob.glob(os.path.join(self.output_dir, '*'))

    def _exec(self, td: TaskDefinition):
        response = self.next.run(td)

        self._save_output_files(response)

        return response

    def _save_output_files(self, response: opt.Response):
        files_found = self._output_files()

        files = response.generated_files

        # We check that there's an entry for each of the files we found
        for f in files_found:
            file_name = f[len(self.output_dir) + 1:]
            if file_name not in files:
                # Creating an empty object is good enough for now
                files[file_name] = {}

        # OPT-114: We save the files as one of the manifest
        with open(os.path.join(self.output_dir, 'files.json'), 'w') as fp:
            json.dump(files, fp)

        # Updating the found files with the new 'files.json' file
        files_found = self._output_files()

        if not files_found:
            return

        logging.info("Uploading some files on S3...")

        for src_file in files_found:
            # OPT-89: Storing files in a "tasks" directory
            dst_file = "tasks/{task_id}/{file}".format(
                task_id=self.task_id,
                file=src_file[len(self.output_dir) + 1:]
            )
            logging.info(
                "Uploading \"%s\" to s3://%s/%s",
                src_file,
                self.s3_repository,
                dst_file
            )
            self.s3_client.upload_file(
                src_file,
                self.s3_repository,
                dst_file,
                ExtraArgs={'ACL': 'public-read'}
            )

        logging.info("Upload done...")

    @staticmethod
    def instantiate(td: TaskDefinition):
        if td.params.get('s3_upload', True) and td.task_id:
            return __class__(
                td.local_params['output_dir'],
                td.local_params.get('s3_repository', ),
                td.task_id
            )
        return None


class SaveFilesOnError(ExecWrapper):
    """
    Save some files around the processing when an error occurs.
    """
    def __init__(self, output_dir: str):
        super().__init__()
        self.output_dir = output_dir
        pass

    def _exec(self, td: TaskDefinition):
        try:
            response = self.next.run(td)
        except:
            self._save_input(td)
            raise

        return response

    def _save_input(self, td: TaskDefinition):
        # If we had an issue, we save the output
        attrs = dir(td)
        for k in ['blueprint', 'setup', 'params', 'local_params', 'context']:
            if k in attrs:
                with open(os.path.join(self.output_dir, '%s.json' % k), 'w') as f:
                    json.dump(getattr(td, k), f)
        pass

    @staticmethod
    def instantiate(td: TaskDefinition):
        if td.params.get('save_files_on_error', True):
            return __class__(td.local_params['output_dir'])
        return None
