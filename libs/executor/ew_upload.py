import glob
import logging
import os
import json
from typing import List

import boto3

from libs.executor.defs import ExecWrapper, TaskDefinition


class S3Upload(ExecWrapper):
    """
    Upload of the results and generation of a manifest file
    """
    def __init__(self, output_dir: str, task_id: str):
        super().__init__()
        self.s3_client = boto3.client('s3')
        self.output_dir = output_dir
        # Note: This part is clunky, this should be fetched from the Config class but the overall
        #       architecture makes it quite hard to do it that way.
        self.s3_repository = 'habx-{env}-optimizer-v2'.format(env=os.getenv('HABX_ENV', 'dev'))
        self.task_id = task_id

    def _output_files(self) -> List[str]:
        return glob.glob(os.path.join(self.output_dir, '*'))

    def _exec(self, td: TaskDefinition):
        try:
            return self.next.run(td)
        except:
            raise
        finally:
            self._upload_files(td)

    def _upload_files(self, td: TaskDefinition):
        files_found = self._output_files()

        files = td.local_context.files

        # We check that there's an entry for each of the files we found
        for f in files_found:
            file_name = f[len(self.output_dir) + 1:]
            if file_name not in files:
                # Creating an empty object is good enough for now
                td.local_context.add_file(name=file_name)

        # OPT-114: We save the files as one of the manifest
        with open(os.path.join(self.output_dir, 'files.json'), 'w') as fp:
            json.dump(files, fp)

        files['files.json'] = {
            'mime': 'application/json',
            'title': 'Files listing'
        }

        # Updating the found files with the new 'files.json' file
        files_found = self._output_files()

        if not files_found:
            return

        logging.info("Uploading some files on S3...")

        for src_file in files_found:
            file_name = src_file[len(self.output_dir) + 1:]
            # OPT-89: Storing files in a "tasks" directory
            dst_file = "tasks/{task_id}/{file}".format(
                task_id=self.task_id,
                file=file_name
            )
            file = files.get(file_name, {})

            logging.info(
                "Uploading \"%s\" to s3://%s/%s",
                src_file,
                self.s3_repository,
                dst_file
            )

            extra_args = {
                'ACL': 'public-read',
                'ContentDisposition': 'inline',
            }

            if file.get('mime'):
                extra_args['ContentType'] = file.get('mime')

            self.s3_client.upload_file(
                src_file,
                self.s3_repository,
                dst_file,
                ExtraArgs=extra_args,
            )

        logging.info("Upload done...")

    @staticmethod
    def instantiate(td: TaskDefinition):
        if td.params.get('s3_upload', True) and td.task_id:
            return __class__(
                td.local_context.output_dir,
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
            return self.next.run(td)
        except:
            self._save_input(td)
            raise

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
            return __class__(td.local_context.output_dir)
        return None
