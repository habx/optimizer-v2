import cProfile
import os
import pstats
#import tracemalloc

import pprofile
#import pyinstrument

from libs.executor.defs import ExecWrapper, TaskDefinition
import libs.optimizer as opt


class PProfile(ExecWrapper):
    def __init__(self, output_dir):
        super().__init__()
        self.output_dir = output_dir

    def _exec(self, td: TaskDefinition):
        prof = pprofile.Profile()
        with prof:
            res = super()._exec(td)
        prof.dump_stats(os.path.join(self.output_dir, 'pprofile_stats.out'))
        return res

    @staticmethod
    def instantiate(td: TaskDefinition):
        if td.params.get('pprofile', False):
            return __class__(td.local_context.output_dir)
        return None


# class PyInstrument(ExecWrapper):
#     FILENAME_HTML = 'pyinstrument.html'
#     FILENAME_TXT = 'pyinstrument.txt'
#
#     def __init__(self, output_dir: str):
#         super().__init__()
#         self.output_dir = output_dir
#         self.profiler = pyinstrument.Profiler()
#
#     def _before(self, td: TaskDefinition):
#         self.profiler.start()
#
#     def _after(self, td: TaskDefinition, resp: opt.Response):
#         self.profiler.stop()
#         with open(os.path.join(self.output_dir, self.FILENAME_HTML), 'w') as fp:
#             fp.write(self.profiler.output_html())
#         with open(os.path.join(self.output_dir, self.FILENAME_TXT), 'w') as fp:
#             fp.write(self.profiler.output_text())
#
#         td.local_context.add_file(
#             self.FILENAME_HTML,
#             ftype='profiling_html',
#             title='PyInstrument profiling (HTML)',
#             mime='text/html'
#         )
#
#         td.local_context.add_file(
#             self.FILENAME_TXT,
#             ftype='profiling_text',
#             title='PyInstrument profiling (text)',
#             mime='text/plain'
#         )
#
#     @staticmethod
#     def instantiate(td: TaskDefinition):
#         if td.params.get('pyinstrument', False):
#             return __class__(td.local_context.output_dir)
#         return None


class CProfile(ExecWrapper):
    """
    Enabling CPU profiling if the "c_profile" parameter is specified
    """

    def __init__(self, output_dir):
        super().__init__()
        self.output_dir = output_dir

    def _before(self, td: TaskDefinition):
        self.cpu_prof = cProfile.Profile()
        self.cpu_prof.enable()

    def _after(self, td: TaskDefinition, resp: opt.Response):
        self.cpu_prof.disable()
        self.cpu_prof.dump_stats(os.path.join(self.output_dir, "cProfile.prof"))
        with open(os.path.join(self.output_dir, 'cProfile.txt'), 'w') as fp:
            stats = pstats.Stats(self.cpu_prof, stream=fp)
            stats.strip_dirs()
            stats.sort_stats('cumulative')
            stats.print_stats()
        self.cpu_prof = None

    @staticmethod
    def instantiate(td: TaskDefinition):
        if td.params.get('c_profile', False):
            return __class__(td.local_context.output_dir)
        return None
