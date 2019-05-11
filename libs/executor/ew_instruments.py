import cProfile
import os
import pstats
import tracemalloc

import pprofile
import pyinstrument

from libs.executor.defs import ExecWrapper, TaskDefinition


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


class PyInstrument(ExecWrapper):
    def __init__(self, output_dir: str):
        super().__init__()
        self.output_dir = output_dir
        self.profiler = pyinstrument.Profiler()

    def _before(self):
        self.profiler.start()

    def _after(self):
        self.profiler.stop()
        with open(os.path.join(self.output_dir, 'pyinstrument.html'), 'w') as fp:
            fp.write(self.profiler.output_html())
        with open(os.path.join(self.output_dir, 'pyinstrument.txt'), 'w') as fp:
            fp.write(self.profiler.output_text())

    @staticmethod
    def instantiate(td: TaskDefinition):
        if td.params.get('pyinstrument', False):
            return __class__(td.local_context.output_dir)
        return None


class CProfile(ExecWrapper):
    """
    Enabling CPU profiling if the "c_profile" parameter is specified
    """

    def __init__(self, output_dir):
        super().__init__()
        self.output_dir = output_dir

    def _before(self):
        self.cpu_prof = cProfile.Profile()
        self.cpu_prof.enable()

    def _after(self):
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


class TraceMalloc(ExecWrapper):
    """
    Enabling malloc monitoring if "traceMalloc" is specified
    """

    def __init__(self, output_dir):
        super().__init__()
        self.output_dir = output_dir

    def _before(self):
        tracemalloc.start()

    def _after(self):
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')

        with open(os.path.join(self.output_dir, 'mem_stats.txt'), 'w') as f:
            for stat in top_stats[:40]:
                f.write("%s\n" % stat)

    @staticmethod
    def instantiate(td: TaskDefinition):
        if td.params.get('tracemalloc', False):
            return __class__(td.local_context.output_dir)
        return None
