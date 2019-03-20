import sys
import os

sys.path.append(os.path.abspath('../'))
import libs.io.reader as reader

if __name__ == '__main__':

    num_files = len(reader.BLUEPRINT_INPUT_FILES)
    for index_plan in range(num_files):
        print("index_plan", type(index_plan))
        command_lauch_grid = "python ../libs/modelers/grid.py -p %i" % (index_plan)
        # command_lauch_grid = "python ../libs/modelers/grid.py -p {args.plan_index}".format(args=args)
        # os.system(command_lauch_grid)

        os.system("python ../libs/modelers/grid.py -p %i" % (index_plan))
        # os.system("python ../libs/modelers/grid.py -p" +str(index_plan))

# os.system("rtl2gds -rtl={args.fileread} -rtl_top={args.module_name} -syn".format(args=args)
