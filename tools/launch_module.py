import sys
import os
import argparse

sys.path.append(os.path.abspath('../'))
import libs.reader as reader

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--module", help="choose launched module",
                        default="grid")
    args = parser.parse_args()
    module = args.module

    num_files = len(reader.BLUEPRINT_INPUT_FILES)
    for index_plan in range(num_files):
        print("index_plan", type(index_plan))
        command_lauch_grid = "python ../libs/" + module + ".py -p " + str(index_plan)
        # command_lauch_grid = "python ../libs/grid.py -p {args.plan_index}".format(args=args)
        # os.system(command_lauch_grid)
        os.system(command_lauch_grid)
        # os.system("python ../libs/grid.py -p %i" % (index_plan))
        # os.system("python ../libs/grid.py -p" +str(index_plan))

# os.system("rtl2gds -rtl={args.fileread} -rtl_top={args.module_name} -syn".format(args=args)
