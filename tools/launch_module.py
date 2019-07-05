import os
import argparse
import libs.io.reader as reader

# launch specified module no all plan in blueprint

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--module", help="choose launched module",
                        default="equipments/doors")
    args = parser.parse_args()
    module = args.module
    files = reader.get_list_from_folder("../resources/blueprints")
    files = [x for x in files if x.endswith('.json')]
    num_files = len(files)
    for index_plan in range(num_files):
        if not index_plan in [1, 2, 3, 4, 5, 44, 16, 48, 20, 43, 15, 13, 18, 39, 9, 36, 11, 55, 19,
                              47, 53, 62, 58, 60, 56, 12]:
            continue
        command_lauch_grid = "python3 ../libs/" + module + ".py -p " + str(index_plan)
        os.system(command_lauch_grid)
