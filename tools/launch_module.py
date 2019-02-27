import os
import argparse
import libs.reader as reader

# launch specified module no all plan in blueprint

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--module", help="choose launched module",
                        default="space_planner")
    args = parser.parse_args()
    module = args.module
    files = reader.get_list_from_folder()
    files = [x for x in files if x.endswith('.json')]
    num_files = len(files)
    for index_plan in range(num_files):
        print("current plan ", files[index_plan])
        command_lauch_grid = "python3 ../libs/" + module + ".py -p " + str(index_plan)
        os.system(command_lauch_grid)
