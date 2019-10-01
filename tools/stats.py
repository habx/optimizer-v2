"""
Publication stats
https://metabase.habx.fr/question/549?maxNbSolutions=1&minNbSolutions=1&status=ok
"""
import matplotlib.pyplot as plt
import json
import seaborn as sns
import pandas as pd

if __name__ == '__main__':

    input_file_path = "query_result_times.json"

    # lines
    with open(input_file_path) as f:
        data = json.load(f)
    time_cpu = []
    time_real = []
    grid_time = []
    seeder_time = []
    space_planner_time = []
    corridor_time = []
    refiner_time = []
    nbr_of_rooms = []
    area = []
    for line in data:
        time_cpu.append(line["time_cpu"])
        time_real.append(line["time_real"])
        times = json.loads(line["times"])
        grid_time.append(times["grid"])
        seeder_time.append(times["seeder"])
        space_planner_time.append(times["space planner"])
        corridor_time.append(times["corridor"])
        refiner_time.append(times["refiner"])
        setup = json.loads(line["setup"])
        nbr_of_rooms.append(len(setup["rooms"]))
        blueprint = json.loads(line["blueprint"])
        blueprint_area = 0
        for space in blueprint["v2"]["spaces"]:
            if space["category"] == "empty":
                blueprint_area += space["area"]
        blueprint_area = blueprint_area/10000 # cm2 --> m2
        area.append(blueprint_area)

    print("time_cpu", time_cpu)
    print("time_real", time_real)
    print("nbr_of_rooms", nbr_of_rooms)
    print("area", area)

    df_time = pd.DataFrame(time_cpu, columns=["time_cpu"])
    area_round =[]
    for a in area:
        area_round.append(round(a))
    df_time["area"] = area
    df_time["area_round"] = area_round
    df_time["nbr_of_rooms"] = nbr_of_rooms
    df_time["time_real"] = time_real

    # plt.scatter(area, time_real)
    # plt.gca().set(xlabel='Area', ylabel='time_real')
    # plt.show()

    # plt.scatter(area, time_cpu, c=nbr_of_rooms, cmap="hsv")
    # plt.legend()
    # plt.gca().set(xlabel='Area', ylabel='time_cpu')
    # plt.show()

    df_boxplot = df_time[(df_time['area'] <= 100)&(df_time['nbr_of_rooms'] <= 10)]
    sns.boxplot("nbr_of_rooms", "time_cpu", data=df_boxplot, color = "w")
    #plt.show()
    plt.savefig("time_boxplot")
    plt.close()

    df_scatter = df_time[(df_time['area'] <= 120)&(df_time['nbr_of_rooms'] <= 10)]
    sns.scatterplot(x="area", y="time_cpu", hue="nbr_of_rooms", data=df_scatter)
    #plt.show()
    plt.savefig("time_scatter")
    plt.close()

    sns.countplot(df_time['nbr_of_rooms'])
    plt.show()
    plt.close()

    ax = sns.countplot(df_time['area_round'])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
    plt.tight_layout()
    plt.show()
    plt.close()


    # plt.scatter(area, grid_time)
    # plt.gca().set(xlabel='Area', ylabel='grid_time')
    # plt.show()
    #
    # plt.scatter(area, seeder_time)
    # plt.gca().set(xlabel='Area', ylabel='seeder_time')
    # plt.show()
    #
    # plt.scatter(area, space_planner_time)
    # plt.gca().set(xlabel='Area', ylabel='space_planner_time')
    # plt.show()
    #
    # plt.scatter(area, corridor_time)
    # plt.gca().set(xlabel='Area', ylabel='corridor_time')
    # plt.show()
    #
    # plt.scatter(area, refiner_time)
    # plt.gca().set(xlabel='Area', ylabel='refiner_time')
    # plt.show()
    #
    #
    #
    # plt.scatter(nbr_of_rooms, grid_time)
    # plt.gca().set(xlabel='nbr_of_rooms', ylabel='grid_time')
    # plt.show()
    #
    # plt.scatter(nbr_of_rooms, seeder_time)
    # plt.gca().set(xlabel='nbr_of_rooms', ylabel='seeder_time')
    # plt.show()
    #
    # plt.scatter(nbr_of_rooms, space_planner_time)
    # plt.gca().set(xlabel='nbr_of_rooms', ylabel='space_planner_time')
    # plt.show()
    #
    # plt.scatter(nbr_of_rooms, corridor_time)
    # plt.gca().set(xlabel='nbr_of_rooms', ylabel='corridor_time')
    # plt.show()
    #
    # plt.scatter(nbr_of_rooms, refiner_time)
    # plt.gca().set(xlabel='nbr_of_rooms', ylabel='refiner_time')
    # plt.show()
    #
    # plt.scatter(nbr_of_rooms, time_real)
    # plt.gca().set(xlabel='nbr_of_rooms', ylabel='time_real')
    # plt.show()
    #
    # plt.scatter(nbr_of_rooms, time_cpu)
    # plt.gca().set(xlabel='nbr_of_rooms', ylabel='time_cpu')
    # plt.show()
