# -*- coding: utf-8 -*-

import os
import argparse
import shutil


# plots all files in specified folder on same html page

def startHTML(repo, module):
    output_filename = os.path.join(repo, 'summary_' + module + '.html')
    html_file = open(output_filename, "w")

    html_str = """
    <html>
        <head>
            <title> Solutions   </title>
            <style>img {margin-left:20px;}</style>
        </head>
        <body>
    """

    html_file.write(html_str)
    html_file.close()


# Integration of .svg in the HTML page
def putSvgIntoHTML(image, repo, module):
    output_filename = os.path.join(repo, 'summary_' + module + '.html')
    html_file = open(output_filename, "a")
    html_str = "<img src=" + image + " alt=/>"
    html_file.write(html_str)
    html_file.close()


# End of the HTML page

def endHTML(repo, module):
    output_filename = os.path.join(repo, 'summary_' + module + '.html')
    html_file = open(output_filename, "a")

    html_str = """
        </body>
    </html>
     """

    html_file.write(html_str)
    html_file.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--module", help="choose launched module",
                        default="grid")
    args = parser.parse_args()
    module = args.module

    repo = os.path.join("../output/plots/", module)
    print("REPO IS", repo)
    startHTML(repo, module)

    # file renaming
    x = next(os.walk(repo))[2]
    for current_x in x:
        init_name = repo + "/" + current_x
        fin_name = init_name.replace(" ", "")
        shutil.move(init_name, init_name.replace(" ", ""))

    x = next(os.walk(repo))[2]
    x.sort()
    for current_x in x:
        print("current_x", current_x)
        if current_x.endswith(".svg"):
            putSvgIntoHTML(current_x, repo, module)
    endHTML(repo, module)
