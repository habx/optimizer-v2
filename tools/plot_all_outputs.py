# -*- coding: utf-8 -*-

import os
import argparse


def StartHTML(repo, module):
    output_filename = repo + 'summary_' + module + '.html'
    Html_file = open(output_filename, "w")

    html_str = """
    <html>
        <head>
            <title> Solutions   </title>
            <style>img {margin-left:20px;}</style>
        </head>
        <body>
    """

    Html_file.write(html_str)
    Html_file.close()


# Integration of .svg in the HTML page
def PutSvgIntoHTML(image, repo, module):
    output_filename = repo + 'summary_' + module + '.html'
    Html_file = open(output_filename, "a")
    html_str = "<img src=" + image + " alt=/>"
    Html_file.write(html_str)
    Html_file.close()


# End of the HTML page

def EndHTML(repo, module):
    output_filename = repo + 'summary_' + module + '.html'
    Html_file = open(output_filename, "a")

    html_str = """
        </body>
    </html>
     """

    Html_file.write(html_str)
    Html_file.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--module", help="choose launched module",
                        default="grid")
    args = parser.parse_args()
    module = args.module

    repo = "../output/plots/" + module + "/"
    print("REPO IS", repo)
    StartHTML(repo, module)

    x = next(os.walk(repo))[2]
    for current_x in x:
        print("current_x", current_x)
        if (current_x.endswith(".svg")):
            PutSvgIntoHTML(current_x, repo, module)
    EndHTML(repo, module)
