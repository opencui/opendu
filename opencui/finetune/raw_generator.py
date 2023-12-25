import sys

from opencui.inference.converter import Generator

if __name__ == '__main__':

    generator = Generator.build()

    print("input:")
    for line in sys.stdin:
        if 'q' == line.rstrip():
            break
        print(generator.generate(line, None))
        print("input:")