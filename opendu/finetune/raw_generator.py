# Copyright (c) 2025 BeThere AI
# All rights reserved.
#
# This source code is licensed under the BeThere AI license.
# See LICENSE file in the project root for full license information.
import sys

from opendu.inference.parser import Generator

if __name__ == '__main__':

    generator = Generator.build()

    print("input:")
    for line in sys.stdin:
        if 'q' == line.rstrip():
            break
        print(generator.generate(line, None))
        print("input:")