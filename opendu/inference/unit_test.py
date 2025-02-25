# Copyright (c) 2025 BeThere AI
# All rights reserved.
#
# This source code is licensed under the BeThere AI license.
# See LICENSE file in the project root for full license information.
import unittest

from opendu.inference.parser import load_parser
from opendu.inference.index import indexing

class AgentTest(unittest.TestCase):
    converter = None

    @classmethod
    def setUpClass(clsc):
        bot_path = "./examples/agent"
        indexing(bot_path)
        index_path = f"{bot_path}/index/"
        AgentTest.converter = load_parser(bot_path, index_path)

    @classmethod
    def tearDownClass(cls):\
        pass

    def testDetectTriggerable(self):
        utterance = "I like to order some food"
        result = AgentTest.converter.detect_triggerables(utterance, [])
        print(result)
        truth = "me.test.foodOrderingModule.FoodOrdering"
        self.assertTrue(len(result) == 1)
        self.assertTrue(result[0] == truth)

    def testFillSlots(self):
        utterance = "I like to order some Pizza"
        slots = [{"name": "dishes", "description": "dishes"}]
        candiates = {"dishes":["pizza", "apple"]}
        result = AgentTest.converter.fill_slots(utterance, slots, candiates)
        print(result)
        self.assertTrue(len(result) == 1)
        self.assertTrue(result["dishes"]["values"][0] == "Pizza")


    def testInference(self):
        utterance = "I like to"
        questions = "Do you need more dish?"
        result = AgentTest.converter.inference(utterance, [questions])
        print(result)



if __name__ == "__main__":
    unittest.main()