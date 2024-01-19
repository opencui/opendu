import unittest
import shutil

from opencui.inference.index import indexing
from opencui.inference.serve import load_converter_from_meta

class AgentTest(unittest.TestCase):
    converter = None

    @classmethod
    def setUpClass(clsc):
        root = "./examples/agent"
        #indexing(root)

        AgentTest.converter = load_converter_from_meta(root)

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
        self.assertTrue(result["dishes"] == "Pizza")


    def testInference(self):
        utterance = "I like to"
        questions = "Do you need more dish?"
        result = AgentTest.converter.inference(utterance, [questions])
        print(result)



if __name__ == "__main__":
    unittest.main()