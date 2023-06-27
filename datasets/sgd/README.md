> This is the data process from the SGD data to intent and slot data. Introduction about SGD,SGDX data could be found in https://github.com/google-research-datasets/dstc8-schema-guided-dialogue.

###  Data process from SGD to intent 

2.Create intent examples

There're some parameters for generating the intent examples.You need indicate the data path of the sgd data,such as ./train/ where the schema.json and dialogue info are included. you could also indicate the output path for the intent data. For generating the negative samples  we need to find the similar utterances in different intents. Here we use the faiss to  speed up this process.So we need to indicate the topk num that we 'd like to find for each utterance.The pos_num and neg_num illustrates seperately  the num of positive sample and negative sample for each intent. we could create the intent examples as follows:

```python
python generate_intent.py 
--base_path=./dev/   
--output=./res/dev/ 
--fix_random_seed=True
--cover_filter=False 
--random_generate=True 
```

3.Create slot examples

Generating the slot examples is similar .

```python
python generate_slot.py \
--base_path=./train/   \
--output=./train/ \
--fix_random_seed=True \
--search_width=10
```

4.Details on  dataset:

### Introduction of SGD dataset:

SGD data consists of two parts,schema and dialog.schema have the info of all services:intent and slot.The format of  the schema  in sgd is like this:

```JSON
 {
    "service_name": "Alarm_1",
    "description": "Manage alarms by getting and setting them easily",
    "entities": [
      {
        "name": "time",
        "description": "Time of the alarm",
        "is_categorical": false,
        "possible_values": []
      },
      {
        "name": "System.String",
        "description": "Name of the alarm",
        "is_categorical": false,
        "possible_values": []
      }
    ],
    "slots": [
      {
        "name": "alarm_time",
        "type": "time",
        "description": "Time of the alarm",
        "is_categorical": false,
        "possible_values": []
      },
      {
        "name": "alarm_name",
        "type": "System.String",
        "description": "Name of the alarm",
        "is_categorical": false,
        "possible_values": []
      },
      {
        "name": "new_alarm_time",
        "type": "time",
        "description": "Time to set for the new alarm",
        "is_categorical": false,
        "possible_values": []
      },
      {
        "name": "new_alarm_name",
        "type": "System.String",
        "description": "Name to use for the new alarm",
        "is_categorical": false,
        "possible_values": []
      }
    ],
    "intents": [
      {
        "name": "GetAlarms",
        "description": "Get the alarms user has already set",
        "is_transactional": false,
        "required_slots": [],
        "optional_slots": {}
      },
      {
        "name": "AddAlarm",
        "description": "Set a new alarm",
        "is_transactional": true,
        "required_slots": [
          "new_alarm_time"
        ],
        "optional_slots": {
          "new_alarm_name": "New alarm"
        }
      }
    ]
  }
```

As we can see, it contains all the intent and slot info for one certain service such as Alarm_1,while in the original dataset there's  no entity type .

The dialogue data is like:

 

```JSON
 {
    "dialogue_id": "1_00000",
    "services": [
      "Restaurants_1"
    ],
    "turns": [
      {
        "frames": [
          {
            "service": "Restaurants_1",
            "slots": [],
            "state": {
              "active_intent": "FindRestaurants",
              "requested_slots": [],
              "slot_values": {}
            }
          }
        ],
        "speaker": "USER",
        "utterance": "I am feeling hungry so I would like to find a place to eat."
      },
      {
        "frames": [
          {
            "actions": [
              {
                "act": "REQUEST",
                "slot": "city",
                "values": []
              }
            ],
            "service": "Restaurants_1",
            "slots": []
          }
        ],
        "speaker": "SYSTEM",
        "utterance": "Do you have a specific which you want the eating place to be located at?"
      },
      {
        "frames": [
          {
            "service": "Restaurants_1",
            "slots": [
              {
                "exclusive_end": 37,
                "slot": "city",
                "start": 29
              }
            ],
            "state": {
              "active_intent": "FindRestaurants",
              "requested_slots": [],
              "slot_values": {
                "city": [
                  "San Jose"
                ]
              }
            }
          }
        ],
        "speaker": "USER",
        "utterance": "I would like for it to be in San Jose."
      },
      ....
     {
        "frames": [
          {
            "actions": [
              {
                "act": "GOODBYE",
                "slot": "",
                "values": []
              }
            ],
            "service": "Restaurants_1",
            "slots": []
          }
        ],
        "speaker": "SYSTEM",
        "utterance": "You are welcome, have a great day"
      }
    ]
  },
```

"services"list all the services in the dialogue .The "turns" consists of  the dialogue info between "USER"  and "SYSTEM". Each utterance corresponds to one utterance.For each utterance ,there maybe more than one intent about it, so each frame in the "frames" matches one "activate_intent" We would like to use the "active_intent" item as the intent for each utterance to generate intent data . Use the "slot_values" in the frame to generate the slot data.



### Introduction of intent/slot dataset:

#### Intent data:

intent data consists of positive parts and negative parts, For each intent ,we need generate the positive utterance pair for it , just combine  the utterances in one intent  in pairs.For the intent data,we only focus on the utterance that brings the new intent .

For example .if the pre utterance of the "USER" has the intent "ReserveHotel", while the current  utterance has the intent  "ReserveHotel" and "SearchRoundtripFlights", then we think this utterance  if focused on the "SearchRoundtripFlights",  or we can say we just find the "rising edge" of the intent in the intent dataset  because the label of intent just indicates the state of which intent the utterance is contained .Here we ignored all "NONE" intent type.

The format of positive sample (need to be updated once figure out how to push to hugging face) :

```
SearchRoundtripFlights        1        I will be visiting a friend and I need to find a round trip flight. Find something to San Fran on today. Make the return flight on the 13th of March.        I am looking for an economy round trip flight from < origin city > to < destination city >. 
SearchRoundtripFlights        1        Can you find me a flight for one leaving from Washington?        I need to search for a round trip flight 
SearchRoundtripFlights        1        Can you find me a round trip flight?        I want to go for a trip to < destination city > and for which I need to search for round trip flights. Can you search for the one which is leaving from < origin city >? I would like to prefer < airlines >.
```

We need keep the original text of the first utterance , and as to the second utterance we need to replace the slot val in this utterance with the  slot name  ,such as   < destination city >.

 

The format of negative sample :

```
HotelFindWithLocale_OnlineHotelBooking        0        I would like to find an hotel room.        I would like to book a hotel please.  
HotelFindWithLocale_OnlineHotelBooking        0        Fine. I need to search a hotel there.        I need your help with something, my demand is to book a room at < name of accommodation >.  
HotelFindWithLocale_OnlineHotelBooking        0        Thanks, now can you find me 2 rooms in a Toronto hotel?        Please book < room number > rooms at this hotel.
```

The  negative sample is used to discriminate the utterance with different intent.We need to find the similar utterance from the different intent .Also we need to replace the slot val with slot name in the second utterance.

#### Slot data:

The slot data also consists of positive samples and negative samples , The positive samples label the slot val in the utterance like that:



```
I am leaving from San Diego to go to Fresno.        from_location        1        18, 26
I am leaving from San Diego to go to Fresno.        to_location        1        37, 42
```

The "from_location"  is the slot name and the slot val in the utterance is labeled like  18,26 which means the range of [18,26] in the string is the val of from_location.

The negative sample is like this:

```
I am leaving from San Diego to go to Fresno.        travelers        0        0, 0
```

In the negative sample  we need to label the slot name that does not have the exact val from the utterance. For example, the"travelers " item does not occur in the  utterance, so we add the "travelers" item as a negative sample for the utterance ,and the span range would be labeled as 0,0. 



### Unified intent

we could generate the unified intent with the span label by this:

```
python generate_intent_overall_nosetdelete_intent_alignment_single.py --base_path=./dev/   --output=./res/dev/  --fix_random_seed=True --cover_filter=True  --random_generate=True 
python generate_intent_overall_nosetdelete_intent_alignment_single.py  --base_path=./train/   --output=./res/train/  --fix_random_seed=True --cover_filter=True  --random_generate=True 
python ./intent_alignment_datagenerate_multi_alignment.py --base_path=./dev/   --output=./res/dev/  --fix_random_seed=True --cover_filter=True  --random_generate=True --pos_num=31240 
python ./intent_alignment_datagenerate_multi_alignment.py  --base_path=./train/   --output=./res/train/  --fix_random_seed=True --cover_filter=True  --random_generate=True  --pos_num=1122165   

```

The num of multi-utterance data is just 10% of the single utterance intent data.

