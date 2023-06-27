python generate_slot.py --base_path=./dev/   --output=./res/dev/ --fix_random_seed=True --search_width=10
python generate_slot.py --base_path=./train/   --output=./res/train/ --fix_random_seed=True --search_width=10 
python generate_intent.py --base_path=./train/   --output=./res/train/  --fix_random_seed=True --cover_filter=False  --random_generate=True 
python generate_intent.py --base_path=./dev/   --output=./res/dev/  --fix_random_seed=True --cover_filter=False  --random_generate=True 
